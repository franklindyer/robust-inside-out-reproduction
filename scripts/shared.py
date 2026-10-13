from collections import Counter, defaultdict
import csv
import gc
import glob
import json
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import tqdm

torch.manual_seed(67)

def live_cuda_tensors_gib():
    n, total = 0, 0
    for o in gc.get_objects():
        try:
            if isinstance(o, torch.Tensor) and o.is_cuda:
                n += 1
                total += o.numel() * o.element_size()
        except ReferenceError:
            continue
    return n, total / 2**30

def load_all_questions():
    q_dict = {}
    for subdir in ["train", "test", "dev"]:
        for fname in glob.glob(f"data/splits/{subdir}/*.tsv"):
            with open(fname, 'r') as f:
                reader = csv.reader(f, delimiter="\t")
                for r in reader:
                    id = int(r[0])
                    q_dict[id] = {
                        "id": id,
                        "question": r[1],
                        "answer": r[2]
                    }
    return q_dict

def sample_model_outputs(model, tokenizer, prompt, sys_prompt="", n=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            num_return_sequences=n,
            max_new_tokens=64,
            temperature=1,
            top_p=0.95,
            num_beams=1,
            stop_strings=["\n"],
            tokenizer=tokenizer,
            pad_token_id=tokenizer.eos_token_id,
            renormalize_logits=False,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True
        )

    prompt_len = inputs.input_ids.shape[1]
    gen = outputs.sequences[:, prompt_len:]

    normalized_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    ).float()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    mask  = (gen != pad_id)
    normalized_scores = normalized_scores.masked_fill(~mask, 0.0)
    lengths = (normalized_scores < 0).sum(dim=1).clamp(min=1)
    norm_logprob = normalized_scores.sum(dim=1)
    unnorm_prob = torch.exp(norm_logprob)
    norm_prob = torch.exp(norm_logprob / lengths)

    texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
    out_counter = Counter(texts)
    
    sampled_map = {}
    for txt, l, norm_p, unnorm_p in zip(texts, lengths, norm_prob, unnorm_prob):
        if not txt in sampled_map:
            sampled_map[txt] = {
                "text": txt,
                "length": l.item(),
                "count": out_counter[txt],
                "p": unnorm_p.item(),
                "p_norm": norm_p.item(),
            }

    del gen
    del normalized_scores
    del mask
    del lengths
    del norm_logprob
    del unnorm_prob
    del norm_prob
    del texts

    sampled_list = sorted(list(sampled_map.values()), key=lambda x: -x["count"])
    return sampled_list

def greedy_model_output(model, tokenizer, prompt, sys_prompt=""):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=64,
            temperature=1,
            top_p=0.95,
            stop_strings=["\n"],
            tokenizer=tokenizer,
            pad_token_id=tokenizer.eos_token_id,
            renormalize_logits=False,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True
        )

    prompt_len = inputs.input_ids.shape[1]
    gen = output.sequences[0, prompt_len:]

    normalized_scores = model.compute_transition_scores(
        output.sequences, output.scores, normalize_logits=True
    ).float()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    mask  = (gen != pad_id)
    normalized_scores = normalized_scores.masked_fill(~mask, 0.0)
    lengths = (normalized_scores < 0).sum(dim=1).clamp(min=1)
    norm_logprob = normalized_scores.sum(dim=1)
    unnorm_prob = torch.exp(norm_logprob)
    norm_prob = torch.exp(norm_logprob / lengths)

    txt = tokenizer.decode(gen, skip_special_tokens=True)

    result = {
        "text": txt,
        "length": lengths.item(),
        "p": unnorm_prob.item(),
        "p_norm": norm_prob.item(),
    }

    del gen
    del normalized_scores
    del mask
    del lengths
    del norm_logprob
    del unnorm_prob
    del norm_prob
    del txt

    return result

def greedy_and_sampled(model, tokenizer, prompt, sys_prompt=""):
    greedy_ans = greedy_model_output(model, tokenizer, prompt, sys_prompt=sys_prompt)
    sampled_ans = sample_model_outputs(model, tokenizer, prompt, sys_prompt=sys_prompt, n=1000)
    return {
        "greedy": greedy_ans,
        "sampled": sampled_ans
    }

# We use a fixed batch size of 500 for these runs
def generate_answer_sample_batch(model_id, batch_num, batch_size=500):
    MODEL_INFO = json.loads(open("data/models.json").read())
    PROMPT_TPL = open("data/templates/sampling.txt", 'r').read() + "\n" # Add a trailing newline, since models tend to add it themselves otherwise
    ALL_QS = load_all_questions()

    out_dirname = f"data/sampled_answers/{model_id}"
    out_fname = f"{out_dirname}/{str((batch_num+1)*batch_size).zfill(5)}.json"
    if not os.path.isdir(out_dirname):
        os.mkdir(out_dirname)

    model_hug_id = MODEL_INFO[model_id]["huggingface"]
    tokenizer = AutoTokenizer.from_pretrained(model_hug_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_hug_id, torch_dtype=torch.bfloat16, device_map="cuda"
    )

    results = {}
    for i in tqdm.tqdm(range(batch_size*batch_num, batch_size*(batch_num+1))):
        if not i in ALL_QS:
            continue
        q = ALL_QS[i]
        prompt = PROMPT_TPL.format(q["question"])
        results[i] = greedy_and_sampled(model, tokenizer, prompt)

        if i % 5 == 0:
            n, gib = live_cuda_tensors_gib()
            print(f"{i:5d}  alloc={torch.cuda.memory_allocated()/2**30:6.2f} GiB  "
                  f"reserved={torch.cuda.memory_reserved()/2**30:6.2f} GiB  "
                  f"live_tensors={n} ({gib:.2f} GiB)")        

    with open(out_fname, 'w') as f:
        f.write(json.dumps(results, indent=4, ensure_ascii=False))
