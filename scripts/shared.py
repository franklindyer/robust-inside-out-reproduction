from collections import Counter, defaultdict
import csv
import gc
import glob
import json
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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

def load_all_questions(folder="data/splits"):
    q_dict = {}
    for subdir in ["train", "test", "dev"]:
        for fname in glob.glob(f"{folder}/{subdir}/*.tsv"):
            with open(fname, 'r') as f:
                qtype = os.path.basename(fname)[:-4]
                reader = csv.reader(f, delimiter="\t")
                for r in reader:
                    id = int(r[0])
                    q_dict[id] = {
                        "id": id,
                        "split": subdir,
                        "type": qtype,
                        "question": r[1],
                        "answer": r[2],
                        "source_entity": r[3]
                    }
    return q_dict

def sample_model_outputs(model, tokenizer, prompt, sys_prompt="", n=100, n_batch=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    sampled_map = {}

    stop_ids = [tokenizer.eos_token_id]
    if tokenizer.pad_token_id is not None:
        stop_ids.append(tokenizer.pad_token_id)
    def seq_key(ids):
        ids = ids.tolist()
        for i, t in enumerate(ids):
            if t in stop_ids:
                return tuple(ids[:i])
        return tuple(ids)

    # Run the samples in minibatches, since doing them all at once gives tensors that are too large
    for batch_num in range(n // n_batch):
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                num_return_sequences=n_batch,
                max_new_tokens=64,
                temperature=1,
                top_p=1,
                top_k=0,
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

        texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
        out_counter = Counter(texts)
        
        for seq, txt, norm_scores in zip(gen, texts, normalized_scores):
            key = seq_key(seq)
            if not key in sampled_map:
                sampled_map[key] = {
                    "text": txt,
                    "tokens": key,
                    "count": 0,
                    "normalized_scores": norm_scores.tolist()[:len(key)]
                }
            sampled_map[key]["count"] += 1
    
    sampled_list = sorted(list(sampled_map.values()), key=lambda x: -x["count"])
    return sampled_list

def greedy_model_output(model, tokenizer, prompt, sys_prompt=""):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    stop_ids = [tokenizer.eos_token_id]
    if tokenizer.pad_token_id is not None:
        stop_ids.append(tokenizer.pad_token_id)
    def seq_key(ids):
        ids = ids.tolist()
        for i, t in enumerate(ids):
            if t in stop_ids:
                return tuple(ids[:i])
        return tuple(ids)
    
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=64,
            temperature=1,
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

    txt = tokenizer.decode(gen, skip_special_tokens=True)
    key = seq_key(gen)
    
    result = {
        "text": txt,
        "key": key,
        "normalized_scores": normalized_scores.tolist()[0][:len(key)]
    }
    
    return result

def greedy_and_sampled(model, tokenizer, prompt, sys_prompt=""):
    greedy_ans = greedy_model_output(model, tokenizer, prompt, sys_prompt=sys_prompt)
    sampled_ans = sample_model_outputs(model, tokenizer, prompt, sys_prompt=sys_prompt, n=1000, n_batch=100)
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

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )

    model_hug_id = MODEL_INFO[model_id]["huggingface"]
    tokenizer = AutoTokenizer.from_pretrained(model_hug_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_hug_id,
        dtype=torch.bfloat16, 
        device_map="auto"
    )
    model.eval()

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

def load_sampled_answers(folder_path):
    j_full = {}
    for fname in os.listdir(folder_path):
        j = json.loads(open(f"{folder_path}/{fname}", 'r').read())
        j_full.update(j)
    return j_full

def answer_options_gen_for_batch(model_id, batch_num, batch_size=500):
    train_gold = json.loads(open("data/selected_answers/train_gold.json").read())
    train_sampled = json.loads(open("data/selected_answers/train_commonly_sampled.json").read())

    sampled_fname = f"./data/sampled_answers/{model_id}/{str(500*(1+batch_num)).zfill(5)}.json"
    sampled_batch = json.loads(open(sampled_fname).read())

    def answer_options_list(q):
        ans_opts = []

        # Gold answer
        ans_opts.append({
            "text": q["answer"],
            "source": "gold"
        })

        # Source entity distractor
        ans_opts.append({
            "text": q["source_entity"],
            "source": "source_entity"
        })

        # Training gold answers
        ans_opts += [
            { 
                "text": ans_text, 
                "source": "train_gold"
            }
            for ans_text in train_gold[q["type"]]
        ]

        # Training commonly sampled answers
        ans_opts += [
            { 
                "text": ans_text, 
                "source": "train_sampled"
            }
            for ans_text in train_sampled[model_id][q["type"]]
        ]

        # Answer sampled > 1 times by the specific model
        ans_opts += [
            {
                "text": sampled["text"],
                "source": "sampled"
            }
            for sampled in sampled_batch[str(q["id"])]["sampled"]
            if sampled["count"] > 1
        ]

        return ans_opts

    return answer_options_list

def next_token_probs_ab(model, tokenizer, prompt):
    a_id = tokenizer.encode("A", add_special_tokens=False)[0]
    b_id = tokenizer.encode("B", add_special_tokens=False)[0]
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=1,
            temperature=1,
            num_beams=1,
            tokenizer=tokenizer,
            pad_token_id=tokenizer.eos_token_id,
            renormalize_logits=False,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True
        )

    logits = outputs.logits[0].float()
    row = logits[0]
    top = torch.topk(row, k=10)
    probs = torch.softmax(row, dim=-1)

    return {
        "a_logit": row[a_id].item(),
        "a_prob": probs[a_id].item(),
        "b_logit": row[b_id].item(),
        "b_prob": probs[b_id].item(),
    }

def generate_tf_answer_batch(model_id, batch_num, batch_size=500):
    MODEL_INFO = json.loads(open("data/models.json").read())
    PROMPT_TPL = open("data/templates/verification.txt", 'r').read()
    ALL_QS = load_all_questions()

    out_dirname = f"data/tf_answers/{model_id}"
    out_fname = f"{out_dirname}/{str((batch_num+1)*batch_size).zfill(5)}.json"
    if not os.path.isdir(out_dirname):
        os.mkdir(out_dirname)

    model_hug_id = MODEL_INFO[model_id]["huggingface"]
    tokenizer = AutoTokenizer.from_pretrained(model_hug_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_hug_id,
        dtype=torch.bfloat16, 
        device_map="auto"
    )
    model.eval()

    results = {}
    opts_gen = answer_options_gen_for_batch(model_id, batch_num)
    for i in tqdm.tqdm(range(batch_size*batch_num, batch_size*(batch_num+1))):
        if not i in ALL_QS:
            continue
        q = ALL_QS[i]
        # Only applies to test set
        if not q["split"] == "test":
            continue

        ans_results = []
        for ans_opt in opts_gen(q):
            prompt = PROMPT_TPL.format(q["question"], ans_opt["text"].strip())
            ab_probs = next_token_probs_ab(model, tokenizer, prompt) 
            ans_opt["ab_probs"] = ab_probs
            ans_results.append(ans_opt)

        results[i] = ans_results

    with open(out_fname, 'w') as f:
        f.write(json.dumps(results, indent=4, ensure_ascii=False))


