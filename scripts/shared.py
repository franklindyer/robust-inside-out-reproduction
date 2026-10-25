from collections import Counter, defaultdict
import csv
import gc
import glob
import json
import os
import string
import tqdm
from unidecode import unidecode

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, BitsAndBytesConfig

torch.manual_seed(67)

def normalize_str(s):
    s = s.strip()
    s = s.lower()
    s = unidecode(s)
    s = s.translate(str.maketrans('', '', string.punctuation)) 
    return s

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

        # Answer sampled > 1 times by the specific model
        ans_opts += [
            {
                "text": sampled["text"].strip(),
                "source": "sampled"
            }
            for sampled in sampled_batch[str(q["id"])]["sampled"]
            if sampled["count"] > 1
        ]

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
                "text": ans_text.strip(), 
                "source": "train_sampled"
            }
            for ans_text in train_sampled[model_id][q["type"]]
        ]

        ans_opts_dedup = []
        ans_raw = set()

        for ans_opt in ans_opts:
            if ans_opt["text"] in ans_raw:
                continue
            ans_opts_dedup.append(ans_opt)
            ans_raw.add(ans_opt["text"])

        return ans_opts_dedup

    return answer_options_list

def answer_options_gen_for_model(model_id, n_batches=15, batch_size=500):
    from qa_metrics.RewardBert import RewardBert

    qs = load_all_questions()
    r_bert = RewardBert(device='cuda')    

    out_dir = f"data/answer_options/{model_id}"
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    all_answer_opts = {}
    for batch_id in range(n_batches):
        get_options_list = answer_options_gen_for_batch(model_id, batch_id, batch_size=batch_size)
        print(f"Generating batch {batch_id} of {n_batches}...")
        for q_id in tqdm.tqdm(range(batch_size*batch_id, batch_size*(batch_id+1))):
            q = qs[q_id]
            answer_options = get_options_list(q)
            gold_text = q["answer"]

            norm_answers_marked_correct = set()
            for i in range(len(answer_options)):
                ans_text = answer_options[i]["text"]
                is_gold = answer_options[i]["source"] == "gold"
                is_correct = is_gold or (r_bert.compute_score(gold_text, ans_text)[0] > 0.5)
                answer_options[i]["correct"] = is_correct 
                if is_correct:
                    norm_answers_marked_correct.add(normalize_str(ans_text))

            # Answer correctness should be "indifferent" to string normalization
            for i in range(len(answer_options)):
                ans_text = answer_options[i]["text"]
                norm_text = normalize_str(ans_text)
                if norm_text in norm_answers_marked_correct:
                    answer_options[i]["correct"] = True 

            all_answer_opts[q_id] = answer_options
 
    with open(f"{out_dir}/all.json", 'w') as f:
        f.write(json.dumps(all_answer_opts, indent=4, ensure_ascii=False))

def next_token_probs_ab(model, tokenizer, prompt):
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

    # Note: we cannot directly get the output probabilities because the tokenizer is not a bijective
    #    mapping, i.e. multiple IDs may correspond to the strings "A" and "B"
    result = {
        "a_logit": None,
        "a_prob": None,
        "b_logit": None,
        "b_prob": None,
    }

    logits = outputs.logits[0].float()
    row = logits[0]
    top = torch.topk(row, k=10)
    probs = torch.softmax(row, dim=-1)[top.indices]
    
    for (tid, lg, p) in zip(top.indices.tolist(), top.values.tolist(), probs.tolist()):
        tok_str = tokenizer.decode([tid])
        if tok_str == "A" and result["a_logit"] is None:
            result["a_logit"] = lg
            result["a_prob"] = p
        elif tok_str == "B" and result["b_logit"] is None:
            result["b_logit"] = lg
            result["b_prob"] = p

    return result

def generate_tf_answer_batches(model_id, num_batches=15, batch_size=500):
    MODEL_INFO = json.loads(open("data/models.json").read())
    PROMPT_TPL = open("data/templates/verification.txt", 'r').read()
    PROMPT_REV_TPL = open("data/templates/verification_reversed.txt", 'r').read()
    ALL_QS = load_all_questions()
    ALL_ANS = json.loads(open(f"data/answer_options/{model_id}/all.json").read())

    model_hug_id = MODEL_INFO[model_id]["huggingface"]
    tokenizer = AutoTokenizer.from_pretrained(model_hug_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_hug_id,
        dtype=torch.bfloat16, 
        device_map="auto"
    )
    model.eval()

    for batch_num in range(num_batches):
        print(f"Generating batch {batch_num} of {num_batches}...")
        
        out_dirname = f"data/tf_answers/{model_id}"
        out_fname = f"{out_dirname}/{str((batch_num+1)*batch_size).zfill(5)}.json"
        if not os.path.isdir(out_dirname):
            os.mkdir(out_dirname)
    
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
            for ans_opt in ALL_ANS[str(i)]:
                prompt = PROMPT_TPL.format(q["question"], ans_opt["text"])
                ab_probs = next_token_probs_ab(model, tokenizer, prompt)
 
                prompt_rev = PROMPT_REV_TPL.format(q["question"], ans_opt["text"])
                ab_rev_probs = next_token_probs_ab(model, tokenizer, prompt_rev) 
          
                ans_results.append({
                    "original": ab_probs,
                    "reversed": ab_rev_probs
                })
    
            results[i] = ans_results
    
        with open(out_fname, 'w') as f:
            f.write(json.dumps(results, indent=4, ensure_ascii=False))

def extract_hidden_states(model, tokenizer, prompts, layer_inds):
    layers = {i: [] for i in layer_inds}
    
    for prompt in tqdm.tqdm(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)

        for i in layer_inds:
            hidden_layer = outputs.hidden_states[i].float()  
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden_layer * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
            layers[i].append(pooled.cpu())

    for i in layers:
        layers[i] = torch.cat(layers[i], dim=0)

    return layers

def extract_hidden_states_batch(model, tokenizer, prompts, layer_inds):
    layers = {i: [] for i in layer_inds}
    
    MAX_LEN = 256
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
    ).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    for i in layer_inds:
        hidden_layer = outputs.hidden_states[i].float()  
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden_layer * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        layers[i] = pooled.cpu()

    return layers

def make_probe_training_vectors(model_id, layer_inds=[0, 24]):
    MODEL_INFO = json.loads(open("data/models.json").read())
    PROMPT_TPL = open("data/templates/verification.txt", 'r').read()
    out_dir = f"data/probe_training/{model_id}"

    model_hug_id = MODEL_INFO[model_id]["huggingface"]
    tokenizer = AutoTokenizer.from_pretrained(model_hug_id)
    model = AutoModel.from_pretrained(
        model_hug_id,
        dtype=torch.bfloat16, 
        device_map="auto"
    )
    model.eval()

    prompts_correct = []
    prompts_incorrect = []
    with open(f"{out_dir}/probe_training.tsv", 'r') as f:
        reader = csv.reader(f, delimiter="\t")
        for r in reader:
            q_id = int(r[0])
            q_text = r[2]
            a_correct = r[3]
            a_incorrect = r[4]

            prompts_correct.append(PROMPT_TPL.format(q_text, a_correct)) 
            prompts_incorrect.append(PROMPT_TPL.format(q_text, a_incorrect))

    h_correct = extract_hidden_states(model, tokenizer, prompts_correct, layer_inds)
    h_incorrect = extract_hidden_states(model, tokenizer, prompts_incorrect, layer_inds)

    torch.save({
        "correct": h_correct,
        "incorrect": h_incorrect
    }, f"{out_dir}/tensors.pt")

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        out = self.linear(x)
        return torch.sigmoid(out)

class HiddenLayersData(Dataset):
    def __init__(self, fname, layer_num):
        all_data = torch.load(fname)
        v_correct = all_data["correct"][layer_num]
        v_incorrect = all_data["incorrect"][layer_num]
        v_all = torch.cat([v_correct, v_incorrect], dim=0)
        self.features = v_all

        self.labels = torch.tensor(
            [1]*v_correct.shape[0] + [0]*v_incorrect.shape[0], 
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        features = self.features[index, :]
        target = self.labels[index]
        return features, target

def train_probe_with_baseline(model_id, target_layer=24, hidden_dim=4096, epochs=2000):
    out_dir = f"data/probe_scores/{model_id}"
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    
    h_emb = HiddenLayersData(f"data/probe_training/{model_id}/tensors.pt", 0)
    h_l24 = HiddenLayersData(f"data/probe_training/{model_id}/tensors.pt", target_layer)
    dl_emb = DataLoader(h_emb, batch_size=32, shuffle=True)
    dl_l24 = DataLoader(h_l24, batch_size=32, shuffle=True)

    mus_emb = h_emb.features.mean(dim=1, keepdim=True)
    sds_emb = h_emb.features.std(dim=1, keepdim=True)
    mus_l24 = h_l24.features.mean(dim=1, keepdim=True)
    sds_l24 = h_l24.features.std(dim=1, keepdim=True)

    model_emb = LogisticRegression(hidden_dim, 1)
    model_l24 = LogisticRegression(hidden_dim, 1)

    criterion = nn.BCELoss()
    optimizer_emb = torch.optim.SGD(model_emb.parameters(), lr=0.01) 
    optimizer_l24 = torch.optim.SGD(model_l24.parameters(), lr=0.01)

    losses_emb = []
    losses_l24 = []

    for epoch in tqdm.tqdm(range(epochs)):
        epoch_loss_emb = 0
        epoch_loss_l24 = 0
        
        for i, ((x_emb, y_emb), (x_l24, y_l24)) in enumerate(zip(dl_emb, dl_l24)):
            # Normalize data
            x_emb = (x_emb - mus_emb)/sds_emb
            x_l24 = (x_l24 - mus_l24)/sds_l24

            y_pred_emb = model_emb(x_emb) 
            y_pred_emb = torch.squeeze(y_pred_emb)

            y_pred_l24 = model_l24(x_l24) 
            y_pred_l24 = torch.squeeze(y_pred_l24)
            
            loss_emb = criterion(y_pred_emb, y_emb)
            epoch_loss_emb += loss_emb
            
            loss_l24 = criterion(y_pred_l24, y_l24)
            epoch_loss_l24 += loss_l24
            
            loss_emb.backward()
            loss_l24.backward()
            
            optimizer_emb.step()
            optimizer_l24.step()
            
            optimizer_emb.zero_grad() 
            optimizer_l24.zero_grad() 

        losses_emb.append(epoch_loss_emb.float())
        losses_l24.append(epoch_loss_l24.float())
        if epoch % 50 == 0:
            print(f"EPOCH {epoch}")
            print(f"EMB model loss: {epoch_loss_emb}")
            print(f"L24 model loss: {epoch_loss_l24}")

    y_pred_emb = torch.squeeze(model_emb(h_emb.features))
    y_pred_l24 = torch.squeeze(model_l24(h_l24.features))
    acc_emb = torch.sum(y_pred_emb.round() == h_emb.labels)/len(h_emb.labels)
    acc_l24 = torch.sum(y_pred_l24.round() == h_l24.labels)/len(h_l24.labels)

    with open(f"{out_dir}/training_stats.tsv", 'w') as f:
        f.write(f"epoch\tbaseline\thidden\n")
        for i, (loss_emb, loss_l24) in enumerate(zip(losses_emb, losses_l24)):
            f.write(f"{i}\t{loss_emb}\t{loss_l24}\n")
    
    with open(f"{out_dir}/final_training_acc.txt", 'w') as f:
        f.write(f"TRAINING COMPLETE\n")
        f.write(f"Number of epochs: {epochs}\n")
        f.write(f"Training accuracy of embedding baseline: {acc_emb}\n")
        f.write(f"Training accuracy of layer-{target_layer} probe: {acc_l24}\n")

    return mus_l24, sds_l24, model_l24


def calculate_probe_scores(model_id, mus, sds, lin_probe, target_layer=24):
    MODEL_INFO = json.loads(open("data/models.json").read())
    PROMPT_TPL = open("data/templates/verification.txt", 'r').read()
    out_dir = f"data/probe_scores/{model_id}"

    model_hug_id = MODEL_INFO[model_id]["huggingface"]
    tokenizer = AutoTokenizer.from_pretrained(model_hug_id)
    model = AutoModel.from_pretrained(
        model_hug_id,
        dtype=torch.bfloat16, 
        device_map="auto"
    )
    model.eval()

    QS = load_all_questions()
    ANS_OPTS = json.loads(open(f"data/answer_options/{model_id}/all.json").read())
    TF_TPL = open("data/templates/verification.txt").read()

    test_qids = list([q_id for q_id in QS if QS[q_id]["split"] == "test"])
    all_scores = { q_id: [] for q_id in test_qids }

    for q_id in tqdm.tqdm(test_qids):
        q = QS[q_id]
        ans_opts = ANS_OPTS[str(q_id)]
        prompts = [TF_TPL.format(q["question"], ans_opt["text"]) for ans_opt in ans_opts]
        
        h_reps = extract_hidden_states_batch(model, tokenizer, prompts, [target_layer])
        h_reps = h_reps[target_layer]
        h_reps = (h_reps - mus)/sds
        probe_scores = torch.squeeze(lin_probe(h_reps)).tolist()
        all_scores[q_id] = probe_scores

    with open(f"{out_dir}/all.json", 'w') as f:
        f.write(json.dumps(all_scores, indent=4, ensure_ascii=False))

def train_and_score_probe(model_id, target_layer=24, hidden_dim=4096, epochs=2000):
    mus, sds, lin_probe = train_probe_with_baseline(
        model_id, 
        target_layer=target_layer, 
        hidden_dim=hidden_dim, 
        epochs=epochs
    )
    calculate_probe_scores(
        model_id,
        mus,
        sds,
        lin_probe,
        target_layer=target_layer
    )
