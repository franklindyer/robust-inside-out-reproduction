from collections import Counter, defaultdict
import csv
import json
import random
import re
import string
from unidecode import unidecode

import sys
sys.path.append("scripts")
from shared import *

SEED = 420
random.seed(SEED)

QS = load_all_questions()

for model_id in ["mistral-7b", "llama3-8b", "gemma2-9b"]:
    print(f"Loading sampled answers for {model_id}...")

    out_dir = f"data/probe_training/{model_id}"
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    gold_sampled_ids = set()

    # Get a list of IDs of questions for which the gold answer was sampled (up to string normalization)
    for batch in range(15):
        print(f"Batch {batch}")
        fname = f"./data/sampled_answers/{model_id}/{str(500*(1+batch)).zfill(5)}.json"
        answers_batch = json.loads(open(fname).read())
 
        for id in answers_batch:
            id = int(id)
            q = QS[id]
            
            q_cat = q["type"]
            gold_norm = normalize_str(q["answer"])
        
            for a in answers_batch[str(id)]["sampled"]:
                a_norm = normalize_str(a["text"])
                if a_norm == gold_norm:
                    gold_sampled_ids.add(id)
                    break
        
    with open(f"{out_dir}/gold_sampled.txt", 'w') as f:
        for id in gold_sampled_ids:
            f.write(f"{id}\n")
  
    # Build training data for "knowledge aware probing"
    # For each training question for which at least one correct answer was sampled,
    #   we include both the gold answer and one incorrect sampled answer 
    answer_opts = json.loads(open(f"data/answer_options/{model_id}/all.json").read())
    training_rows = []
    for s_id in answer_opts:
        id = int(s_id)
        q = QS[id]
        if not q["split"] == "train":
            continue

        sampled_correct = [r for r in answer_opts[s_id] if r["correct"] and r["source"] == "sampled"]
        sampled_incorrect = [r for r in answer_opts[s_id] if not r["correct"] and r["source"] == "sampled"]
        if (len(sampled_correct) > 0 or id in gold_sampled_ids) and len(sampled_incorrect) > 0:
            training_rows.append([
                str(id),
                QS[id]["source_entity"],
                QS[id]["question"],
                QS[id]["answer"], 
                sampled_incorrect[0]["text"].strip()
            ])
 
    with open(f"{out_dir}/probe_training.tsv", 'w') as f:
        for r in training_rows:
            f.write(f"{'\t'.join(r)}\n")
