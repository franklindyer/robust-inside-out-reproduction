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

def normalize_ans(s):
    s = s.strip()
    s = unidecode(s)
    s = s.translate(string.punctuation)
    return s

SEED = 420
random.seed(SEED)

QS = load_all_questions()

train_gold = defaultdict(lambda: set())
test_gold = defaultdict(lambda: set())

for id in QS:
    q = QS[id]
    if q["split"] == "train":
        train_gold[q["type"]].add(q["answer"])
    elif q["split"] == "test":
        test_gold[q["type"]].add(q["answer"])

for q_cat in train_gold:
    train_gold[q_cat] = list(random.sample(list(train_gold[q_cat]), k=20))

for q_cat in test_gold:
    test_gold[q_cat] = list(test_gold[q_cat])
    random.shuffle(test_gold[q_cat])

with open("data/selected_answers/train_gold.json", 'w') as f:
    f.write(json.dumps(train_gold, indent=4, ensure_ascii=False))
    
with open("data/selected_answers/test_gold.json", 'w') as f:
    f.write(json.dumps(test_gold, indent=4, ensure_ascii=False))

commonly_sampled_by_model = {}

for model_id in ["mistral-7b", "llama3-8b", "gemma2-9b"]:
    print(f"Loading sampled answers for {model_id}...")
    # answers = load_sampled_answers(f"./data/sampled_answers/{model_id}")

    train_sampled_by_cat = defaultdict(lambda: Counter())
    answers_unnorm = defaultdict(lambda: Counter())

    for batch in range(15):
        print(f"Batch {batch}")
        fname = f"./data/sampled_answers/{model_id}/{str(500*(1+batch)).zfill(5)}.json"
        answers_batch = json.loads(open(fname).read())
 
        # for id in tqdm.tqdm(range(7500)):
        for id in answers_batch:
            id = int(id)
            if not QS[id]["split"] == "train":
                continue
            q_cat = QS[id]["type"]

            for a in answers_batch[str(id)]["sampled"]:
                if a["count"] <= 1:
                    continue
                a_norm = normalize_ans(a["text"])
                answers_unnorm[a_norm][a["text"]] += 1
                train_sampled_by_cat[q_cat][a_norm] += 1
        
        commonly_sampled_by_model[model_id] = {
            q_cat: [
                answers_unnorm[ans_norm].most_common()[0][0].strip() 
                for ans_norm, _ in train_sampled_by_cat[q_cat].most_common()[:50]
            ]
            for q_cat in train_sampled_by_cat 
        }

with open("data/selected_answers/train_commonly_sampled.json", 'w') as f:
    f.write(json.dumps(commonly_sampled_by_model, indent=4, ensure_ascii=False))

