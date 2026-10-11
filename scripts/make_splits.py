from collections import Counter, defaultdict
import csv
import json
import random
import re

SEED = 420
random.seed(SEED)

SOURCES = json.loads(open("data/split.json").read())["sources"]

TRAIN_ENTITIES = set()
TEST_ENTITIES = set()

def load_wikidata(fdata):
    fname = fdata["filename"]
    full_fname = f"./data/wikidata/{fname}"
    data = defaultdict(lambda: [])
    
    with open(full_fname, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for r in reader:
            src_col = r[fdata["source_col"] + "Label"]
            tgt_col = r[fdata["target_col"] + "Label"]
            if re.fullmatch("Q[0-9]+", src_col):
                continue
            if re.fullmatch("Q[0-9]+", tgt_col):
                continue
            data[src_col].append(r)
   
    data_list = []
    for k in data:
        if len(data[k]) == 1: # We only want questions with unique answers
            data_list.append(data[k][0])

    random.shuffle(data_list)

    train_rows = []
    test_rows = []

    for r in data_list:
        source_entity = r[fdata["source_col"]]
        target_entity = r[fdata["target_col"]]
        if len(train_rows) < fdata["train"]:
            if not (source_entity in TEST_ENTITIES or target_entity in TEST_ENTITIES):
                train_rows.append(r)
                TRAIN_ENTITIES.add(source_entity)
                TRAIN_ENTITIES.add(target_entity)
        elif len(test_rows) < fdata["test"]:
            if not (source_entity in TRAIN_ENTITIES or target_entity in TRAIN_ENTITIES):
                test_rows.append(r)
                TEST_ENTITIES.add(source_entity)
                TEST_ENTITIES.add(target_entity)
        else:
            break

    if len(train_rows) < fdata["train"] or len(test_rows) < fdata["test"]:
        print(f"WARNING: {fdata['name']} data underpopulated with {len(train_rows)}/{len(test_rows)}")

    return (train_rows, test_rows)

GLOBAL_ID = 0

for source_data in SOURCES:
    train_rows, test_rows = load_wikidata(source_data)

    n_test = len(test_rows)
    n_dev = n_test // 10
    dev_rows = test_rows[:n_dev]
    test_rows = test_rows[n_dev:] 
   
    for dirname, data_rows in [("train", train_rows), ("test", test_rows), ("dev", dev_rows)]: 
        with open(f"data/splits/{dirname}/{source_data['filename']}", 'w') as f:
            for r in data_rows:
                question = source_data["tpl"].format(r[source_data["source_col"] + "Label"])
                answer = r[source_data["target_col"] + "Label"]
                f.write(f"{GLOBAL_ID}\t{question}\t{answer}\n")
                GLOBAL_ID += 1   
