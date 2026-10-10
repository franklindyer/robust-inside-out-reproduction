from collections import Counter, defaultdict
import csv
import json
import re

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
            if re.fullmatch("Q[0-9]+", src_col):
                continue
            data[src_col].append(r)
   
    data_list = []
    for k in data:
        if len(data[k]) == 1: # We only want questions with unique answers
            data_list.append(data[k][0])

    train_rows = []
    test_rows = []

    for r in data_list:
        source_entity = r[fdata["source_col"]]
        target_entity = r[fdata["target_col"]]
        if len(train_rows) < fdata["train"]:
            if not (source_entity in TEST_ENTITIES or target_entity in TEST_ENTITIES):
                train_rows.append(r)
        elif len(test_rows) < fdata["test"]:
            if not (source_entity in TRAIN_ENTITIES or target_entity in TRAIN_ENTITIES):
                test_rows.append(r)
        else:
            break

    if len(train_rows) < fdata["train"] or len(test_rows) < fdata["test"]:
        print(f"WARNING: {fdata['name']} data underpopulated with {len(train_rows)}/{len(test_rows)}")

    return (train_rows, test_rows)

GLOBAL_ID = 0

for source_data in SOURCES:
    train_rows, test_rows = load_wikidata(source_data)
    
    with open(f"data/splits/train/{source_data['filename']}", 'w') as f:
        for r in train_rows:
            question = source_data["tpl"].format(r[source_data["source_col"] + "Label"])
            answer = r[source_data["target_col"] + "Label"]
            f.write(f"{GLOBAL_ID}\t{question}\t{answer}\n")
            GLOBAL_ID += 1   
 
    with open(f"data/splits/test/{source_data['filename']}", 'w') as f:
        for r in test_rows:
            question = source_data["tpl"].format(r[source_data["source_col"] + "Label"])
            answer = r[source_data["target_col"] + "Label"]
            f.write(f"{GLOBAL_ID}\t{question}\t{answer}\n")
            GLOBAL_ID += 1
