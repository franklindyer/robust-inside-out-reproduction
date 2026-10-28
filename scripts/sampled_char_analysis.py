from collections import Counter, defaultdict
import json
import os
import sys
import tqdm
import unicodeblock.blocks

model_id = sys.argv[1]

sampled_counts = {}

many_total = 0
once_total = 1
many_sampled_chars = Counter()
once_sampled_chars = Counter()

sampled_lengths = defaultdict(lambda: [])

data_dir = f"data/sampled_answers/{model_id}" 
for fname in os.listdir(data_dir):
    if fname[0] == ".":
        continue
    sampled = json.loads(open(f"{data_dir}/{fname}").read())
    for qid in tqdm.tqdm(list(sampled.keys())):
        count_counts = Counter([ans["count"] for ans in sampled[qid]["sampled"]])
        sampled_counts[int(qid)] = dict(count_counts.most_common())

        many_sampled_lengths = []
        once_sampled_lengths = []
        for ans in sampled[qid]["sampled"]:
            uniq_chars = set([unicodeblock.blocks.of(c) for c in ans["text"].strip()])
            if ans["count"] == 1:
                once_sampled_chars.update(uniq_chars)
                once_sampled_lengths.append(len(ans["text"]))
                once_total += 1
            else:
                many_sampled_chars.update(uniq_chars)
                many_sampled_lengths.append(len(ans["text"]))
                many_total += 1

        sampled_lengths[int(qid)].append({
            "many": many_sampled_lengths,
            "once": once_sampled_lengths 
        })        

    print(fname)

out_dir = f"data/analysis_data/{model_id}"
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

with open(f"{out_dir}/sampled_counts.json", 'w') as f:
    f.write(json.dumps(sampled_counts, indent=4, ensure_ascii=False))

with open(f"{out_dir}/sampled_blocks.json", 'w') as f:
    f.write(json.dumps({
        "many_total": many_total,
        "once_total": once_total,
        "many": dict(many_sampled_chars.most_common()),
        "once": dict(once_sampled_chars.most_common())
    }, indent=4, ensure_ascii=False))

with open(f"{out_dir}/sampled_lengths.json", 'w') as f:
    f.write(json.dumps(sampled_lengths, indent=4, ensure_ascii=False))
