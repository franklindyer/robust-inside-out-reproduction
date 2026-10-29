import json
import sys

sys.path.append("scripts")
from shared import *

fname = sys.argv[1]

QS = load_all_questions()

rewardbert_reflexivity = {}

ans_opts = json.loads(open(fname).read())
for sid in ans_opts:
    if ans_opts[sid][0]["source"] == "gold" and not ans_opts[sid][0]["correct"]:
        rewardbert_reflexivity[sid] = {
            "subject": QS[int(sid)]["source_entity"], 
            "object": QS[int(sid)]["answer"], 
            "relation": QS[int(sid)]["type"], 
        }

with open(f"data/analysis_data/rewardbert.json", 'w') as f:
    f.write(json.dumps({
        "reflexivity_violation": rewardbert_reflexivity
    }, indent=4, ensure_ascii=False))
