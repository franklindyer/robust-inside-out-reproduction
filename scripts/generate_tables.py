import jinja2
import json

JLOAD = jinja2.FileSystemLoader(searchpath="./templates/")
JENV = jinja2.Environment(loader=JLOAD)

### AVERAGE K-VALUE TABLES ###

k_avgs = json.loads(open("data/analysis_data/k_tests.json").read())
k_avgs = { tuple(k.split("|")): k_avgs["avgs"][k] for k in k_avgs["avgs"] }

### ANALYSIS OF ONCE-SAMPLED ANSWERS ###

sampled_blocks = json.loads(open("data/analysis_data/llama3-8b/sampled_blocks.json").read())
sampled_blocks_res = [
    {
        "name": blk_name,
        "once_prob": sampled_blocks["once"][blk_name]/sampled_blocks["once_total"], 
        "many_prob": (sampled_blocks["many"].get(blk_name) or 0)/sampled_blocks["many_total"]
    }
    for blk_name in sampled_blocks["once"]
    if not blk_name == "null"
]
for r in sampled_blocks_res:
    r["per_dec"] = 1 - r["many_prob"]/r["once_prob"]
sampled_blocks_res.sort(key=lambda r: -r["per_dec"]*r["once_prob"])
sampled_blocks_res = sampled_blocks_res[:15]

tbl_tpl = JENV.get_template("tables.tpl")
tbl_out = tbl_tpl.render(
    k_avgs = k_avgs,
    sampled_blocks = sampled_blocks_res
)

with open(f"data/analysis_data/tex_tables.txt", 'w') as f:
    f.write(tbl_out)
