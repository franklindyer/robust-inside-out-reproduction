import json
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn
import sys

model_ids = ["mistral-7b", "llama3-8b", "gemma2-9b"]

gold_prob_gaps = {}
gold_prob_deficits = {}

for model_id in model_ids:
    gold_prob_gaps[model_id] = []
    gold_prob_deficits[model_id] = []

    base_dir = f"data/tf_answers/{model_id}"
    for fname in os.listdir(base_dir):
        if fname[0] == ".":
            continue
        probs = json.loads(open(f"{base_dir}/{fname}").read())

        gold_prob_gaps[model_id] += [
            (r["original"]["a_prob"] or 0) - (r["reversed"]["b_prob"] or 0)
            for rs in probs.values()
            for r in rs[:1]
        ]

        gold_prob_deficits[model_id] += [
            (rs[0]["original"]["a_prob"] or 0) + (rs[0]["original"]["b_prob"] or 0)
            for rs in probs.values()
            for r in rs[:1]
        ]

for model_id in model_ids:
    seaborn.distplot(
        gold_prob_gaps[model_id], 
        label=model_id,
        bins=np.linspace(-1, 1, 50),
    )
plt.xlim([-0.5, 1.0])
plt.legend()
plt.title("Gold P(true) difference when listed first vs. second")
plt.savefig(f"data/img/gold_prob_gaps.svg", format='svg')

plt.clf()
for model_id in model_ids:
    seaborn.distplot(
        gold_prob_deficits[model_id], 
        label=model_id,
        bins=np.linspace(0, 1, 50),
        kde=False
    )
plt.xlim([0, 1.0])
plt.legend(loc="upper left")
plt.title("Gold P(true) + P(false)")
plt.savefig(f"data/img/gold_prob_deficits.svg", format='svg')
