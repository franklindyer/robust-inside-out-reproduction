import json
import sys

sys.path.append("./scripts")

from shared import *

model_id = sys.argv[1]

model_data = json.loads(open("data/models.json").read())

model_dim = model_data[model_id]["model_dim"]
layer_num = model_data[model_id]["target_layer"]

if __name__ == "__main__":
    # Train probe with sampled incorrect answers and calc scores
    train_and_score_probe(
        model_id,
        hidden_dim=model_dim,
        target_layer=layer_num,
        incorrect_col="incorrect", 
        out_fname="all_incorrect_sampled"
    )

    # Train probe with incorrect gold answers and calc scores
    train_and_score_probe(
        model_id, 
        hidden_dim=model_dim,
        target_layer=layer_num,
        incorrect_col="incorrect_gold", 
        out_fname="all_incorrect_gold"
    )
