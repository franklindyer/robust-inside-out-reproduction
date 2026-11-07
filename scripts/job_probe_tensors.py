import sys

sys.path.append("./scripts")

from shared import *

model_id = sys.argv[1]

model_data = json.loads(open("data/models.json").read())

layer_num = model_data[model_id]["target_layer"]

if __name__ == "__main__":
    make_probe_training_vectors(model_id, layer_inds=[0, layer_num])
