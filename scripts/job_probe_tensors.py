import sys

sys.path.append("./scripts")

from shared import *

model_id = sys.argv[1]

if __name__ == "__main__":
    make_probe_training_vectors(model_id)
