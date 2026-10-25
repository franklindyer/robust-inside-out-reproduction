import sys

sys.path.append("./scripts")

from shared import *

model_id = sys.argv[1]

if __name__ == "__main__":
    train_and_score_probe(model_id)
