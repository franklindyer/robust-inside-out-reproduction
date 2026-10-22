import sys

sys.path.append("./scripts")

from shared import *

model_id = sys.argv[1]

if __name__ == "__main__":
    answer_options_gen_for_model(model_id)
