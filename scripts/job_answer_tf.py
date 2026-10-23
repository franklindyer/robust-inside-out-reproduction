import sys

sys.path.append("./scripts")

from shared import *

model_id = sys.argv[1]

if __name__ == "__main__":
    generate_tf_answer_batch(model_id)
