import sys

sys.path.append("scripts")

from shared import *

QS = load_all_questions()
opts_gen = answer_options_gen_for_batch("mistral-7b", 0)

id = 0
q = QS[0]
print(q["question"])
for ans_opt in opts_gen(q):
    print(ans_opt["text"].strip() + "\t" + ans_opt["source"])
