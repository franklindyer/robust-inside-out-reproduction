import csv
import glob
import os

def load_all_questions():
    q_dict = {}
    for subdir in ["train", "test", "dev"]:
        for fname in glob.glob(f"data/splits/{subdir}/*.tsv"):
            with open(fname, 'r') as f:
                reader = csv.reader(f, delimiter="\t")
                for r in reader:
                    q_dict[r[0]] = {
                        "id": int(r[0]),
                        "question": r[1],
                        "answer": r[2]
                    }
    return q_dict
