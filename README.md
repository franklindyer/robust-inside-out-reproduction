This repo contains code and some human-readable metadata for an attempted reproduction of the paper *Inside-Out: Hidden Factual Knowledge in LLMs* (Gekhman et al., 2025). This code was written as a course project for the class *Robust evaluation and interpretability of language models*, taught by Andreas Waldis at the Universität Tübingen in SoSe 2026.

This is still a work in progress! Draft of final report coming soon!

Final experimental data is not uploaded to this repo; see instead [this HuggingFace repo](https://huggingface.co/datasets/frpzzd/inside-out-reproduction) for the experimental data, including intermediate outputs for all parts of the data pipeline.

The following Python files in `scripts` are involved in the main experimental pipeline; all others are only for post-analysis (figure generation, summary statistics, etc). As a convention, long-running scripts (run on HPC machines for the main experiments) are prefixed with `job_`.

- `shared.py` contains several helpful for various parts of the experimental pipeline  
- `make_splits.py` generates training, testing and dev splits, depending on `data/wikidata` and writing output to `data/splits`  
- `job_answer_sample.py` samples 1000 answers for a given model for each question, depending on `data/splits` and writing output to `data/sampled_answers`  
- `job_list_options.py` generates a list of alternative answer choices for a given model and for each question, depending on `data/sampled_answers` and writing to `data/answer_options`  
    - also adds correctness judgments to answers using RewardBert  
- `job_answer_tf.py` calculates prompt-based T/F scores for a given model for all questions, depending on `data/answer_options` and writing to `data/tf_answers`  
- `make_selected_answers.py` aggregates train and test gold answers for each question category, as well as commonly sampled answers, depending on `data/sampled_answers` and writing the aggregated files to the same folder  
- `make_probe_training_questions.py` determines a list of questions to use for (knowledge-aware) probe training, depending on `data/answer_options` and writing to `data/probe_training`  
- `job_probe_tensors.py` pre-generates Pytorch tensors for the questions chosen to train the internal probe, depending on `data/probe_training` and writing to the same folder  
- `job_train_and_score_probe.py` trains the internal probe on training tensors and uses the trained probe to score testing questions, depending on the tensors in `data/probe_training` and writing scores to `data/probe_scores`  
    - includes 2 variants: one in which the probe is trained using model-sampled distrators, and one in which the distractors are other gold Wikidata entities from the same category
- `data/calc_k_scores.py` calculate the final `K` and `K*` scores for external and internal scoring functions, depending on the scores from `data/tf_answers` and `data/probe_scores` and writing to `k_scores`  

Some additional useful information:

- `data/split.json` contains a JSON representation of the distribution of Wikidata relations used for the dataset, as well as training and testing splits  
- `data/models.json` contains metadata about the models used in the experiment  
- `data/templates` contains the prompt templates used for the experiment  
