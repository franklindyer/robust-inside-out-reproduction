This data is provided as the output of an experiment attempting a partial reproduction of the paper *Inside-Out: Hidden Factual Knowledge in LLMs* (Gekhman et al., 2025). The experiment was run as a course project for the class *Robust evaluation and interpretability of language models*, taught by Andreas Waldis at the Universität Tübingen in SoSe 2026.

The following are brief descriptions of the contents of each top-level file and directory:

- `models.json` contains a collection of model metadata including HuggingFace tags.  
- `split.json` contains a list of Wikidata relations used for the experiment as well as some metadata about the files in the `wikidata` directory and the number of questions from each relation used for training and testing.  
- `templates` contains prompt templates used for the experiments.  
- `wikidata` contains raw TSV outputs to Wikidata queries executed through the SPARQL engine.  
- `splits` contains the `train`, `test` and `dev` splits for the experiments, each of which consists of an entity-centric question-answer pair and a unique ID.  
- `sampled_answers` contains 1000 model-generated answers to each question at a temperature of `T=1`; because of the magnitude of the data, we group them in multiple files with 500 questions per file.  
- `answer_options` contains a final list of answer options considered for each question, including the gold answer as well as any answers sampled > 1 times by each model.  
- `tf_answers` contains correct/incorrect judgment probabilities for each model and each question in the `test` split. Though unnecessary for the original experiment, we include probabilities for both positive and negative answers, as well as probabilities for the prompt template variant in which the answer choices are reversed. This data might allow for an interesting additional analysis of model stability under irrelevant changes to the prompt.  
- `probe_training` contains the final data selected for use in probe training, consisting of a single correct and a single incorrect (sampled) answer for each training question for which the model sampled at least one correct answer. It also contains a `torch` tensorfile containing the hidden representations of the validation prompts for the correct and incorrect answers to each question.  
- `probe_scores` contains the results of the logistic probe training process on the tensors stored in the respective model directory of `probe_training`.  
- `k_scores` contains the per-question internal and external k-scores for each model.  
