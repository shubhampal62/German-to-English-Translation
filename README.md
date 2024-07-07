# German-to-English-Translation
**Project Overview**
<br />
This project implements a German to English translation system using transformer models, leveraging the WMT 2016 dataset. The project explores three setups: training an encoder-decoder transformer model from scratch, performing zero-shot evaluation with the t5-small model, and fine-tuning the t5-small model for translation. Evaluation metrics include BLEU, METEOR, and BERTScore.
<br />
**Datasets**
<br />
Training Dataset: WMT 2016 German-English dataset (first 50,000 samples)
Validation Dataset: WMT 2016 German-English validation dataset
Test Dataset: WMT 2016 German-English test dataset
<br />
**Project Setups**
**Setup 2A: Encoder-Decoder Transformer Model (PyTorch)**
-Trained a sequence-to-sequence encoder-decoder transformer model from scratch using PyTorch.
-Followed the PyTorch tutorial for language translation with nn.Transformer and torchtext.
-Evaluated on validation and test datasets using BLEU, METEOR, and BERTScore.
**Setup 2B: Zero-shot Evaluation with t5-small Model**
-Performed zero-shot evaluation using the t5-small model from HuggingFace.
-Generated translations by prepending a specific prefix to the input sentence as per the T5 model documentation.
-Evaluated on validation and test datasets using BLEU, METEOR, and BERTScore.
**Setup 2C: Fine-tuning the t5-small Model**
-Fine-tuned the t5-small model for German-to-English translation using the training data.
-Followed the HuggingFace translation tutorial.
-Trained for at least two epochs with at least one layer set to trainable.
-Evaluated on validation and test datasets using BLEU, METEOR, and BERTScore.
**Evaluation Metrics**
-String-based Metrics: BLEU, METEOR
-Machine Learning Based Metric: BERTScore
**Results and Analysis**
-Generated plots for training and validation loss vs. epochs for setups 2A and 2C.
-Analyzed and explained the loss plots.
-Compared the performance differences between the three setups.
-Reported all evaluation metrics for all setups in a detailed report.
**Inference Pipelines**
-Developed an inference pipeline for real-time translation where the user can input a German sentence and receive translations from each setup.
-Created a CSV input/output pipeline to process files with German sentences and produce translations.
**Tech Stack**
-Programming Languages: Python
-Libraries: HuggingFace Datasets, Transformers, PyTorch, Numpy, Pandas, Matplotlib, SciPy, Scikit-Learn
