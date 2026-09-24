# Financial LLM Optimization

This repository contains experiments for my COSC695 Independent Study, **Optimizing Small Language Models for Financial Signal Extraction: Accuracy, Efficiency, and Quantitative Performance**.

The initial notebooks focus on understanding financial NLP inference, generative language-model inference, financial dataset preparation, and parameter-efficient fine-tuning.

## Notebooks

### 1. `01_Finbert_inference.ipynb`

This notebook introduces financial sentiment classification using **FinBERT**.

The notebook demonstrates the complete inference pipeline:

* Loading the FinBERT tokenizer and model from Hugging Face
* Tokenizing financial text
* Inspecting input token IDs
* Passing the encoded text through FinBERT
* Examining raw model logits
* Applying softmax to obtain class probabilities
* Predicting one of three financial sentiment classes:

  * Positive
  * Negative
  * Neutral

The purpose of this notebook is to understand how a pretrained financial language model performs inference before moving into model fine-tuning.

---

### 2. `02_qwen_inference.ipynb`

This notebook introduces generative LLM inference using **Qwen2.5-1.5B-Instruct**.

Unlike FinBERT, which directly produces classification probabilities, Qwen is a causal language model that generates text token by token.

The notebook demonstrates:

* Loading the Qwen tokenizer
* Loading the Qwen2.5-1.5B-Instruct model
* Creating a financial sentiment classification prompt
* Using Qwen's chat template
* Tokenizing the prompt
* Generating model output
* Decoding the generated tokens into a sentiment response

This experiment establishes the untuned Qwen model as a baseline before financial fine-tuning.

---

### 3. `03_qwen_fingpt_dataset.ipynb`

This notebook begins the fine-tuning portion of the project using the **FinGPT financial sentiment dataset**.

The dataset used is:

`FinGPT/fingpt-sentiment-train`

The notebook includes:

* Loading the FinGPT dataset from Hugging Face
* Inspecting its `instruction`, `input`, and `output` fields
* Selecting a 500-example subset for an initial training experiment
* Converting FinGPT examples into Qwen chat format
* Tokenizing training examples
* Creating labels for supervised fine-tuning
* Configuring LoRA using Hugging Face PEFT
* Attaching LoRA adapters to Qwen2.5-1.5B-Instruct
* Training the LoRA adapters on financial sentiment data
* Saving the trained LoRA adapter

The LoRA configuration used in the initial experiment was:

* Rank (`r`): 16
* LoRA alpha: 32
* Target modules: `q_proj`, `v_proj`
* Dropout: 0.05
* Task type: Causal Language Modeling

The resulting parameter counts were:

* **Total parameters:** 1,545,893,376
* **Trainable LoRA parameters:** 2,179,072
* **Trainable percentage:** approximately 0.141%

This demonstrates parameter-efficient fine-tuning: almost all of the original Qwen parameters remain frozen while only a small set of LoRA adapter parameters are updated.

## Current Progress

Completed:

* FinBERT financial sentiment inference
* Qwen2.5-1.5B inference
* FinGPT dataset preparation
* Qwen chat-format preprocessing
* LoRA configuration
* LoRA fine-tuning on an initial 500-example dataset
* LoRA adapter saving

Next steps:

* Evaluate the LoRA model on unseen financial text
* Compare base Qwen with LoRA-tuned Qwen
* Perform conventional supervised fine-tuning experiments
* Implement QLoRA
* Compare LoRA and QLoRA based on accuracy, memory usage, training time, and computational efficiency
* Expand the experiments toward SEC filing analysis and financial signal extraction
