---
base_model: Qwen/Qwen3-1.7B
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen3-1.7B
- lora
- sft
- transformers
- trl
---

# Model Card for Model ID

Provide answers when a question is asked about VIT

## Model Details

### Model Description

An FAQ dataset-fine tuned on the model that can answer questions relating to VIT Vellore University(mostly admission related)


- **Developed by:** Robin Bijo
- **Funded by :** Prof. Nivedita S
- **Shared by :** Jey Sudharsan Balaji and Rahul James Emmanuel
- **Model type:** Qwen3
- **Language(s) (NLP):** en
- **License:** MIT
- **Finetuned from model:** Qwen/Qwen3-1.7B

### Model Sources 

- **Repository:** https://huggingface.co/Qwen/Qwen3-1.7B


## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

[More Information Needed]

### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

[More Information Needed]

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

[More Information Needed]

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

[More Information Needed]

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.

## How to Get Started with the Model

Use the code below to get started with the model.

[More Information Needed]

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

[More Information Needed]

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->


### Framework versions

- PEFT 0.21.2