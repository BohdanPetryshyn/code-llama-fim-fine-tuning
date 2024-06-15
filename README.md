# Code Llama fill-in-the-middle fine-tuning

This repository allows you to fine-tune the Code Llama model to fill in the middle on your own dataset by mirroring the process described in the original [Code Llama paper](https://arxiv.org/abs/2308.12950). Infilling (filling in the middle) models are optimal for code completion tasks, where the model is given a prefix and a suffix and is asked to fill the middle.

## How to use

### Prepare dataset and upload it to Hugging Face Hub.

The dataset must contain a column "content" with the code files you want to train the model on.

Example dataset with OpenAPI definitions is available at [here](https://huggingface.co/datasets/BohdanPetryshyn/openapi-completion-refined).

### Train the model using Google Colab

You can train the Code Llama 7B model using Google Colab with an A100 GPU. Use this notebook to train the model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fCnfuKJHzH9afwN0vcFF75daLDtHlN26?usp=sharing)

The notebook will save the trained adapter to your Hugging Face account. The adapter can be used with the Python Transformers library for inference (see [docs](https://huggingface.co/docs/transformers/main/en/peft)). To create a standalone model, you can merge the adapter with the original model. The merged model can be used with the Hugging Face Inference Endpoints to serve the model as an API.

### Obtain a standalone model

This Google Colab notebook can be used to merge the fine-tuned adapter with the Code Llama 7B model using the free Tesla T4 GPU but requires high-RAM: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1aTCVkdopRp1o7Se68FlWw6RrlKvn0jTh?usp=sharing)

### Serve the model as an API

The merged model can be used with the [Hugging Face Inference Endpoints](https://huggingface.co/inference-endpoints/dedicated) to serve the model as an API. Code Llama 7B model requires a single Nvidia A10G runtime which costs $1.00 per hour at the time of writing.

## Related Publication

This repository contains the code and data for the paper "Optimizing Large Language Models for OpenAPI Code Completion" by [Bohdan Petryshyn](https://orcid.org/0009-0003-4030-4842) and [Mantas Lukoševičius](https://orcid.org/0000-0001-7963-285X).

### Abstract

Recent advancements in Large Language Models (LLMs) and their utilization in code generation tasks have significantly reshaped the field of software development. Despite the remarkable efficacy of code completion solutions in mainstream programming languages, their performance lags when applied to less ubiquitous formats such as OpenAPI definitions. This study evaluates the OpenAPI completion performance of GitHub Copilot, a prevalent commercial code completion tool, and proposes a set of task-specific optimizations leveraging Meta's open-source model Code Llama. A semantics-aware OpenAPI completion benchmark proposed in this research is used to perform a series of experiments through which the impact of various prompt-engineering and fine-tuning techniques on the Code Llama model's performance is analyzed. The fine-tuned Code Llama model reaches a peak correctness improvement of 55.2% over GitHub Copilot despite utilizing 25 times fewer parameters than the commercial solution's underlying Codex model. Additionally, this research proposes an enhancement to a widely used code infilling training technique, addressing the issue of underperformance when the model is prompted with context sizes smaller than those used during training.

### Citation

If you found the dataset or the fine-tuning code helpful, please reference the original paper:

**BibTeX:**

```
@misc{petryshyn2024optimizing,
      title={Optimizing Large Language Models for OpenAPI Code Completion}, 
      author={Bohdan Petryshyn and Mantas Lukoševičius},
      year={2024},
      eprint={2405.15729},
      archivePrefix={arXiv},
      primaryClass={id='cs.SE' full_name='Software Engineering' is_active=True alt_name=None in_archive='cs' is_general=False description='Covers design tools, software metrics, testing and debugging, programming environments, etc. Roughly includes material in all of ACM Subject Classes D.2, except that D.2.4 (program verification) should probably have Logics in Computer Science as the primary subject area.'}
}
```

**APA:**

```
Petryshyn, B., & Lukoševičius, M. (2024). Optimizing Large Language Models for OpenAPI Code Completion. arXiv preprint arXiv:2405.15729.
```

## Acknowledgements

This repository is adapted from https://github.com/pacman100/LLM-Workshop, which supports fine-tuning a number of models, including Code Llama. However, a number of problems were encountered when using the original repository with Code Llama. This repository contains improvements like context-level infilling (vs. document-level infilling), usage of correct Code Llama special tokens, among others.
