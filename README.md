# LSH KV Cache Eviction (ICLR 2025)

# Introduction
Official code repository for the paper:
LSH Tells You What to Discrad: An Adaptive Locality-Sensitive Strategy For KV Cache Compression

Our implementation is based on cold-compress. We are working on merging our implementation in to the main branch of cold-compress. 


# Installation
```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/nightly/
```

After logging in with `huggingface-cli login`, run any of the following:

```bash
bash scripts/prepare_llama3.sh
```

This will download model and tokenizer files from HuggingFace for [`Meta-Llama-3-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) and save them into a usable format inside `./checkpoints`.


# Quick Start

## Generate Response
```
python generate.py --cache_strategy lsh --lsh_dim 8 --prompt "What does the llama say?" --checkpoint_path ./checkpoints/meta-llama/Meta-Llama-3-8B-Instruct/model.pth
```

This will generate a response from a compiled Llama-3 with 8-bit LSH eviction (`--cache_strategy lsh --lsh_dim 8`).


## Run a single experiment using a cache config

```
python eval.py --cache_config lsh --lsh_dim 8 --tasks gsm8k
```

For a list of tasks, please refer to `tasks.py`. For a list of cache configs please refer to `COLD_COMPRESS_README.md` and `cache.py`.  

`eval.py` creates a directory under `results` based on the supplied cache arguments to dump the raw predictions and metrics for memory usage and task-specific performance. For


## Experiments in Parallel

Here is how to run the L2, LSH and Full cache config comparision experiements in parallel using multiple GPUs on a single machine.

### Free Response Question Answering

Before you run question answering experiments, you need to set an openai api key

```
export OPENAI_API_KEY=[api key]
```
because the GPT4-Judge requires openai api access. To turn off the GPT4-Judge metric, please edit the corresponding class in `tasks.py` and remove GPT4-Judge from the its list of metrics. 

To run the `gsm8k` free response question answering experiments: 

```
python parallelize_evals.py --command_file experiments/gsm8k.txt --num_gpus 8
```

To run the `medqa` free response question answering experiments: 
```
python parallelize_evals.py --command_file experiments/medqa.txt --num_gpus 8
```

Replace the number of GPUs with the correct number of GPUs on your machine.

### Multiple Choice

To run the `gsm8k_mc` multiple choice experiments: 

```
python parallelize_evals.py --command_file experiments/gsm8k_,c.txt --num_gpus 8
```

To run the `medqa_mc` multiple choice experiments: 
```
python parallelize_evals.py --command_file experiments/medqa_mc.txt --num_gpus 8
```

### Long Context

To run the `needle in a haystack` long context experiments:
```
python parallelize_evals.py --command_file experiments/rulerniah.txt --num_gpus 8
```

To run the `common words` long context experiments:
```
python parallelize_evals.py --command_file experiments/cwe.txt --num_gpus 8
```

# Visualizations

To generate visualizations used in the paper, please refer to instructions in `VISUALIZATIONS.ipynb`.