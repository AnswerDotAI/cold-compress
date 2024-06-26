import argparse
import os

import claudette
from claudette import models as anthropic_models, Chat

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from task import AutoTask
from evaluate import load
import pandas as pd
from scipy.stats import spearmanr
import numpy as np

import json

import datasets
from datasets import Dataset

np.random.seed(1992)
from tqdm import tqdm
import regex as re

try:
    import nltk

    nltk.sent_tokenize("test")
except:
    nltk.download("punkt")


assert (
    "ANTHROPIC_API_KEY" in os.environ
), "Please set the ANTHROPIC_API_KEY environment variable."


ASK_LLM_PROMPT_RANK = """# Instruction
    You are shown a question, answer(s), and context sentences with varying relevance and helpfulness.
    - Rank each of the context sentences in order of importance for answering the question.
    - Include all of the context sentences, ranked from most to least helpful.
    - Format your response as such: 
        "1. [sentence number] sentence
         2. [sentence number] sentence..."

    # Question
    {question}

    # Answer
    {answer}

    # Context
    {context}"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="The maximum number of examples to process for each split.",
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["squality"],
        help="The dataset to use.",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        # Default dir is the parent directory of this file in a "data" folder
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
        help="The output directory.",
    )

    args = parser.parse_args()

    out_dict = {"questions": [], "answers": [], "contexts": [], "rankings": []}

    task = AutoTask.from_name(args.tasks[0])

    out_hf_path = os.path.join(args.out_dir, f"{task.name}_claude_extract1ed")
    os.makedirs(out_hf_path, exist_ok=True)

    dataset = {
        "train": task.get_train(),
        "validation": task.get_validation(),
        "test": task.get_test(),
    }

    model = [m for m in anthropic_models][2] # [opus, sonnet, haiku]

    for i, ex in tqdm(enumerate(dataset["train"])):
        if i == args.max_examples: break

        chat = Chat(model, sp="""You are a helpful assistant.""")

        sents = nltk.sent_tokenize(ex["context"] or ex["question"])
        labels = ex["labels"]

        ctx_sent_delim = "\n".join(
            f"[{i + 1}] " + re.sub("\s+", " ", sent).strip() for i, sent in enumerate(sents)
        )

        prompt = ASK_LLM_PROMPT_RANK.format(
            question=ex["question"],
            context=ctx_sent_delim,
            answer="\n".join([re.sub("\s+", " ", l).strip() for i, l in enumerate(labels)]),
        )

        response = chat(prompt)

        out_dict["questions"].append(ex["question"])
        out_dict["answers"].append(labels)
        out_dict["contexts"].append(ctx_sent_delim)
        out_dict["rankings"].append(response.content[0].text)


    with open(f"{args.out_dir}/squality_ctx_rankings.json", "w") as f:
        json.dump(out_dict, f)
    
