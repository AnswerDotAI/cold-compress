import random
from abc import ABC, abstractmethod
from string import ascii_uppercase
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset

from metric import AutoMetric
from tokenizer import get_tokenizer
import json

class EvaluationTask(ABC):
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    mandatory_cols = ["context", "question", "prompt", "labels"]
    requires_logits = False
    requires_perplexity = False

    def __init__(
        self,
        prompt_template,
        max_tokens,
        model_max_length,
        tokenizer,
        hf_args=None,
        **kwargs,
    ):
        self.prompt_template = prompt_template
        self.max_tokens = max_tokens
        self.model_max_length = model_max_length
        self.tokenizer = tokenizer
        self.hf_args = hf_args
        self.num_samples = kwargs.pop("num_samples", -1)
        self.random_seed = kwargs.pop("random_seed", 0)

        # Download the dataset
        self._download()

        # Lazy process each split as needed
        self.is_ready = {
            self.train_split: False,
            self.validation_split: False,
            self.test_split: False,
        }

    def _download(self):
        # Can over-write if not using HF
        self.dataset = load_dataset(*self.hf_args)

    def get_split(self, split):
        remove_cols = [
            col
            for col in self.dataset[split].column_names
            if col not in self.mandatory_cols
        ]
        if not self.is_ready[split]:
            split_data = self.dataset[split]
            split_data = split_data.map(
                self.prepare_batch, batched=True, remove_columns=remove_cols
            )

            # Filter out examples that could be too long for the model
            filtered_data = split_data.filter(
                lambda x: len(self.tokenizer(x["prompt"])) + self.max_tokens
                <= self.model_max_length
            )
            print(
                f"Filtered {len(split_data) - len(filtered_data)} examples from split {split}"
            )

            if self.num_samples > 0 and len(filtered_data) > self.num_samples:
                n = min(self.num_samples, len(filtered_data))
                print(f"Randomly sample {n} examples")
                # Use a fixed seed for reproducibility
                random.seed(self.random_seed)
                inds = random.Random(n).sample(range(len(filtered_data)), n)
                filtered_data = filtered_data.select(inds)

            self.dataset[split] = filtered_data
            self.is_ready[split] = True

        return self.dataset[split]

    def get_train(self):
        return self.get_split(self.train_split)

    def get_validation(self):
        return self.get_split(self.validation_split)

    def get_test(self):
        return self.get_split(self.test_split)

    def compute_metrics(self, predictions, split, dataset):
        assert self.is_ready[split], f"Split {split} has not been processed yet."
        assert (
            len(dataset) == len(predictions)
        ), f"Number of predictions and labels must match ({len(predictions)} != {len(dataset)})."
        return self._compute_metrics(dataset["prompt"], predictions, dataset["labels"])

    def _compute_metrics(
        self, prompts: list, predictions: list, labels: list[str | list[str]]
    ):
        return {
            metric_name: metric.compute(prompts, predictions, labels)
            for metric_name, metric in self.metrics.items()
        }

    def train_metrics(self, predictions):
        return self.compute_metrics(predictions, self.train_split, self.get_train())

    def validation_metrics(self, predictions):
        return self.compute_metrics(
            predictions, self.validation_split, self.get_validation()
        )

    def test_metrics(self, predictions):
        return self.compute_metrics(predictions, self.test_split, self.get_test())

    def prepare_batch(self, batch):
        keys = list(batch.keys())
        n = len(batch[keys[0]])
        processed = {k: [] for k in self.mandatory_cols}
        for i in range(n):
            row = {k: v[i] for k, v in batch.items()}
            out = {k: None for k in self.mandatory_cols}
            out = self.prepare_row(row)
            # Most tasks will return a single dictionary example from a single row
            if type(out) != list:
                out = [out]
            for x in out:
                for k in self.mandatory_cols:
                    processed[k].append(x.get(k, None))
        return processed

    @abstractmethod
    def prepare_row(self, row) -> dict | list[dict]:
        """Process a single row from the dataset."""
        pass


class LogitEvaluationTask(EvaluationTask):
    def __init__(self, prompt_template, max_tokens, hf_args=None, **kwargs):
        super().__init__(prompt_template, max_tokens, hf_args=hf_args, **kwargs)
        self.requires_logits = True

    @abstractmethod
    def _process_logits(self, logits, split):
        """Process logits and return predictions."""
        pass

    def compute_metrics(self, predictions, split, dataset):
        # LogitEvaluationTask will get logits instead of token predictions, so we need to process them first
        predictions = self._process_logits(predictions, split)
        return super().compute_metrics(predictions, split, dataset)


class Squality(EvaluationTask):
    DEFAULT_PROMPT_TEMPLATE = """You are given a story and a question. Answer the question in a single paragraph.

====STORY====
{story}

====QUESTION====
{question}"""

    def __init__(
        self, prompt_template=DEFAULT_PROMPT_TEMPLATE, max_tokens=1024, **kwargs
    ):
        super().__init__(
            prompt_template, max_tokens, hf_args=["pszemraj/SQuALITY-v1.3"], **kwargs
        )

        self.metrics = {
            "BertScore": AutoMetric.from_name("bertscore"),
            "Rouge": AutoMetric.from_name("rouge"),
            "LLM-Rouge": AutoMetric.from_name("llm-rouge"),
        }

    def prepare_row(self, row: dict):
        story = row["document"].strip()
        questions = row["questions"]
        out = []
        for question in questions:
            question_text = question["question_text"].strip()
            prompt = self.prompt_template.format(
                story=story, question=question["question_text"]
            )
            labels = [resp["response_text"].strip() for resp in question["responses"]]
            out_row = {
                "prompt": prompt,
                "context": story,
                "question": question_text,
                "labels": labels,
            }
            out.append(out_row)
        return out


class TriviaQA(EvaluationTask):
    DEFAULT_PROMPT_TEMPLATE = """You are given a question and potentially relevant context from Wikipedia. Answer the question without any explanation.

====CONTEXT====
{context}

====QUESTION====
{question}"""

    def __init__(
        self, prompt_template=DEFAULT_PROMPT_TEMPLATE, max_tokens=1024, **kwargs
    ):
        self.use_web = kwargs.pop("use_web", False)

        super().__init__(
            prompt_template, max_tokens, hf_args=["trivia_qa", "rc"], **kwargs
        )

        self.metrics = {
            "BertScore": AutoMetric.from_name("bertscore"),
            "Rouge": AutoMetric.from_name("rouge"),
            "LLM-Rouge": AutoMetric.from_name("llm-rouge"),
        }

    def prepare_row(self, row: dict):
        wikis = row["entity_pages"]
        webs = row["search_results"]

        wiki_n = len(wikis["title"])
        web_n = len(webs["title"])

        contexts = []

        for i in range(wiki_n):
            contexts.append("# " + wikis["title"][i] + "\n" + wikis["wiki_context"][i])

        if self.use_web:
            for j in range(web_n):
                contexts.append(
                    "# "
                    + webs["title"][j]
                    + "\n"
                    + webs["description"][j]
                    + "\n"
                    + webs["search_context"][j]
                )

        context_str = "\n\n".join(contexts)
        question = row["question"]
        labels = row["answer"]["aliases"]
        if row["answer"]["value"] not in labels:
            labels.append(row["answer"]["value"])
        assert len(labels) > 0
        return {
            "context": context_str,
            "question": question,
            "prompt": self.prompt_template.format(
                context=context_str, question=question
            ),
            "labels": labels,
        }


class Dolomites(EvaluationTask):
    DEFAULT_PROMPT_TEMPLATE = """You need to perform a writing task from the field of {field}.
You are given (1) a task description which contains input and output sections, and (2) an example input for this task, which is a sample of the input sections of the task with concrete details.
You need to generate the output sections for the given example input.

IMPORTANT:
- Make sure the length of each output section matches the required length and the section headers are exactly the same.
- Make sure the output follows the structure of the output sections in the task description, is factually accurate and detailed.

====TASK DESCRIPTION====
{task_description}

====EXAMPLE INPUT====
{example_input}"""

    def __init__(
        self, prompt_template=DEFAULT_PROMPT_TEMPLATE, max_tokens=1024, **kwargs
    ):
        super().__init__(
            prompt_template, max_tokens, hf_args=["fladhak/dolomites"], **kwargs
        )

        # Dolomites test split does not have references, so we will use validation split for testing
        self.test_split = "validation"

        self.metrics = {
            "BertScore": AutoMetric.from_name("bertscore"),
            "Rouge": AutoMetric.from_name("rouge"),
            "LLM-Rouge": AutoMetric.from_name("llm-rouge"),
        }

    def prepare_row(self, row: dict):
        field = row["field"]
        task_objective = row["task_objective"]
        task_procedure = row["task_procedure"]
        task_input = row["task_input"]
        task_output = row["task_output"]
        task_notes = row["task_notes"]
        example_input = row["example_input"]
        ref = row["example_output"]

        task_description = f"Task objective: {task_objective}\nTask prodecedure: {task_procedure}\nTask input: {task_input}\nTask output: {task_output}"
        if task_notes is not None:
            task_description += f"\nAdditional notes: {task_notes}"

        prompt = self.prompt_template.format(
            field=field, task_description=task_description, example_input=example_input
        )

        return {
            "prompt": prompt,
            "field": field,
            "context": task_description,
            "question": example_input,
            "labels": ref,
        }


class QMSum(EvaluationTask):
    DEFAULT_PROMPT_TEMPLATE = """You will be shown a meeting transcipt along with a query. Your task is to carefully read the transcript and provide a concise answer to the query.

====MEETING TRANSCRIPT====
{transcript}

====QUERY====
{query}"""

    def __init__(
        self, prompt_template=DEFAULT_PROMPT_TEMPLATE, max_tokens=1024, **kwargs
    ):
        super().__init__(
            prompt_template, max_tokens, hf_args=["fladhak/qmsum"], **kwargs
        )

        self.metrics = {
            "BertScore": AutoMetric.from_name("bertscore"),
            "Rouge": AutoMetric.from_name("rouge"),
            "LLM-Rouge": AutoMetric.from_name("llm-rouge"),
        }

    def prepare_row(self, row: dict):
        transcript = "\n\n".join(
            [f"{x['speaker']}: {x['content']}" for x in row["transcript"]]
        )
        query = row["query"]
        answer = row["answer"]

        prompt = self.prompt_template.format(transcript=transcript, query=query)

        return {
            "prompt": prompt,
            "context": transcript,
            "labels": answer,
        }


class Musique(EvaluationTask):
    DEFAULT_PROMPT_TEMPLATE = """You will be shown several paragraphs from Wikipedia along with a question. Your task is to carefully read the paragraphs and provide a concise answer to the question.
IMPORTANT: You should only use the infomation provided in the paragraphs to answer the question.

====PARAGRAPHS====
{paragraphs}

====QUESTION====
{question}"""

    def __init__(
        self, prompt_template=DEFAULT_PROMPT_TEMPLATE, max_tokens=128, **kwargs
    ):
        super().__init__(
            prompt_template, max_tokens, hf_args=["fladhak/musique"], **kwargs
        )

        # Musique test split does not have references, so we will use validation split for testing
        self.test_split = "validation"

        self.metrics = {
            "BertScore": AutoMetric.from_name("bertscore"),
            "Rouge": AutoMetric.from_name("rouge"),
            "LLM-Rouge": AutoMetric.from_name("llm-rouge"),
        }

    def prepare_row(self, row: dict):
        paragraphs = "\n\n".join(
            [f"{x['title']}:\n{x['paragraph_text']}" for x in row["paragraphs"]]
        )
        question = row["question"]
        answers = [row["answer"]] + row["answer_aliases"]

        prompt = self.prompt_template.format(paragraphs=paragraphs, question=question)

        return {
            "prompt": prompt,
            "context": paragraphs,
            "question": question,
            "labels": answers,
        }


class TruthfulQA(LogitEvaluationTask):
    DEFAULT_PROMPT_TEMPLATE = """You will be shown a question along with several possible answers. Please carefully read the question and the answer choices and pick the best answer.
IMPORTANT: You should simply provide the letter corresponding to the answer choice that you picked. You do not need to write out the entire answer or provide any explanation.

====QUESTION====
{question}

====ANSWER CHOICES====
{choices}"""

    def __init__(self, prompt_template=DEFAULT_PROMPT_TEMPLATE, max_tokens=1, **kwargs):
        super().__init__(
            prompt_template,
            max_tokens,
            hf_args=["truthfulqa/truthful_qa", "multiple_choice"],
            **kwargs,
        )

        # Musique test split does not have references, so we will use validation split for testing
        self.test_split = "validation"

        self.metrics = {
            "Accuracy": AutoMetric.from_name("accuracy"),
        }
        self.mandatory_cols = self.mandatory_cols.copy() + ["num_choices"]

    def prepare_row(self, row: dict):
        question = row["question"]
        choices = "\n".join(
            [
                f"{char}. {opt}"
                for char, opt in zip(ascii_uppercase, row["mc1_targets"]["choices"])
            ]
        )
        answer = ascii_uppercase[row["mc1_targets"]["labels"].index(1)]

        prompt = self.prompt_template.format(question=question, choices=choices)

        return {
            "prompt": prompt,
            "question": question,
            "context": choices,
            "labels": answer,
            "num_choices": len(row["mc1_targets"]["choices"]),
        }

    def _process_logits(self, logits, split):
        preds = []
        for l, nc in zip(logits, self.get_split(split)["num_choices"]):
            pred = [l[ascii_uppercase[i]] for i in range(nc)]
            preds.append(ascii_uppercase[np.argmax(pred)])

        return preds


class ScrollsQuality(LogitEvaluationTask):
    """
    Evaluation dataset derived from `tau/scrolls`.
    It is processed into a suitable format here: https://huggingface.co/datasets/rbiswasfc/quality.
    Test split doesn't have ground truths, hence it will use validation split as an alternative.
    """

    DEFAULT_PROMPT_TEMPLATE = """You will be given a context, a question related to that context, and four possible answer choices. Carefully read the context, question, and answer choices, then select the best answer.
IMPORTANT: Provide only the letter corresponding to your chosen answer. Do not write out the full answer or give any explanation.

====CONTEXT====
{context}

====QUESTION====
{question}

====ANSWER CHOICES====
{choices}"""

    def __init__(self, prompt_template=DEFAULT_PROMPT_TEMPLATE, max_tokens=1, **kwargs):
        super().__init__(
            prompt_template, max_tokens, hf_args=["rbiswasfc/quality"], **kwargs
        )

        self.metrics = {
            "Accuracy": AutoMetric.from_name("accuracy"),
        }
        self.test_split = "validation"  #     Test split doesn't have ground truths - use validation split

        self.mandatory_cols = self.mandatory_cols.copy() + ["num_choices"]

    def prepare_row(self, row: dict):
        context = row["context"]
        question = row["question"]
        choices = row["choices"]
        num_choices = len(choices)
        answer = ascii_uppercase[row["label"]]

        choices = "\n".join(
            [f"{char}. {opt}" for char, opt in zip(ascii_uppercase, choices)]
        )

        return {
            "context": context,
            "question": question,
            "prompt": self.prompt_template.format(
                context=context, question=question, choices=choices
            ),
            "labels": answer,
            "num_choices": num_choices,
        }

    def _process_logits(self, logits, split):
        preds = []
        for l, nc in zip(logits, self.get_split(split)["num_choices"]):
            pred = [l[ascii_uppercase[i]] for i in range(nc)]
            preds.append(ascii_uppercase[np.argmax(pred)])

        return preds


class RulerQA(EvaluationTask):
    """
    RULER hotpotqa task with 8k context length. (context length can be adjusted as needed)
    """

    DEFAULT_PROMPT_TEMPLATE = "{task_input}"

    def __init__(
        self, prompt_template=DEFAULT_PROMPT_TEMPLATE, max_tokens=32, **kwargs
    ):
        super().__init__(
            prompt_template,
            max_tokens,
            hf_args=["rbiswasfc/ruler", "qa_2_8k"],
            **kwargs,
        )

        self.metrics = {
            "StringMatch": AutoMetric.from_name("ruler-string-match", match_part=True),
        }
        self.test_split = "validation"

    def prepare_row(self, row: dict):
        task_input = row["input"]

        question = task_input.split("Question:")[-1].split("Answer:")[0].strip()
        context = task_input.split("Question:")[0].strip()

        prompt = self.prompt_template.format(task_input=task_input)
        answer = row["outputs"]  # List[str]

        return {
            "context": context,
            "question": question,
            "prompt": prompt,
            "labels": answer,
        }


class PG19(EvaluationTask):
    """
    Generating the first ~8k tokens from PG-19 book corpus given the title.
    """

    DEFAULT_PROMPT_TEMPLATE = """You are given the title of a book and the first few words. Your job is to write it.

====TITLE====
{title}

====START OF BOOK====
{story_start}"""

    def __init__(self, prompt_template=DEFAULT_PROMPT_TEMPLATE, **kwargs):
        # Change max_tokens here if you want longer contexts
        max_tokens = kwargs.pop("seq_length")
        super().__init__(
            prompt_template,
            max_tokens=max_tokens,
            hf_args=["emozilla/pg19-test"],
            **kwargs,
        )
        self.train_split = None
        self.validation_split = None
        self.story_snippet_size = 256
        self.requires_perplexity = True

    def truncate(self, text: str):
        # Don't tokenize the whole book
        # Wp's ~1.5:1 wrt text tokens (Can re-write to be more exact with wp tokenizer if needed)
        text = " ".join(text.split(" ")[: int(self.max_tokens // 1.5)])
        return text

    def prepare_row(self, row: dict):
        story = self.truncate(row["text"])
        toks = story.split(" ")
        story_start, story_end = (
            " ".join(toks[: self.story_snippet_size]),
            " ".join(toks[self.story_snippet_size :]),
        )
        title = row["short_book_title"]
        prompt = self.prompt_template.format(title=title, story_start=story_start)
        return {
            "context": story_start,
            "question": f"How would you write a book with the title: {title}",  # Dummy question - not used in prompt but a required column
            "prompt": prompt,
            "labels": [story_end],
        }


class RulerNIAH(EvaluationTask):
    """
    RULER Multi-keys Needle-in-a-haystack (NIAH) task with 8k context length. (context length can be adjusted as needed)
    """

    DEFAULT_PROMPT_TEMPLATE = "{task_input}"

    def __init__(
        self, prompt_template=DEFAULT_PROMPT_TEMPLATE, max_tokens=128, **kwargs
    ):
        super().__init__(
            prompt_template,
            max_tokens,
            hf_args=["rbiswasfc/ruler", "niah_multikey_1_8k"],
            **kwargs,
        )

        self.metrics = {
            "StringMatch": AutoMetric.from_name("ruler-string-match", match_part=False),
        }
        self.test_split = "validation"

    def prepare_row(self, row: dict):
        task_input = row["input"]
        prompt = self.prompt_template.format(task_input=task_input)
        answer = row["outputs"]  # List[str]

        return {
            "context": "",
            "question": "",
            "prompt": prompt,
            "labels": answer,
        }
        
class RulerNIAH4K(EvaluationTask):
    """
    RULER Multi-keys Needle-in-a-haystack (NIAH) task with 4k context length. (context length can be adjusted as needed)
    """

    DEFAULT_PROMPT_TEMPLATE = "{task_input}"

    def __init__(
        self, prompt_template=DEFAULT_PROMPT_TEMPLATE, max_tokens=128, **kwargs
    ):
        super().__init__(
            prompt_template,
            max_tokens,
            hf_args=["rbiswasfc/ruler", "niah_multikey_1_4k"],
            **kwargs,
        )

        self.metrics = {
            "StringMatch": AutoMetric.from_name("ruler-string-match", match_part=False),
        }
        self.test_split = "validation"

    def prepare_row(self, row: dict):
        task_input = row["input"]

        prompt = self.prompt_template.format(task_input=task_input)
        answer = row["outputs"]  # List[str]

        return {
            "context": "",
            "question": "",
            "prompt": prompt,
            "labels": answer,
        }


class RulerVT(EvaluationTask):
    """
    RULER Multi-hop Tracing: Variable Tracking (VT) task with 8k context length. (context length can be adjusted as needed)
    """

    DEFAULT_PROMPT_TEMPLATE = "{task_input}"

    def __init__(
        self, prompt_template=DEFAULT_PROMPT_TEMPLATE, max_tokens=30, **kwargs
    ):
        super().__init__(
            prompt_template,
            max_tokens,
            hf_args=["rbiswasfc/ruler", "vt_8k"],
            **kwargs,
        )

        self.metrics = {
            "StringMatch": AutoMetric.from_name("ruler-string-match", match_part=False),
        }
        self.test_split = "validation"

    def prepare_row(self, row: dict):
        task_input = row["input"]

        question = task_input.split("Question:")[-1].split("Answer:")[0].strip()
        context = task_input.split("Question:")[0].strip()

        prompt = self.prompt_template.format(task_input=task_input)
        answer = row["outputs"]  # List[str]

        return {
            "context": context,
            "question": question,
            "prompt": prompt,
            "labels": answer,
        }


class RulerVT4K(EvaluationTask):
    """
    RULER Multi-hop Tracing: Variable Tracking (VT) task with 4k context length. (context length can be adjusted as needed)
    """

    DEFAULT_PROMPT_TEMPLATE = "{task_input}"

    def __init__(
        self, prompt_template=DEFAULT_PROMPT_TEMPLATE, max_tokens=30, **kwargs
    ):
        super().__init__(
            prompt_template,
            max_tokens,
            hf_args=["rbiswasfc/ruler", "vt_4k"],
            **kwargs,
        )

        self.metrics = {
            "StringMatch": AutoMetric.from_name("ruler-string-match", match_part=False),
        }
        self.test_split = "validation"

    def prepare_row(self, row: dict):
        task_input = row["input"]

        question = task_input.split("Question:")[-1].split("Answer:")[0].strip()
        context = task_input.split("Question:")[0].strip()

        prompt = self.prompt_template.format(task_input=task_input)
        answer = row["outputs"]  # List[str]

        return {
            "context": context,
            "question": question,
            "prompt": prompt,
            "labels": answer,
        }



class RulerCWE(EvaluationTask):
    """
    RULER Aggregation: Common Words (CWE) task with 8k context length. (context length can be adjusted as needed)
    """

    DEFAULT_PROMPT_TEMPLATE = "{task_input}"

    def __init__(
        self, prompt_template=DEFAULT_PROMPT_TEMPLATE, max_tokens=120, **kwargs
    ):
        super().__init__(
            prompt_template,
            max_tokens,
            hf_args=["rbiswasfc/ruler", "cwe_8k"],
            **kwargs,
        )

        self.metrics = {
            "StringMatch": AutoMetric.from_name("ruler-string-match", match_part=False),
        }
        self.test_split = "validation"

    def prepare_row(self, row: dict):
        task_input = row["input"]

        question = task_input.split("Question:")[-1].split("Answer:")[0].strip()
        context = task_input.split("Question:")[0].strip()

        prompt = self.prompt_template.format(task_input=task_input)
        answer = row["outputs"]  # List[str]

        return {
            "context": context,
            "question": question,
            "prompt": prompt,
            "labels": answer,
        }



class RulerCWE4K(EvaluationTask):
    """
    RULER Aggregation: Common Words (CWE) task with 4k context length. (context length can be adjusted as needed)
    """

    DEFAULT_PROMPT_TEMPLATE = "{task_input}"

    def __init__(
        self, prompt_template=DEFAULT_PROMPT_TEMPLATE, max_tokens=120, **kwargs
    ):
        super().__init__(
            prompt_template,
            max_tokens,
            hf_args=["rbiswasfc/ruler", "cwe_4k"],
            **kwargs,
        )

        self.metrics = {
            "StringMatch": AutoMetric.from_name("ruler-string-match", match_part=False),
        }
        self.test_split = "validation"

    def prepare_row(self, row: dict):
        task_input = row["input"]

        question = task_input.split("Question:")[-1].split("Answer:")[0].strip()
        context = task_input.split("Question:")[0].strip()

        prompt = self.prompt_template.format(task_input=task_input)
        answer = row["outputs"]  # List[str]

        return {
            "context": context,
            "question": question,
            "prompt": prompt,
            "labels": answer,
        }


class RepoBench(EvaluationTask):
    DEFAULT_PROMPT_TEMPLATE = """You will be given python files from a code repository, with the current file being shown last. Your task is to predict the next line of code in the current file.
NOTE: You should only predict the next line in the current file. Do not produce more than one line, and do not provide any explanation.

====REPOSITORY====
{repo}"""

    def __init__(
        self, prompt_template=DEFAULT_PROMPT_TEMPLATE, max_tokens=1024, **kwargs
    ):
        super().__init__(
            prompt_template, max_tokens, hf_args=["fladhak/reprobench"], **kwargs
        )

        self.metrics = {
            "ExactMatch": AutoMetric.from_name("exact_match"),
            "Levenshtein": AutoMetric.from_name("levenshtein"),
        }

    def prepare_row(self, row: dict):
        repo = row["prompt"]
        ref = row["ref"]

        prompt = self.prompt_template.format(repo=repo)

        return {
            "prompt": prompt,
            "context": None,
            "labels": ref,
        }
 
    
class GSM8K(EvaluationTask):
    """
    GSM8K (Grade School Math 8K) is a dataset of 8.5K high quality linguistically diverse grade school math word problems. 
    The dataset was created to support the task of question answering on basic mathematical problems that require multi-step reasoning.
    """
    DEFAULT_PROMPT_TEMPLATE = "{question}"

    def __init__(
        self, prompt_template=DEFAULT_PROMPT_TEMPLATE, max_tokens=210, **kwargs
    ):
        super().__init__(
            prompt_template,
            max_tokens,
            hf_args=["openai/gsm8k", "main"],
            **kwargs,
        )

        self.metrics = {
            "BertScore": AutoMetric.from_name("bertscore"),
            "Rouge": AutoMetric.from_name("rouge"),
            "ChatGPT-Rouge": AutoMetric.from_name("chatgpt-rouge"),
            "ChatGPTJudge": AutoMetric.from_name("chatgpt-as-a-judge"),
        }
        self.validation_split = None

    def prepare_row(self, row: dict):
        prompt = self.prompt_template.format(question=row['question'])
        answer = row["answer"]

        return {
            # "context": None,
            # "question": None,
            "prompt": prompt,
            "labels": answer,
        }


class GSM8K_MC(EvaluationTask):
    """
    GSM8K (Grade School Math 8K) is a dataset of 8.5K high quality linguistically diverse grade school math word problems. 
    The dataset was created to support the task of question answering on basic mathematical problems that require multi-step reasoning.
    This task is the multiple choice version of GSM8K.
    """
    DEFAULT_PROMPT_TEMPLATE = """{question}
    
    Choices:
    A. {choice_A}
    B. {choice_B}
    C. {choice_C}
    D. {choice_D}

    Choose an answer from the choices given. IMPORTANT: Provide only the letter corresponding to your chosen answer. Do not write out the full answer or give any explanation.
    """

    def __init__(
        self, prompt_template=DEFAULT_PROMPT_TEMPLATE, max_tokens=1, **kwargs
    ):
        super().__init__(
            prompt_template,
            max_tokens,
            hf_args=["guipenedo/gsm8k-mc"],
            **kwargs,
        )

        self.metrics = {
            "Accuracy": AutoMetric.from_name("accuracy"),
            "ExactMatch": AutoMetric.from_name("exact_match"),
        }
        self.validation_split = None

    def prepare_row(self, row: dict):
        choice_a = row["A"]
        choice_b = row["B"]
        choice_c = row["C"]
        choice_d = row["D"]
        question = row["Question"]
        answer = row["Answer"]
        prompt = self.prompt_template.format(question=question, choice_A=choice_a, choice_B=choice_b, choice_C=choice_c, choice_D=choice_d)

        return {
            "context": None,
            "question": question,
            "prompt": prompt,
            "labels": answer,
        }
    

class GSM8KDEBUG(EvaluationTask):
    """
    GSM8K (Grade School Math 8K) is a dataset of 8.5K high quality linguistically diverse grade school math word problems. 
    The dataset was created to support the task of question answering on basic mathematical problems that require multi-step reasoning.
    """
    DEFAULT_PROMPT_TEMPLATE = "{question}"

    def __init__(
        self, prompt_template=DEFAULT_PROMPT_TEMPLATE, max_tokens=210, **kwargs
    ):
        super().__init__(
            prompt_template,
            max_tokens,
            hf_args=["openai/gsm8k", "main"],
            **kwargs,
        )

        self.metrics = {
            "BertScore": AutoMetric.from_name("bertscore"),
            "Rouge": AutoMetric.from_name("rouge"),
            "ChatGPT-Rouge": AutoMetric.from_name("chatgpt-rouge"),
            "ChatGPTJudge": AutoMetric.from_name("chatgpt-as-a-judge"),
        }
        self.validation_split = None
    
    def _download(self):
        # Can over-write if not using HF
        self.dataset = load_dataset(*self.hf_args)
        self.dataset['train'] = self.dataset['train'].select(range(100))
        self.dataset['test'] = self.dataset['test'].select(range(20))

    def prepare_row(self, row: dict):
        prompt = self.prompt_template.format(question=row['question'])
        answer = row["answer"]

        return {
            # "context": None,
            # "question": None,
            "prompt": prompt,
            "labels": answer,
        }



class MEDQA(EvaluationTask):
    DEFAULT_PROMPT_TEMPLATE = """{system}

====QUESTION====
{question}
"""

    def __init__(self, prompt_template=DEFAULT_PROMPT_TEMPLATE, max_tokens=1, **kwargs):
        super().__init__(
            prompt_template,
            max_tokens,
            hf_args=None,
            **kwargs,
        )

        self.test_split = "test"
        self.test_split = "test"

        self.metrics = {
            "BertScore": AutoMetric.from_name("bertscore"),
            "Rouge": AutoMetric.from_name("rouge"),
            "ChatGPT-Rouge": AutoMetric.from_name("chatgpt-rouge"),
            "ChatGPTJudge": AutoMetric.from_name("chatgpt-as-a-judge"),
        }

    def _download(self):
        self.json_path = Path(__file__).parent / "data" / "medqa.json"
        data_files = {'test': [str(self.json_path)]}
        self.dataset = load_dataset("json", data_files=data_files, split="test")
        self.dataset = self.dataset.train_test_split(test_size=0.1, seed=42)

    def prepare_row(self, row: dict):
        system = row["system"]
        question = row["question"]
        answer = row["answer"]
        prompt = self.prompt_template.format(system=system, question=question)

        return {
            "prompt": prompt,
            "question": question,
            "labels": answer,
        }
        


class MEDQA_MC(EvaluationTask):
    DEFAULT_PROMPT_TEMPLATE = """You are a expert medical professional specializing in diagnosing and recommending treatments for various conditions. Carefully assess the patient's symptoms, medical history, and clinical details to determine the most appropriate treatment. Choose an answer from the choices given in the question. IMPORTANT: Provide only the letter corresponding to your chosen answer. Do not write out the full answer or give any explanation.
    
====QUESTION====
{question}
"""

    def __init__(self, prompt_template=DEFAULT_PROMPT_TEMPLATE, max_tokens=1, **kwargs):
        super().__init__(
            prompt_template,
            max_tokens,
            hf_args=None,
            **kwargs,
        )

        self.test_split = "test"

        self.metrics = {
            "Accuracy": AutoMetric.from_name("accuracy"),
        }
    
    def _download(self):
        self.json_path = Path(__file__).parent / "data" / "medqa.json"
        data_files = {'test': [str(self.json_path)]}
        self.dataset = load_dataset("json", data_files=data_files, split="test")
        self.dataset = self.dataset.train_test_split(test_size=0.1, seed=42)
       
    def prepare_row(self, row: dict):
        system = row["system"]
        question = row["question"]
        label = row["label"]
        prompt = self.prompt_template.format(system=system, question=question)

        return {
            "prompt": prompt,
            "question": question,
            "labels": label,
        }
    
      
class PasskeyRetrieval(EvaluationTask):
    """
    LLama3 1024 pass key retrieval eval task
    """
    DEFAULT_PROMPT_TEMPLATE = "{task_input}"

    def __init__(
        self, prompt_template=DEFAULT_PROMPT_TEMPLATE, max_tokens=100, **kwargs
    ):
        super().__init__(
            prompt_template,
            max_tokens,
            hf_args=["nanotron/llama3-1024-passkey-retrieval-eval"],
            **kwargs,
        )

        self.metrics = {
            "StringMatch": AutoMetric.from_name("ruler-string-match", match_part=False),
        }
        self.test_split = "train"

    def prepare_row(self, row: dict):
        task_input = row["prompt"]

        prompt = self.prompt_template.format(task_input=task_input)
        answer = row["answer"]  # List[str]

        return {
            "context": "",
            "question": "",
            "prompt": prompt,
            "labels": str(answer),
        }


class PassageRetrieval(EvaluationTask):
    """
    LongBench passage retrieval eval task
    """
    DEFAULT_PROMPT_TEMPLATE = """Given 30 English Wikipedia paragraphs numbered from 1 to 30, please determine which paragraph the given summary corresponds to. 
====QUESTION====

{context}
    
====Summary====

{task_input}

Please provide the paragraph number that best matches the summary in the format "Paragraph X".
"""

    def __init__(
        self, prompt_template=DEFAULT_PROMPT_TEMPLATE, max_tokens=5, **kwargs
    ):
        super().__init__(
            prompt_template,
            max_tokens,
            hf_args=["THUDM/LongBench", "passage_retrieval_en"],
            **kwargs,
        )

        self.metrics = {
            "Accuracy": AutoMetric.from_name("accuracy"),
            "ExactMatch": AutoMetric.from_name("exact_match"),
            "StringMatch": AutoMetric.from_name("ruler-string-match", match_part=False),
        }
        self.test_split = "test"
        
    def prepare_row(self, row: dict):
        task_input = row["input"]
        context = row["context"]

        prompt = self.prompt_template.format(task_input=task_input, context=context)
        answer = row["answers"][0]

        return {
            "context": context,
            "question": task_input,
            "prompt": prompt,
            "labels": answer,
        }
    

TASK_MAPPING = {
    "dolomites": Dolomites,
    "musique": Musique,
    "pg19": PG19,
    "qmsum": QMSum,
    "repobench": RepoBench,
    "rulerqa": RulerQA,
    "rulerniah": RulerNIAH4K,
    "rulervt": RulerVT4K,
    "rulercwe": RulerCWE4K,
    "scrollsquality": ScrollsQuality,
    "squality": Squality,
    "triviaqa": TriviaQA,
    "truthfulqa": TruthfulQA,
    "gsm": GSM8K,
    "gsm_debug": GSM8KDEBUG,
    "gsm_mc": GSM8K_MC,
    "medqa": MEDQA,
    "medqa_mc": MEDQA_MC,
    "passkey": PasskeyRetrieval,
    "passage": PassageRetrieval,
}


class AutoTask:
    def __init__(self):
        raise EnvironmentError(
            "This class is designed to be instantiated only through the from_name method"
        )

    def from_name(task_name, **kwargs):
        if task_name not in TASK_MAPPING:
            raise ValueError(
                f"Task {task_name} not found. Available tasks: {TASK_MAPPING.keys()}"
            )
        return TASK_MAPPING[task_name](**kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Test out implementation of EvaluationTask")

    parser.add_argument(
        "--task", type=str, default="triviaqa", choices=TASK_MAPPING.keys()
    )
    parser.add_argument("--compute_stats", action="store_true", default=False)
    parser.add_argument("--num_samples", default=int(1e10), type=int)

    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path(__file__).resolve().parent
        / "checkpoints/meta-llama/Meta-Llama-3-8B-Instruct/model.pth",
        help="Model checkpoint path.",
    )

    args = parser.parse_args()

    is_chat = (
        "chat" in str(args.checkpoint_path).lower()
        or "instruct" in str(args.checkpoint_path).lower()
    )

    tokenizer_path = args.checkpoint_path.parent / "tokenizer.model"
    if not tokenizer_path.is_file():
        # If there's no tokenizer.model, try to load the tokenizer from the parent directory
        # NOTE: We assume the tokenizer in the parent directory is compatible with huggingface transformers
        tokenizer_path = args.checkpoint_path.parent

    tokenizer = get_tokenizer(tokenizer_path, args.checkpoint_path, is_chat=is_chat)

    # Dummy values
    task_kwargs = {
        "model_max_length": int(1e10),
        "num_samples": args.num_samples,
        "tokenizer": tokenizer.encode_prompt if is_chat else tokenizer.encode,
    }

    def num_toks(x):
        return len(task_kwargs["tokenizer"](x))

    if args.compute_stats:
        stats = []
        for task_name in TASK_MAPPING.keys():
            print(f"Computing stats for {task_name}")
            task = AutoTask.from_name(task_name, **task_kwargs)
            test = task.get_test()

            prompts = test["prompt"]
            labels = test["labels"]

            prompt_tokens = sum([num_toks(p) for p in test["prompt"]]) / len(test)
            num_references = sum(
                [1 if type(l) != list else len(l) for l in labels]
            ) / len(test)

            avg_reference_len = []
            for l in labels:
                if type(l) != list:
                    l = [l]
                avg_reference_len.append(sum([num_toks(x) for x in l]) / len(l))
            avg_reference_len = sum(avg_reference_len) / len(avg_reference_len)

            avg_n_choices = (
                None
                if "num_choices" not in test
                else sum(test["num_choices"]) / len(test)
            )

            stats.append(
                {
                    "task": task_name,
                    "n": len(test),
                    "is_mcqa": task.requires_logits,
                    "prompt_tokens": prompt_tokens,
                    "label_tokens": avg_reference_len,
                    "n_choices": avg_n_choices,
                }
            )

        stats = pd.DataFrame(stats)
        stats_fn = Path(__file__).parent / "cache_configs" / "task_stats.csv"
        stats = stats.sort_values("task").reset_index(drop=True)
        stats.to_csv(stats_fn, index=False)
    else:
        task = AutoTask.from_name(args.task, **task_kwargs)
        test = task.get_test()
        print("Example test datapoint:\n\n")
        ex = test[0]
        for k, v in ex.items():
            print(f"{k}:\n{v}\n\n")

        train_predictions = ["This is a train prediction"] * len(task.dataset["train"])
        test_predictions = ["This is a test prediction"] * len(test)

        print("A 'not ready' error should be displayed below:\n\n")
        try:
            task.train_metrics(predictions=train_predictions)
        except Exception as e:
            print(e)

        print("A 'length mismatch' error should be displayed below:\n\n")
        try:
            task.test_metrics(predictions=test_predictions[:-1])
        except Exception as e:
            print(e)

        print("Dummy metrics for test split:\n\n")
        print(task.test_metrics(predictions=test_predictions))
