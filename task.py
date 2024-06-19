from abc import ABC, abstractmethod

from datasets import load_dataset

from metric import AutoMetric


class EvaluationTask(ABC):
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    mandatory_cols = ["context", "question", "prompt", "labels"]

    def __init__(self, prompt_template, max_tokens, hf_args=None, **kwargs):
        self.prompt_template = prompt_template
        self.max_tokens = max_tokens
        self.hf_args = hf_args

        # Download the dataset
        self._download(**kwargs)

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
            self.dataset[split] = self.dataset[split].map(
                self.prepare_batch, batched=True, remove_columns=remove_cols
            )
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
        return self._compute_metrics(predictions, dataset)

    def _compute_metrics(self, predictions: list, labels: list[str | list[str]]):
        return {
            metric_name: metric.compute(predictions, labels)
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


class Squality(EvaluationTask):
    DEFAULT_PROMPT_TEMPLATE = """You are given a story and a question. Answer the question in a paragraph.

Story:
{story}

Question:
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

Context:
{context}

Question:
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


class ScrollsQuality(EvaluationTask):
    DEFAULT_PROMPT_TEMPLATE = """You are given a question and a relevant context. Answer the question without any explanation.

Context:
{context}

Question:
{question}"""

    def __init__(
        self, prompt_template=DEFAULT_PROMPT_TEMPLATE, max_tokens=1024, **kwargs
    ):
        super().__init__(
            prompt_template, max_tokens, hf_args=["rbiswasfc/quality"], **kwargs
        )

        self.metrics = {
            "BertScore": AutoMetric.from_name("bertscore"),
            "Rouge": AutoMetric.from_name("rouge"),
        }

    def prepare_row(self, row: dict):
        context = row["context"]
        question = row["question"]
        choices = row["choices"]
        answer = choices[row["label"]]

        return {
            "context": context,
            "question": question,
            "prompt": self.prompt_template.format(context=context, question=question),
            "labels": answer,
        }

    def get_test(self):
        # test split from quality dataset doesn't have ground truth -> using validation split as alternative
        return self.get_split(self.validation_split)


TASK_MAPPING = {
    "squality": Squality,
    "triviaqa": TriviaQA,
    "scrolls_quality": ScrollsQuality,
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

    args = parser.parse_args()

    task = AutoTask.from_name(args.task)

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
