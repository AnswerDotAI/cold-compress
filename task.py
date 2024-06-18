from datasets import load_dataset
from metric import AutoMetric


class EvaluationTask:
    def __init__(self, prompt_template, max_tokens, **kwargs):
        self.prompt_template = prompt_template
        self.max_tokens = max_tokens
        self._load_data(**kwargs)

    def _load_data(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def get_eval_prompts(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def compute_metrics(self, predictions):
        raise NotImplementedError("This method should be overridden by subclasses.")


class Squality(EvaluationTask):
    DEFAULT_PROMPT_TEMPLATE = """You are given a story and a question. Answer the question in a paragraph.

    Story:
    {story}

    Question:
    {question}"""

    def __init__(
        self, prompt_template=DEFAULT_PROMPT_TEMPLATE, max_tokens=1024, **kwargs
    ):
        super().__init__(prompt_template, max_tokens, **kwargs)

    def _load_data(self):
        self.dataset = load_dataset("pszemraj/SQuALITY-v1.3")["test"]
        eval_prompts = []
        references = []
        for datum in self.dataset:
            story = datum["document"]
            for question in datum["questions"]:
                eval_prompts.append(
                    self.prompt_template.format(
                        story=story, question=question["question_text"]
                    )
                )
                references.append(
                    [resp["response_text"] for resp in question["responses"]]
                )
        self.eval_prompts = eval_prompts
        self.references = references
        self.metrics = {
            "BertScore": AutoMetric.from_name("bertscore"),
            "Rouge": AutoMetric.from_name("rouge"),
        }

    def get_eval_prompts(self):
        return self.eval_prompts

    def compute_metrics(self, predictions):
        return {
            metric_name: metric.compute(predictions, self.references)
            for metric_name, metric in self.metrics.items()
        }


TASK_MAPPING = {"squality": Squality}


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
