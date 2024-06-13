from abc import ABC, abstractmethod
from datasets import load_dataset, Dataset, DatasetDict


class Benchmark(ABC):
    def __init__(self, name: str):
        self.name = name
        self.dataset = self.download()

    @abstractmethod
    def download(self) -> Dataset | DatasetDict:
        pass

    def get_train(self) -> Dataset:
        return self.dataset["train"]

    def get_test(self) -> Dataset:
        return self.dataset["test"]

    def get_validation(self) -> Dataset:
        return self.dataset["validation"]

    @abstractmethod
    def question(self, ex: dict) -> str:
        pass

    @abstractmethod
    def instruction(self, ex: dict) -> str:
        pass

    @abstractmethod
    def context(self, ex: dict) -> str:
        pass

    @abstractmethod
    def answer(self, ex: dict) -> str:
        pass


class TriviaQA(Benchmark):
    def __init__(self, use_web: bool = False):
        super().__init__("triviaqa")
        self.use_web = use_web

    def download(self) -> Dataset:
        return load_dataset("trivia_qa", "rc")

    def question(self, ex: dict) -> str:
        return ex["question"]

    def instruction(self) -> str:
        return "Answer the following question."

    def context(self, ex: dict) -> str:
        """
        Create a single string from the entity pages and (optionally) search results.
        TODO: Format uniformly with other datasets and decide on whether to include search results.
        """
        wikis = ex["entity_pages"]
        webs = ex["search_results"]

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

        return context_str.strip()

    def answer(self, ex: dict) -> str | list[str]:
        # TriviaQA allows for any answer within a predefined set of aliases
        assert ex["answer"]["value"] in ex["answer"]["aliases"]
        return ex["answer"]["aliases"]


class Dolomites(Benchmark):
    def __init__(self):
        super().__init__("dolomites")
        self.use_web = False

    def download(self) -> Dataset:
        return load_dataset("sarahpann/processed_dolomites")

    def question(self, ex: dict) -> str:
        return ex["input"]

    def instruction(self) -> str:
        return ""

    def context(self, ex: dict) -> str:
        return ""

    def answer(self, ex: dict) -> str:
        return ex["output"]

    def get_test(self) -> Dataset:
        return


BENCHMARKS = {"triviaqa": TriviaQA, "dolomites": Dolomites}


if __name__ == "__main__":
    dataset = TriviaQA(use_web=False)

    print(dataset.answer(dataset.get_train()[0]))
