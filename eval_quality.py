import re

from datasets import load_dataset

quality_multiple_choice_pattern = re.compile(r" *\([A-D]\) *")


def _normalize_answer(text):
    return " ".join(text.split()).strip()


def _drop_duplicates_in_input(untokenized_dataset):
    # from scrolls/evaluator/dataset_evaluator.py

    indices_to_keep = []
    id_to_idx = {}
    outputs = []
    for i, (id_, output) in enumerate(
        zip(untokenized_dataset["id"], untokenized_dataset["output"])
    ):
        if id_ in id_to_idx:
            outputs[id_to_idx[id_]].append(output)
            continue
        indices_to_keep.append(i)
        id_to_idx[id_] = len(outputs)
        outputs.append([output])
    untokenized_dataset = untokenized_dataset.select(indices_to_keep).flatten_indices()
    untokenized_dataset = untokenized_dataset.remove_columns("output")
    untokenized_dataset = untokenized_dataset.add_column("outputs", outputs)
    return untokenized_dataset


def _process_doc_prepended_question(doc):
    input = doc["input"]
    split = input.find("\n\n")
    return {
        "id": doc["id"],
        "pid": doc["pid"],
        "input": input,
        "outputs": doc["outputs"],
        "question": input[0:split],
        "text": input[split + 2 :],
    }


def process_doc(doc):
    doc = _process_doc_prepended_question(doc)

    split = doc["text"].find("\n\n", doc["text"].find("(D)"))
    choices_text = doc["text"][:split]

    doc["text"] = doc["text"][split:].strip()
    doc["choices"] = [
        _normalize_answer(choice)
        for choice in re.split(quality_multiple_choice_pattern, choices_text)[1:]
    ]
    doc["gold"] = doc["choices"].index(_normalize_answer(doc["outputs"][0]))
    return doc


def get_quality_dataset():
    """
    download and processes the quality dataset following the lm-evaluation-harness scrolls_quality task

    The processed dataset has the following train & validation splits with 2523 & 2086 examples respectively.
    fields to be used during evaluation:
    - question: the question prompt
    - text: the context
    - choices: list of choices (4 in total)
    - gold: index of the correct choice
    """
    quality_dataset = load_dataset("tau/scrolls", "quality")
    del quality_dataset["test"]  # drop test split -> no ground truths
    for split in quality_dataset:
        quality_dataset[split] = _drop_duplicates_in_input(quality_dataset[split])
    quality_dataset = quality_dataset.map(process_doc)
    print(quality_dataset)
    return quality_dataset


if __name__ == "__main__":
    get_quality_dataset()

# DatasetDict({
#     train: Dataset({
#         features: ['id', 'pid', 'input', 'outputs', 'question', 'text', 'choices', 'gold'],
#         num_rows: 2523
#     })
#     validation: Dataset({
#         features: ['id', 'pid', 'input', 'outputs', 'question', 'text', 'choices', 'gold'],
#         num_rows: 2086
#     })
# })
