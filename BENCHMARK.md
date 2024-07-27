# Description of Long-Context Tasks in the Eval Harness

These tasks can be found in `./tasks.py` and are invoked from the `eval.py` harness with the `--tasks` parameter.

## Synthetic

### [RULER](https://arxiv.org/abs/2404.06654)

RULER defines a set of synthetic tasks designed to test a model’s long-context understanding.

Tasks include needle in a haystack (NIAH), variable tracking (VT), question answering (QA), and common word extraction (CWE).

## Domain-Specific

### [Dolomites](https://arxiv.org/abs/2405.05938)

Evaluates the model’s ability to perform domain-specific methodical writing tasks such as writing a differential diagnosis for a patient, or writing a lesson plan for students.

## Coding

### [RepoBench](https://arxiv.org/abs/2306.03091)

This task tests the model’s ability to understand coding repositories and make correct predictions for code completion.

## QA

### [MuSiQue](https://arxiv.org/abs/2108.00573)

MuSiQue is a question-answering dataset that tests the model’s ability to perform multihop reasoning over a long input context.

## [TruthfulQA](https://arxiv.org/abs/2109.07958)

TruthfulQA tests the models ability to answer questions truthfully across a broad set of categories such as health, law, finance, and politics.

## Language Modeling

### [PG19](https://github.com/google-deepmind/pg19)

This task tests the model’s ability to generate longform text (~8K tokens) by providing a title and first initial words of a book.

## Summarization

## [QMSum](https://arxiv.org/abs/2104.05938)

A meeting summarization dataset that evaluates the model’s ability to select and summarize content that is relevant to the given query.

## [SQuALITY](https://arxiv.org/abs/2205.11465)

SQuALITY is a question-focused summarization dataset, which tests the models ability to understand long narratives and select and summarize content relevant to the provided question.

## [QuALITY](https://arxiv.org/abs/2112.08608v2)

QuALITY tests the model’s ability to understand and answer questions about long narratives.
