import argparse
import os

from claudette import models as anthropic_models, Chat

import torch
from data_utils import BENCHMARKS
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


if os.path.exists("/workspace/.cache/"):
    os.environ["HF_DATASETS_CACHE"] = "/workspace/.cache/huggingface/datasets"
    os.environ["TRANSFORMERS_CACHE"] = "/workspace/.cache/huggingface/transformers"

assert "ANTHROPIC_API_KEY" in os.environ, "Please set the ANTHROPIC_API_KEY environment variable."


PROMPT_TEMPLATES = {
    # "triviaqa": "Compress the information in the retrieved documents into a 1-3 sentence summary " \
    #     "that could be used to answer the question: %s\n\nRetrieved Documents:\n%s"
    "triviaqa": "Compress the information in the retrieved documents into a 1-3 sentences " \
        "such that it includes only information relevant to answering the question: %s\n\nRetrieved Documents:\n%s"
}


PREFILL = "Compressed Documents: "

# We will evaluate each summary based on if it improves
# Downstream performance
# p(answer|summarized prompt) - p(answer|original prompt)
SCORER_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # "microsoft/Phi-3-mini-128k-instruct"


def compute_likelihoods(ctx, q, answers, instruction, tokenizer, scorer, max_ctx_len=None, batch_size=8):
    # We need enough tokens for the instruction, question, and answer (subtract max context by 1024 to be safe)
    max_ctx_len = max_ctx_len or tokenizer.model_max_length - 1024
    ctx_tokens = len(tokenizer.encode(ctx))
    if ctx_tokens > max_ctx_len:
        ctx_words = ctx.split(" ")
        keep_num_words = round(len(ctx_words) * max_ctx_len / ctx_tokens)
        ctx = " ".join(ctx_words[:keep_num_words])

    prefill = "Answer:\n"
    chat = [
        {"role": "user", "content": f"{instruction}\n\n{ctx}\n\n{q}"},
    ]

    prompt = tokenizer.apply_chat_template(chat, tokenize=False) + prefill
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    n = len(prompt_ids)

    ans_ids = [tokenizer.encode(ans, add_special_tokens=False) for ans in answers]
    input_ids = [prompt_ids + ans_id for ans_id in ans_ids]
    max_len = max(map(len, input_ids))

    # Pad the input_ids up to max_len
    input_ids = torch.LongTensor([ids + [tokenizer.pad_token_id] * (max_len - len(ids)) for ids in input_ids])

    labels = input_ids.clone()
    # No Loss on the prompt
    labels[:, :n] = -100
    is_pad = input_ids == tokenizer.pad_token_id
    labels[is_pad] = -100

    input_ids = input_ids.to(scorer.device)
    labels = labels.to(scorer.device)

    logits = []
    for i in range(0, len(input_ids), batch_size):
        with torch.no_grad():
            logits.append(scorer(input_ids[i:i+batch_size]).logits)
    logits = torch.cat(logits, dim=0)

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')

    scores = []
    for i in range(len(ans_ids)):
        _shift_logits = shift_logits[i].view(-1, scorer.config.vocab_size)
        _shift_labels = shift_labels[i].view(-1)
        ll = (- loss_fct(_shift_logits, _shift_labels)).item()
        scores.append(ll)

    return {"max": max(scores), "mean": sum(scores) / len(scores)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, default="haiku", choices=["haiku", "sonnet", "opus"])
    parser.add_argument("--dataset", type=str, default="triviaqa")
    parser.add_argument("--min_toks", type=int, default=50, help="If context has < min_toks, there is no need to summarize it. The summary is itself.")
    parser.add_argument("--batch_size", type=int, default=2, help="Max batch size for the scorer.")
    parser.add_argument("--max_ctx_len", type=int, default=None, help="An optional override to max context window of model.")
    args = parser.parse_args()

    model = [m for m in anthropic_models if args.name in str(m)][0]

    dataset = BENCHMARKS[args.dataset]()

    train = dataset.get_train()

    scorer = AutoModelForCausalLM.from_pretrained(
        SCORER_MODEL,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).eval().to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(SCORER_MODEL)

    for ex in tqdm(train):
        # Claudette is stateful so you need to re-instantiate each time
        chat = Chat(model, sp="""You are a helpful and concise assistant.""")

        q, ctx = dataset.question(ex), dataset.context(ex)
        ctx_toks = len(ctx.split(" "))

        if ctx_toks < args.min_toks:
            summary = ctx
        else:
            # prompt = PROMPT_TEMPLATES[args.dataset] % (q, ctx)

            # summary = chat(prompt, prefill=PREFILL).content[0].text

            # assert summary.startswith(PREFILL)
            # summary = summary[len(PREFILL):].strip()
            summary = "This is a summary."

            answers = dataset.answer(ex)

            if type(answers) == str:
                answers = [answers]
            
            original_scores = compute_likelihoods(
                ctx, q, answers, dataset.instruction(), tokenizer, scorer,
                max_ctx_len=args.max_ctx_len, batch_size=args.batch_size
            )
            compressed_scores = compute_likelihoods(
                summary, q, answers, dataset.instruction(), tokenizer, scorer,
                max_ctx_len=args.max_ctx_len, batch_size=args.batch_size
            )

            print("Original")
            print(original_scores)

            print("Compressed")
            print(compressed_scores)
