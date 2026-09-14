"""Touchless Typing Trainer — cluster sequence to English translation.

Subcommands:
  prepare  Convert an English corpus into cluster-digit sequences and build
           the word-frequency vocabulary used by the constrained decoder.
  train    Fine-tune a seq2seq model (BART/T5) on cluster -> English pairs.
  infer    Translate a cluster sequence to English (hybrid decoder if a
           vocab file is available, raw beam search otherwise).
  eval     Measure sentence/word accuracy of the decoder on held-out data.
"""

import argparse
import ast
import csv
import json
import random
import re

import numpy as np
import pandas as pd

from clusters import CLUSTER_MAPS, PREFIX, encode_text

random.seed(13)


def normalize(text):
    """Lowercase and keep only letters, apostrophes, and single spaces.

    The gaze keyboard can only produce letters and spaces, so targets are
    normalized to the same alphabet (apostrophes survive because their
    letters still re-encode to the same cluster code, e.g. don't -> dont).
    """
    text = text.lower()
    text = re.sub(r"[^a-z' ]+", " ", text)
    '''text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(^|\s)'+|'+(\s|$)", " ", text)  # drop free-standing quotes
    return re.sub(r"\s+", " ", text).strip()'''


def prepare_dataset(input_csv, output_csv, num_clusters, max_samples=None):
    df = pd.read_csv(
        input_csv,
        sep="\t",
        header=None,
        usecols=[0],
        names=["text"],
        quoting=csv.QUOTE_NONE,
        on_bad_lines="skip",
    )
    df["text"] = df["text"].astype(str).map(normalize)
    df = df[df["text"].str.contains(r"[a-z]")]
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    if max_samples:
        df = df.head(max_samples)
    df["cluster"] = df["text"].map(lambda t: encode_text(t, num_clusters))

    dataset = []
    for i, row in df.iterrows():
        dataset.append({
            "id": str(i),
            "translation": str({"cluster": row["cluster"], "en": row["text"]}),
        })
    out = pd.DataFrame(dataset)
    out.to_csv(output_csv, header=True, index=False)
    print(f"Saved {len(out)} examples to {output_csv}")

    # Word-frequency vocabulary for the cluster-constrained decoder
    unigrams, bigrams = {}, {}
    for text in df["text"]:
        words = text.split()
        prev = "<s>"
        for w in words:
            unigrams[w] = unigrams.get(w, 0) + 1
            key = f"{prev} {w}"
            bigrams[key] = bigrams.get(key, 0) + 1
            prev = w
    vocab_path = f"vocab-{num_clusters}.json"
    with open(vocab_path, "w") as f:
        json.dump({"num_clusters": num_clusters, "unigrams": unigrams, "bigrams": bigrams}, f)
    print(f"Saved vocabulary ({len(unigrams)} words, {len(bigrams)} bigrams) to {vocab_path}")


def preprocess_function(examples, tokenizer):
    inputs = [PREFIX + ast.literal_eval(ex)["cluster"] for ex in examples["translation"]]
    targets = [ast.literal_eval(ex)["en"] for ex in examples["translation"]]
    return tokenizer(inputs, text_target=targets, max_length=256, truncation=True)


def make_compute_metrics(tokenizer):
    import evaluate

    metric = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = [p.strip() for p in tokenizer.batch_decode(preds, skip_special_tokens=True)]
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = [[l.strip()] for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        exact = np.mean([p == l[0] for p, l in zip(decoded_preds, decoded_labels)])
        gen_len = np.mean([np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds])
        return {"bleu": round(result["score"], 4),
                "exact_match": round(float(exact), 4),
                "gen_len": round(float(gen_len), 4)}

    return compute_metrics


def train(args):
    from datasets import load_dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    dataset = load_dataset("csv", data_files=args.dataset)["train"]
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    dataset = dataset.train_test_split(test_size=0.2, seed=13)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenized = dataset.map(lambda ex: preprocess_function(ex, tokenizer), batched=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        generation_max_length=64,
        fp16=args.fp16,
        push_to_hub=args.push_to_hub,
        report_to=[],
        logging_steps=50,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(tokenizer),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if args.push_to_hub:
        trainer.push_to_hub()


def infer(args):
    if args.vocab:
        from decoder import HybridDecoder

        decoder = HybridDecoder(vocab_path=args.vocab, model_dir=args.model,
                                num_clusters=args.clusters)
        result = decoder.decode(args.text)
        print(f"[{result.method}] consistent={result.consistent}")
        print(result.text)
        return

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    inputs = tokenizer(PREFIX + args.text, return_tensors="pt")
    # deterministic beam search: sampling (top-k/top-p) trades accuracy for variety
    outputs = model.generate(**inputs, max_new_tokens=64, num_beams=5, early_stopping=True)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


def evaluate_decoder(args):
    from decoder import HybridDecoder

    df = pd.read_csv(args.dataset)
    rows = [ast.literal_eval(t) for t in df["translation"]]
    random.shuffle(rows)
    rows = rows[: args.samples]

    decoder = HybridDecoder(vocab_path=args.vocab, model_dir=args.model,
                            num_clusters=args.clusters)
    sent_hits, word_hits, word_total, consistent_count = 0, 0, 0, 0
    methods = {}
    for row in rows:
        result = decoder.decode(row["cluster"])
        methods[result.method] = methods.get(result.method, 0) + 1
        consistent_count += result.consistent
        if result.text == row["en"]:
            sent_hits += 1
        ref, hyp = row["en"].split(), result.text.split()
        word_total += len(ref)
        word_hits += sum(r == h for r, h in zip(ref, hyp))

    n = len(rows)
    print(f"Evaluated {n} held-out sentences (model={'yes' if args.model else 'no'})")
    print(f"  Sentence exact-match : {sent_hits / n:.2%}")
    print(f"  Word accuracy        : {word_hits / word_total:.2%}")
    print(f"  Cluster-consistent   : {consistent_count / n:.2%}")
    print(f"  Decode methods       : {methods}")


def main():
    parser = argparse.ArgumentParser(description="Touchless Typing Trainer - cluster sequence to English translation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare", help="Convert English text to cluster sequences")
    prep.add_argument("--input", default="data/eng-fra-original.csv")
    prep.add_argument("--output", default="translation-cluster-eng-dataset.csv")
    prep.add_argument("--clusters", type=int, default=9, choices=sorted(CLUSTER_MAPS))
    prep.add_argument("--max-samples", type=int, default=None)

    tr = subparsers.add_parser("train", help="Fine-tune a seq2seq model")
    tr.add_argument("--dataset", default="translation-cluster-eng-dataset.csv")
    tr.add_argument("--model", default="facebook/bart-large")
    tr.add_argument("--output-dir", default="cluster_to_text_model")
    tr.add_argument("--epochs", type=int, default=4)
    tr.add_argument("--batch-size", type=int, default=16)
    tr.add_argument("--learning-rate", type=float, default=2e-5)
    tr.add_argument("--max-samples", type=int, default=None)
    tr.add_argument("--fp16", action="store_true")
    tr.add_argument("--push-to-hub", action="store_true")

    inf = subparsers.add_parser("infer", help="Translate a cluster sequence to English")
    inf.add_argument("--model", default=None, help="Fine-tuned model directory or hub id")
    inf.add_argument("--vocab", default=None, help="vocab-N.json for hybrid constrained decoding")
    inf.add_argument("--clusters", type=int, default=9, choices=sorted(CLUSTER_MAPS))
    inf.add_argument("--text", required=True, help="Cluster sequence, e.g. '51663 531 421'")

    ev = subparsers.add_parser("eval", help="Evaluate decoder accuracy on held-out sentences")
    ev.add_argument("--dataset", default="translation-cluster-eng-dataset.csv")
    ev.add_argument("--vocab", default="vocab-9.json")
    ev.add_argument("--model", default=None)
    ev.add_argument("--clusters", type=int, default=9, choices=sorted(CLUSTER_MAPS))
    ev.add_argument("--samples", type=int, default=500)

    args = parser.parse_args()
    if args.command == "prepare":
        prepare_dataset(args.input, args.output, args.clusters, args.max_samples)
    elif args.command == "train":
        train(args)
    elif args.command == "infer":
        if not args.model and not args.vocab:
            parser.error("infer requires --model and/or --vocab")
        infer(args)
    elif args.command == "eval":
        evaluate_decoder(args)


if __name__ == "__main__":
    main()
