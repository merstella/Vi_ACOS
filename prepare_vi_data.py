# coding=utf-8

import argparse
import os
import unicodedata
from collections import defaultdict

from datasets import load_from_disk
from transformers import AutoTokenizer
from pyvi import ViTokenizer


SPLIT_MAP = {
    "train": "train",
    "validation": "dev",
    "test": "test",
}

SENTIMENT_MAP = {
    "TIÊU_CỰC": 0,
    "TRUNG_TÍNH": 1,
    "TÍCH_CỰC": 2,
}


def _strip_accents(text):
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")


def normalize_sentiment(sentiment):
    if sentiment is None:
        return None
    raw = sentiment.strip().upper()
    raw = raw.replace(" ", "_").replace("-", "_")
    if raw in SENTIMENT_MAP:
        return SENTIMENT_MAP[raw]
    no_accent = _strip_accents(raw)
    if no_accent in SENTIMENT_MAP:
        return SENTIMENT_MAP[no_accent]
    alt = {
        "TIEU_CUC": 0,
        "TRUNG_TINH": 1,
        "TICH_CUC": 2,
    }
    if no_accent in alt:
        return alt[no_accent]
    raise ValueError("Unknown sentiment: {}".format(sentiment))


def is_implicit(text):
    if text is None:
        return True
    raw = text.strip()
    if not raw:
        return True
    return raw.upper() in {"NULL", "NONE", "N/A", "NA", "NA.", "_", "-"}


def vn_tokenize(text):
    tokenized = ViTokenizer.tokenize(text or "")
    return tokenized.split()


def normalize_token(token):
    return token.lower()


def find_spans(tokens, phrase_tokens):
    spans = []
    if not phrase_tokens:
        return spans
    n = len(phrase_tokens)
    for i in range(len(tokens) - n + 1):
        if tokens[i:i + n] == phrase_tokens:
            spans.append((i, i + n))
    return spans


def choose_closest_span(aspect_spans, opinion_spans):
    best = None
    best_pair = (None, None)
    for a_span in aspect_spans:
        for o_span in opinion_spans:
            dist = abs(a_span[0] - o_span[0])
            key = (dist, a_span[0], o_span[0])
            if best is None or key < best:
                best = key
                best_pair = (a_span, o_span)
    return best_pair


def bpe_tokenize_words(words, tokenizer):
    bpe_tokens = []
    word_to_bpe = []
    for word in words:
        pieces = tokenizer.tokenize(word)
        if not pieces:
            pieces = [tokenizer.unk_token]
        start = len(bpe_tokens)
        bpe_tokens.extend(pieces)
        end = len(bpe_tokens)
        word_to_bpe.append((start, end))
    return bpe_tokens, word_to_bpe


def map_word_span_to_bpe(span, word_to_bpe):
    if span is None:
        return (-1, -1)
    start, end = span
    if start < 0 or end <= start:
        return (-1, -1)
    bpe_start = word_to_bpe[start][0]
    bpe_end = word_to_bpe[end - 1][1]
    return (bpe_start, bpe_end)


def build_phrase_cache():
    cache = {}

    def _get_tokens(phrase):
        if phrase in cache:
            return cache[phrase]
        tokens = vn_tokenize(phrase)
        cache[phrase] = tokens
        return tokens

    return _get_tokens


def process_split(dataset, split_name, out_dir, domain, tokenizer):
    split_key = SPLIT_MAP.get(split_name, split_name)
    quad_path = os.path.join(out_dir, "{}_{}_quad_bert.tsv".format(domain, split_key))
    pair_path = os.path.join(out_dir, "{}_{}_pair.tsv".format(domain, split_key))

    os.makedirs(out_dir, exist_ok=True)

    missing_aspect = 0
    missing_opinion = 0
    phrase_tokenizer = build_phrase_cache()
    categories = set()

    with open(quad_path, "w", encoding="utf-8") as quad_f, open(pair_path, "w", encoding="utf-8") as pair_f:
        for idx, example in enumerate(dataset):
            text = example.get("text", "")
            quads = example.get("labels", []) or []

            word_tokens = vn_tokenize(text)
            word_tokens_norm = [normalize_token(t) for t in word_tokens]
            bpe_tokens, word_to_bpe = bpe_tokenize_words(word_tokens, tokenizer)

            quad_entries = []
            quad_seen = set()
            pair_labels = defaultdict(set)

            for quad in quads:
                if len(quad) != 4:
                    continue
                aspect, category, polarity, opinion = quad
                categories.add(category)
                senti_id = normalize_sentiment(polarity)

                aspect_span = None
                opinion_span = None

                if not is_implicit(aspect):
                    aspect_tokens = phrase_tokenizer(aspect)
                    aspect_norm = [normalize_token(t) for t in aspect_tokens]
                    aspect_spans = find_spans(word_tokens_norm, aspect_norm)
                else:
                    aspect_spans = []

                if not is_implicit(opinion):
                    opinion_tokens = phrase_tokenizer(opinion)
                    opinion_norm = [normalize_token(t) for t in opinion_tokens]
                    opinion_spans = find_spans(word_tokens_norm, opinion_norm)
                else:
                    opinion_spans = []

                if aspect_spans and opinion_spans:
                    aspect_span, opinion_span = choose_closest_span(aspect_spans, opinion_spans)
                elif aspect_spans:
                    aspect_span = aspect_spans[0]
                    opinion_span = None
                elif opinion_spans:
                    opinion_span = opinion_spans[0]
                    aspect_span = None

                if not aspect_spans and not is_implicit(aspect):
                    missing_aspect += 1
                if not opinion_spans and not is_implicit(opinion):
                    missing_opinion += 1

                bpe_aspect = map_word_span_to_bpe(aspect_span, word_to_bpe)
                bpe_opinion = map_word_span_to_bpe(opinion_span, word_to_bpe)

                aspect_str = "{},{}".format(bpe_aspect[0], bpe_aspect[1])
                opinion_str = "{},{}".format(bpe_opinion[0], bpe_opinion[1])
                quad_entry = "{} {} {} {}".format(aspect_str, category, senti_id, opinion_str)
                if quad_entry not in quad_seen:
                    quad_entries.append(quad_entry)
                    quad_seen.add(quad_entry)

                pair_key = (aspect_str, opinion_str)
                pair_labels[pair_key].add("{}#{}".format(category, senti_id))

            if not quad_entries or not bpe_tokens:
                continue

            token_str = " ".join(bpe_tokens)
            quad_f.write(token_str + "\t" + "\t".join(quad_entries) + "\n")

            for (aspect_str, opinion_str), labels in pair_labels.items():
                label_str = " ".join(sorted(labels))
                pair_f.write("{}####{} {}\t{}\n".format(token_str, aspect_str, opinion_str, label_str))

    return categories, missing_aspect, missing_opinion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to load_from_disk output (e.g., data_vi_normalized)")
    parser.add_argument("--out_dir", required=True, help="Output directory for tokenized_data")
    parser.add_argument("--domain", default="vi", help="Domain name for output files")
    parser.add_argument("--phobert_model", default="vinai/phobert-base", help="Model name or path")
    args = parser.parse_args()

    dataset = load_from_disk(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.phobert_model, use_fast=False)

    all_categories = set()
    total_missing_aspect = 0
    total_missing_opinion = 0

    for split in dataset.keys():
        categories, miss_a, miss_o = process_split(
            dataset[split], split, args.out_dir, args.domain, tokenizer
        )
        all_categories.update(categories)
        total_missing_aspect += miss_a
        total_missing_opinion += miss_o

    categories_path = os.path.join(args.out_dir, "{}_categories.txt".format(args.domain))
    with open(categories_path, "w", encoding="utf-8") as f:
        for cate in sorted(all_categories):
            f.write(cate + "\n")

    print("Done. categories =", len(all_categories))
    print("Missing aspect spans =", total_missing_aspect)
    print("Missing opinion spans =", total_missing_opinion)


if __name__ == "__main__":
    main()
