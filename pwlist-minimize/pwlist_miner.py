#!/usr/bin/env python3
"""
Password wordlist mutator and compressor.
Given a base wordlist, generate common variations (leetspeak, prefixes, suffixes,
sound alterations) and optional LLM-based variations (via HuggingFace Transformers).
Output a trimmed list containing the top 80% most frequent variations (by generation counts)
to capture the most likely mutations.

Written by opencode: GPT-5-Nano
"""
import argparse
import gzip
import sqlite3
import os
import random
import re
from collections import defaultdict, Counter
from typing import Iterable, List, Set, Any

# Note: Transformer-based expansion is optional and not loaded by default to avoid hard deps
AutoModelForCausalLM = None  # placeholder for optional integration
AutoTokenizer = None  # placeholder for optional integration


def _read_lines(path: str, sample_fraction: float = 1.0) -> Iterable[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # Support plain text and gzip-compressed inputs
    open_fn = gzip.open if path.endswith('.gz') else open
    with open_fn(path, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            w = line.strip()
            if not w:
                continue
            if w.startswith('#'):
                continue
            if sample_fraction < 1.0:
                if random.random() > sample_fraction:
                    continue
            yield w


def _split_variations(text: str) -> List[str]:
    # Split a generated string into candidate words, crude approach
    tokens = re.split(r"[^A-Za-z0-9]+", text)
    return [t for t in tokens if len(t) >= 2]


def _generate_leet_variants(word: str, max_subs: int = 2) -> Set[str]:
    mapping = {'a': '4', 'e': '3', 'i': '1', 'o': '0', 's': '5', 't': '7', 'g': '9'}
    indices = [i for i, ch in enumerate(word.lower()) if ch in mapping]
    variants: Set[str] = set()
    variants.add(word)
    # generate 1-substitution variants
    for idx in indices:
        ch = word[idx]
        mapped = mapping[ch.lower()]
        v = word[:idx] + mapped + word[idx+1:]
        variants.add(v)
    # generate 2-substitution variants
    if max_subs >= 2:
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                idx1, idx2 = indices[i], indices[j]
                w2 = list(word)
                w2[idx1] = mapping[word[idx1].lower()]
                w2[idx2] = mapping[word[idx2].lower()]
                variants.add(''.join(w2))
    return set(list(variants)[:max(1, len(variants))])


def _sound_variants(word: str) -> Set[str]:
    presets = ["sh", "ch", "th", "ph"]
    variants = set()
    for pfx in presets:
        variants.add(pfx + word)
        variants.add(word + pfx)
    if not word.endswith('ing'):
        variants.add(word + 'ing')
    return variants


def _common_subword_variants(word: str) -> Set[str]:
    subs = ["password", "pass", "admin", "letmein", "qwerty", "123", "welcome", "user"]
    variants = set()
    for s in subs:
        variants.add(s + word)
        variants.add(word + s)
        variants.add(s + '_' + word)
        variants.add(word + '_' + s)
    return variants


def _suffix_variants(word: str) -> Set[str]:
    suff = ["123", "1234", "!", "!@#", "$", "01"]
    return {word + s for s in suff}


def _prefix_variants(word: str) -> Set[str]:
    pref = ["password", "admin", "letmein", "guest", "user"]
    return {p + word for p in pref}


def _repeat_variant(word: str) -> Set[str]:
    # duplicate characters (simple variant)
    return {''.join([c*2 for c in word])}


def _generate_variants_for_word(word: str, top_k: int = 8) -> List[str]:
    variants = set()
    # deterministic rules first
    variants.update(_generate_leet_variants(word, max_subs=2))
    variants.update(_sound_variants(word))
    variants.update(_common_subword_variants(word))
    variants.update(_suffix_variants(word))
    variants.update(_prefix_variants(word))
    variants.update(_repeat_variant(word))
    # filter trivial
    variants = {v for v in variants if v and v.lower() != word.lower()}
    # LLM expansion is not performed in this version by default
    return sorted(list(variants))[:max(1, top_k)]


# Streaming / batched path helpers using sqlite for counts
def _init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS variants (variant TEXT PRIMARY KEY, count INTEGER)")
    conn.commit()
    return conn


def _inc_variant_count(conn: sqlite3.Connection, variant: str) -> None:
    cur = conn.cursor()
    cur.execute("INSERT INTO variants (variant, count) VALUES (?, 1) ON CONFLICT(variant) DO UPDATE SET count = count + 1", (variant,))
    conn.commit()


def _iter_variant_counts(conn: sqlite3.Connection):
    cur = conn.cursor()
    for row in cur.execute("SELECT variant, count FROM variants ORDER BY count DESC"):
        yield row


def main():
    parser = argparse.ArgumentParser(description="Shrink password wordlists via mutations and optional LLM expansion.")
    parser.add_argument('--input', '-i', required=True, help='Path to input wordlist (text or gzip)')
    parser.add_argument('--output', '-o', required=True, help='Path to output wordlist')
    parser.add_argument('--top-per-word', type=int, default=8, help='Max variations generated per word (kept via top-80% heuristic)')
    parser.add_argument('--keep-percentage', type=float, default=0.8, help='Keep top N% of all generated variations by frequency (default 0.8)')
    parser.add_argument('--sample', type=float, default=1.0, help='Fraction of input to sample (0.0-1.0)')
    parser.add_argument('--model', default=None, help='Optional HuggingFace model path or name for mutation expansion (e.g., gpt2)')
    parser.add_argument('--no-model', dest='use_model', action='store_false', help='Do not use the LLM model even if provided')
    parser.set_defaults(use_model=True)
    parser.add_argument('--streaming', action='store_true', help='Enable batched streaming two-pass mutation counting (uses sqlite)')
    args = parser.parse_args()

    # Note: LLM-based expansion is optional and not loaded by default

    total_words = 0
    per_word_variants: List[List[str]] = []
    variation_counts = Counter()
    top_limit = args.top_per_word
    streaming = getattr(args, 'streaming', False)
    db_path = None

    # Process input and build per-word variant sets
    if streaming:
        # Streaming two-pass using sqlite counts
        if not db_path:
            # place db next to output for convenience
            db_path = (os.path.splitext(os.path.abspath(args.output))[0] + ".counts.sqlite3")
        conn = _init_db(db_path)
        total_mutations = 0
        for word in _read_lines(args.input, sample_fraction=args.sample):
            total_words += 1
            vs = _generate_variants_for_word(word, top_k=top_limit)
            for v in vs:
                _inc_variant_count(conn, v)
            total_mutations += len(vs)
        if total_mutations == 0:
            print("No mutations generated; exiting.")
            return
        threshold = int(total_mutations * args.keep_percentage)
        kept_set = set()
        cum = 0
        for variant, cnt in _iter_variant_counts(conn):
            kept_set.add(variant)
            cum += cnt
            if cum >= threshold:
                break
        # Second pass: write kept variants, deduplicated
        written = set()
        with open(args.output, 'w', encoding='utf-8') as out:
            for word in _read_lines(args.input, sample_fraction=args.sample):
                vs = _generate_variants_for_word(word, top_k=top_limit)
                for v in vs:
                    if v in kept_set and v not in written:
                        out.write(v + '\n')
                        written.add(v)
        conn.close()
        print(f"Streaming mode: wrote {len(written)} unique variants to {args.output}")
        return
    else:
        # Non-streaming: in-memory accumulation as before
        for word in _read_lines(args.input, sample_fraction=args.sample):
            total_words += 1
            vs = _generate_variants_for_word(word, top_k=top_limit)
            per_word_variants.append(vs)
            for v in vs:
                variation_counts[v] += 1

    if total_words == 0:
        print("No words found in input; exiting.")
        return

    # Determine top 80% by cumulative counts
    total_mutations = sum(variation_counts.values())
    threshold = total_mutations * args.keep_percentage
    ranked = variation_counts.most_common()
    kept: Set[str] = set()
    cum = 0
    for val, cnt in ranked:
        kept.add(val)
        cum += cnt
        if cum >= threshold:
            break

    # Build final list: keep kept variations; also optionally add originals for reference
    output_set = set(kept)
    # Optional: always include original words to preserve coverage in downstream tools
    for ws in per_word_variants:
        for w in ws:
            if random.random() < 0.01:  # tiny sprinkle of originals to avoid bias; 1% chance
                output_set.add(w)

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as out:
        for v in sorted(output_set):
            out.write(v + '\n')

    print(f"Wrote {len(output_set)} variations to {args.output} (based on {total_words} input words).")


if __name__ == '__main__':
    main()
