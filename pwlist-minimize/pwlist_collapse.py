#!/usr/bin/env python3
"""
Collapse an expanded password wordlist into a compact seed list.

Goal: given a large set of known passwords (e.g., RockYou/Seclists variants),
produce a small base wordlist such that applying common password mutations
(as used by John the Ripper jumbo rules and similar) yields coverage close to
the original expanded list.

Written by opencode: GPT-5-Nano
"""

import argparse
import gzip
import os
import random
import json
from typing import Iterable, List, Set, Dict, Tuple, Optional

# Optional offline LM support (transformers) for scoring seed mutations
LM_LOADED = False
TOKENIZER = None
MODEL = None
def _load_llm(model_path: str, cache_dir: Optional[str] = None):
    global LM_LOADED, TOKENIZER, MODEL
    if not model_path:
        LM_LOADED = False
        return None
    try:
        import importlib
        transformers = importlib.import_module("transformers")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        TOKENIZER = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        MODEL = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)
        MODEL.eval()
        LM_LOADED = True
        return (TOKENIZER, MODEL)
    except Exception:
        LM_LOADED = False
        return None

def _llm_score_string(tokenizer, model, text: str) -> float:
    try:
        # simple causal LM log-likelihood scoring: sum log P(next_token | prefix)
        # tokenize into ids; ensure we work with small strings
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= 1:
            return -1e9
        import torch
        device = next(model.parameters()).device
        with torch.no_grad():
            score = 0.0
            for i in range(1, len(ids)):
                ctx_ids = torch.tensor([ids[:i]], device=device)
                outputs = model(ctx_ids)
                logits = outputs.logits[0, -1, :]
                probs = torch.softmax(logits, dim=-1)
                score += float(torch.log(probs[ids[i]]).item())
            return score
    except Exception:
        return -1e9


def _read_lines(path: str, sample_fraction: float = 1.0) -> Iterable[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
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


def _leet(word: str, max_subs: int = 2) -> Set[str]:
    mapping = {'a': '4', 'e': '3', 'i': '1', 'o': '0', 's': '5', 't': '7', 'g': '9'}
    indices = [i for i, ch in enumerate(word.lower()) if ch in mapping]
    variants = {word}
    for idx in indices:
        ch = word[idx]
        mapped = mapping[ch.lower()]
        variants.add(word[:idx] + mapped + word[idx+1:])
    if max_subs >= 2:
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                wlist = list(word)
                wlist[indices[i]] = mapping[word[indices[i]].lower()]
                wlist[indices[j]] = mapping[word[indices[j]].lower()]
                variants.add(''.join(wlist))
    return set(list(variants))


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
        variants.update({s + word, word + s, s + '_' + word, word + '_' + s})
    return variants


def _suffix_variants(word: str) -> Set[str]:
    suff = ["123", "1234", "!", "!@#", "$", "01"]
    return {word + s for s in suff}


def _prefix_variants(word: str) -> Set[str]:
    pref = ["password", "admin", "letmein", "guest", "user"]
    return {p + word for p in pref}


def _repeat_variant(word: str) -> Set[str]:
    return {''.join([c*2 for c in word])}


def _generate_variants_for_word(word: str, top_k: int = 8) -> List[str]:
    variants: Set[str] = set()
    variants.update(_leet(word, max_subs=2))
    variants.update(_sound_variants(word))
    variants.update(_common_subword_variants(word))
    variants.update(_suffix_variants(word))
    variants.update(_prefix_variants(word))
    variants.update(_repeat_variant(word))
    variants = {v for v in variants if v and v.lower() != word.lower()}
    return sorted(list(variants))[:max(1, top_k)]


def main():
    parser = argparse.ArgumentParser(description="Collapse a large expanded password wordlist into a compact seedlist using mutation rules.")
    parser.add_argument('--input', '-i', required=True, help='Path to input wordlist (text or gzip)')
    parser.add_argument('--output', '-o', required=True, help='Path to output seed wordlist')
    parser.add_argument('--target-coverage', type=float, default=0.9, help='Target coverage of input words by seed mutations (0-1)')
    parser.add_argument('--max-seeds', type=int, default=1000, help='Maximum number of seed words to output')
    parser.add_argument('--max-variants-per-seed', type=int, default=16, help='Max number of mutation variants generated per seed')
    parser.add_argument('--sample', type=float, default=1.0, help='Fraction of input to sample (0.0-1.0)')
    parser.add_argument('--seed-sample', type=int, default=20000, help='Number of seed candidates to consider (sampling from input)')
    parser.add_argument('--report', help='Optional path to write a JSON report of coverage')
    parser.add_argument('--llm-model', dest='llm_model', default=None, help='Local Transformer-based LM model path or HF model name (offline download to cache first)')
    parser.add_argument('--llm-max-seeds', dest='llm_max_seeds', type=int, default=1000, help='Max seeds when LM scoring is enabled')
    parser.add_argument('--llm-cache-dir', dest='llm_cache_dir', default=None, help='Directory to cache/download LM files (transformers cache)')
    args = parser.parse_args()

    # Read input words (lowercased for normalization)
    words = []
    for w in _read_lines(args.input, sample_fraction=args.sample):
        words.append(w.lower())
    if not words:
        print("No words found in input.")
        return
    input_set = set(words)

    # Seed candidate selection: sample a subset if the input is large
    if len(words) > args.seed_sample:
        random.seed(0)
        seed_candidates = random.sample(words, args.seed_sample)
    else:
        seed_candidates = list(dict.fromkeys(words))  # preserve order and dedupe

    # Build variant closures for candidates (limited per-seed)
    seed_variants: Dict[str, Set[str]] = {}
    for s in seed_candidates:
        vset = set(_generate_variants_for_word(s, top_k=args.max_variants_per_seed))
        # Include the seed itself if it exists in the input to help coverage
        if s in input_set:
            vset.add(s)
        seed_variants[s] = vset

    # LM setup (optional)
    lm_tokenizer = None
    lm_model = None
    if getattr(args, 'llm_model', None):
        llm = _load_llm(args.llm_model, cache_dir=getattr(args, 'llm_cache_dir', None))
        if llm:
            lm_tokenizer, lm_model = llm
        else:
            print("Warning: failed to load LM model; proceeding with deterministic collapse.")
            lm_tokenizer = None
            lm_model = None

    # Greedy set cover: pick seeds that cover as many uncovered words as possible
    uncovered = set(input_set)
    seeds: List[str] = []

    # Pre-compute quick lookup: for each seed, which input words can it cover (intersection)
    seed_coverage: Dict[str, Set[str]] = {}
    for s, vs in seed_variants.items():
        covered = set()
        for w in vs:
            if w in input_set:
                covered.add(w)
        seed_coverage[s] = covered
    # If LM is available, precompute per-seed LM scores over expansion set
    seed_lm_score: Dict[str, float] = {}
    if lm_tokenizer is not None and lm_model is not None:
        # Load the expanded set into memory for LM scoring references
        # expanded_set is the same as input_set here since input is the expanded list
        expanded_set = set(input_set)
        for s in seed_candidates:
            cov = seed_coverage.get(s, set())
            # compute total lm score for coverage words
            total = 0.0
            count = 0
            for w in seed_variants.get(s, set()):
                if w in expanded_set:
                    sc = _llm_score_string(lm_tokenizer, lm_model, w)
                    if sc > -1e8:
                        total += sc
                        count += 1
            seed_lm_score[s] = (total if count > 0 else 0.0) / (count if count>0 else 1)
    else:
        for s in seed_candidates:
            seed_lm_score[s] = 0.0

    # Normalize LM scores to a 0..1 range for weighting
    lm_vals = [v for v in seed_lm_score.values()]
    if lm_vals:
        lm_min, lm_max = min(lm_vals), max(lm_vals)
        norm_fn = (lambda x: 0.0) if lm_max == lm_min else (lambda x: (x - lm_min) / (lm_max - lm_min))
    else:
        norm_fn = lambda x: 0.0

    while uncovered and len(seeds) < getattr(args, 'llm_max_seeds', args.max_seeds):
        # pick the seed with the largest intersection with uncovered
        best_seed = None
        best_cov = set()
        best_count = 0
        best_priority = -1.0
        for s, cov in seed_coverage.items():
            if s in seeds:
                continue
            c = len(cov & uncovered)
            lm_norm = norm_fn(seed_lm_score.get(s, 0.0))
            priority = c * (1.0 + lm_norm)
            if priority > best_priority:
                best_priority = priority
                best_count = c
                best_seed = s
                best_cov = cov
        if best_seed is None or best_count == 0:
            # No more coverage possible with remaining seeds
            break
        seeds.append(best_seed)
        uncovered -= (best_cov & uncovered)

    # Write seeds to output (lowercase by design)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as out:
        for s in seeds:
            out.write(s + '\n')

    # Optional reporting: coverage stats
    covered_count = len(input_set) - len(uncovered)
    coverage = covered_count / len(input_set)
    print(f"Seeds: {len(seeds)}; Coverage: {coverage:.4f} ({covered_count}/{len(input_set)})")
    if args.report:
        import json
        report = {
            'input_size': len(input_set),
            'seed_count': len(seeds),
            'coverage': coverage,
            'covered_words': sorted(list(input_set - uncovered))[:1000],  # first 1000 examples
        }
        with open(args.report, 'w', encoding='utf-8') as rf:
            json.dump(report, rf, indent=2)


if __name__ == '__main__':
    main()
