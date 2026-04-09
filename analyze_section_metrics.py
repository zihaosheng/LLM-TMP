#!/usr/bin/env python3
"""
Compute BLEU-4 and ROUGE (1/2/L) between `label` and `predict`,
grouped by Section (4-9) extracted from each item's `prompt`.

Input format: JSON Lines, each line like:
{"prompt": "... Section 4 - ...", "predict": "...", "label": "..."}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, Iterable, List, Optional, Tuple

try:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
except Exception:  # pragma: no cover
    sentence_bleu = None  # type: ignore[assignment]
    SmoothingFunction = None  # type: ignore[assignment]

try:
    import jieba  # type: ignore
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # type: ignore
    from rouge_chinese import Rouge  # type: ignore

    jieba.initialize()
except ImportError:
    print("Please install llamafactory with `pip install -e .[metrics]`.")
    raise


SECTION_RE = re.compile(r"\bSection\s*([4-9])\b", flags=re.IGNORECASE)


def extract_section(prompt: str) -> Optional[str]:
    m = SECTION_RE.search(prompt or "")
    if not m:
        return None
    return f"Section{m.group(1)}"


def _tokenize(text: str) -> List[str]:
    """
    Heuristic tokenization:
    - If whitespace exists, split on whitespace.
    - Otherwise, fall back to character-level (helps for CJK/no-spaces text).
    """
    if text is None:
        return []
    # s = text.strip()
    # if not s:
    #     return []
    # if re.search(r"\s", s):
    #     return s.split()
    # return list(s)
    
    return list(jieba.cut(text))


def bleu4_sentence(reference: str, hypothesis: str) -> float:
    """
    BLEU-4 computed by NLTK (sentence_bleu) with smoothing.
    Returns a value in [0, 1].
    """
    if sentence_bleu is None or SmoothingFunction is None:
        raise RuntimeError("nltk is not installed. Please run: pip install nltk")

    ref_toks = _tokenize(reference)
    hyp_toks = _tokenize(hypothesis)
    if not ref_toks or not hyp_toks:
        return 0.0

    chencherry = SmoothingFunction()
    return float(
        sentence_bleu(
            [list(reference)],
            list(hypothesis),
            # weights=(0.25, 0.25, 0.25, 0.25),
            # smoothing_function=chencherry.method1,
            smoothing_function=SmoothingFunction().method3, # method3 is the default method
        )
    ) * 100


def rouge_scores_f1(reference: str, hypothesis: str) -> Tuple[float, float, float]:
    """
    ROUGE computed by rouge-chinese.
    Returns (rouge-1 f, rouge-2 f, rouge-l f) in [0, 1].
    """
    if Rouge is None:
        raise RuntimeError("rouge-chinese is not installed. Please run: pip install rouge-chinese")

    rouge = Rouge()
    # rouge_chinese expects args as (hyps, refs)
    scores = rouge.get_scores(" ".join(_tokenize(hypothesis)) or "", " ".join(_tokenize(reference)) or "")
    if not scores:
        return 0.0, 0.0, 0.0
    s0 = scores[0]
    r1 = float(s0.get("rouge-1", {}).get("f", 0.0)) * 100
    r2 = float(s0.get("rouge-2", {}).get("f", 0.0)) * 100
    rl = float(s0.get("rouge-l", {}).get("f", 0.0)) * 100
    return r1, r2, rl


@dataclass
class Metrics:
    bleu4: float = 0.0
    rouge1: float = 0.0
    rouge2: float = 0.0
    rougel: float = 0.0

    def add(self, other: "Metrics") -> None:
        self.bleu4 += other.bleu4
        self.rouge1 += other.rouge1
        self.rouge2 += other.rouge2
        self.rougel += other.rougel

    def div(self, k: int) -> "Metrics":
        if k <= 0:
            return Metrics()
        return Metrics(
            bleu4=self.bleu4 / k,
            rouge1=self.rouge1 / k,
            rouge2=self.rouge2 / k,
            rougel=self.rougel / k,
        )


def compute_metrics(label: str, predict: str) -> Metrics:
    r1, r2, rl = rouge_scores_f1(label, predict)
    return Metrics(
        bleu4=bleu4_sentence(label, predict),
        rouge1=r1,
        rouge2=r2,
        rougel=rl,
    )


def iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSON decode failed at line {i}: {e}") from e


def fmt(x: float) -> str:
    return f"{x:.2f}"


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default="paper/Qwen2.5-7B-Instruct-1M/alan_predict_2025-04-04/generated_predictions.jsonl",
        help="Path to generated_predictions.jsonl",
    )
    p.add_argument(
        "--output_json",
        default=None,
        help="Optional path to write per-section averages as JSON",
    )
    args = p.parse_args(argv)

    sums: DefaultDict[str, Metrics] = defaultdict(Metrics)
    counts: DefaultDict[str, int] = defaultdict(int)
    all_sum = Metrics()
    all_count = 0
    missing_section = 0
    missing_fields = 0

    for obj in iter_jsonl(args.input):
        prompt = obj.get("prompt", "")
        section = extract_section(prompt)
        if section is None:
            missing_section += 1
            continue

        label = obj.get("label", None)
        pred = obj.get("predict", None)
        if label is None or pred is None:
            missing_fields += 1
            continue

        m = compute_metrics(str(label), str(pred))
        sums[section].add(m)
        counts[section] += 1
        all_sum.add(m)
        all_count += 1

    sections = [f"Section{i}" for i in range(4, 10)]
    print("section    \tcount\tbleu4\trouge1\trouge2\trougel")
    out: Dict[str, Dict[str, float]] = {}
    for sec in sections:
        c = counts.get(sec, 0)
        avg = sums.get(sec, Metrics()).div(c)
        print(
            f"{sec}\t{c}\t{fmt(avg.bleu4)}\t{fmt(avg.rouge1)}\t{fmt(avg.rouge2)}\t{fmt(avg.rougel)}"
        )
        out[sec] = {
            "count": float(c),
            "bleu4": avg.bleu4,
            "rouge1_f1": avg.rouge1,
            "rouge2_f1": avg.rouge2,
            "rougeL_f1": avg.rougel,
        }

    all_avg = all_sum.div(all_count)
    print(
        f"All     \t{all_count}\t{fmt(all_avg.bleu4)}\t{fmt(all_avg.rouge1)}\t{fmt(all_avg.rouge2)}\t{fmt(all_avg.rougel)}"
    )
    out["All"] = {
        "count": float(all_count),
        "bleu4": all_avg.bleu4,
        "rouge1_f1": all_avg.rouge1,
        "rouge2_f1": all_avg.rouge2,
        "rougeL_f1": all_avg.rougel,
    }

    if missing_section or missing_fields:
        print(
            f"\n[warn] skipped: missing_section={missing_section}, missing_label_or_predict={missing_fields}",
            file=sys.stderr,
        )

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

