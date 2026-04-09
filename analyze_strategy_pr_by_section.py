#!/usr/bin/env python3
"""
从 generated_predictions.jsonl 中按 Section 统计 Strategy 集合的 precision / recall。

仅处理 Section 4、7、8（prompt 中以「Section N - ...」标识任务段）。
Strategy 来自 markdown 表格第一列（表头为 Strategy），并对单元格做 <br> 等清洗后比较。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Optional, Set, Tuple

SECTION_RE = re.compile(r"\bSection\s*([4-9])\b", flags=re.IGNORECASE)
BR_RE = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)


def _is_separator_row(line: str) -> bool:
    s = line.strip()
    if not s.startswith("|"):
        return False
    for ch in s:
        if ch not in "|-: \t":
            return False
    return "-" in s


def extract_section(prompt: str) -> Optional[str]:
    m = SECTION_RE.search(prompt or "")
    if not m:
        return None
    return f"Section{m.group(1)}"


def _split_table_row(line: str) -> List[str]:
    line = line.strip()
    if not line.startswith("|"):
        return []
    return [c.strip() for c in line.strip().strip("|").split("|")]


def extract_strategies(text: str) -> Set[str]:
    """从文本中解析所有「首列为 Strategy」的 markdown 表格的第一列数据行。"""
    if not text:
        return set()
    text = BR_RE.sub(" ", text)
    lines = text.splitlines()
    out: Set[str] = set()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped.startswith("|"):
            i += 1
            continue
        cells = _split_table_row(stripped)
        if not cells:
            i += 1
            continue
        if cells[0].lower() != "strategy":
            i += 1
            continue
        i += 1
        if i < len(lines) and _is_separator_row(lines[i]):
            i += 1
        while i < len(lines):
            row_line = lines[i]
            if "--- End of Section" in row_line:
                break
            rs = row_line.strip()
            if not rs.startswith("|"):
                break
            rcells = _split_table_row(rs)
            if rcells and rcells[0]:
                first = rcells[0]
                if re.match(r"^[-:\s]+$", first):
                    i += 1
                    continue
                if first.lower() == "strategy":
                    i += 1
                    continue
                norm = " ".join(first.split())
                if norm:
                    out.add(norm)
            i += 1
        continue
    return out


def precision_recall(pred: Set[str], label: Set[str]) -> Tuple[float, float]:
    """集合意义下的 precision / recall（逐样本集合匹配）。"""
    if not pred and not label:
        return 1.0, 1.0
    if not pred or not label:
        return 0.0, 0.0
    inter = len(pred & label)
    return inter / len(pred), inter / len(label)


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


TARGET_SECTIONS = {"Section4", "Section7", "Section8"}


def main() -> int:
    p = argparse.ArgumentParser(description="按 Section 统计 Strategy precision/recall 均值")
    p.add_argument(
        "--input",
        default="paper/Qwen2.5-7B-Instruct-1M/alan_predict_2025-04-04/generated_predictions.jsonl",
        help="generated_predictions.jsonl 路径",
    )
    p.add_argument(
        "--json-out",
        default=None,
        help="可选：将 per-section 结果写入 JSON",
    )
    args = p.parse_args()

    sums_p: DefaultDict[str, float] = defaultdict(float)
    sums_r: DefaultDict[str, float] = defaultdict(float)
    counts: DefaultDict[str, int] = defaultdict(int)
    skipped_section = 0
    missing_section = 0

    for obj in iter_jsonl(args.input):
        prompt = obj.get("prompt", "")
        sec = extract_section(prompt)
        if sec is None:
            missing_section += 1
            continue
        if sec not in TARGET_SECTIONS:
            skipped_section += 1
            continue

        label = extract_strategies(str(obj.get("label", "")))
        pred = extract_strategies(str(obj.get("predict", "")))
        pr, rc = precision_recall(pred, label)
        sums_p[sec] += pr
        sums_r[sec] += rc
        counts[sec] += 1

    print("Section   \tcount\tavg_precision\tavg_recall")
    out: Dict[str, Dict[str, float]] = {}
    for sec in sorted(TARGET_SECTIONS):
        c = counts[sec]
        if c == 0:
            print(f"{sec}\t0\t(n/a)\t(n/a)")
            out[sec] = {"count": 0.0, "avg_precision": float("nan"), "avg_recall": float("nan")}
        else:
            ap = sums_p[sec] / c
            ar = sums_r[sec] / c
            print(f"{sec}\t{c}\t{ap:.4f}\t{ar:.4f}")
            out[sec] = {"count": float(c), "avg_precision": ap, "avg_recall": ar}

    total = sum(counts[s] for s in TARGET_SECTIONS)
    if total > 0:
        ap_all = sum(sums_p[s] for s in TARGET_SECTIONS) / total
        ar_all = sum(sums_r[s] for s in TARGET_SECTIONS) / total
        print(f"All(4+7+8)\t{total}\t{ap_all:.4f}\t{ar_all:.4f}")

    print(
        f"# skipped (not 4/7/8): {skipped_section}, missing section in prompt: {missing_section}",
        file=__import__("sys").stderr,
    )

    if args.json_out:
        import json as _json

        with open(args.json_out, "w", encoding="utf-8") as f:
            _json.dump(out, f, indent=2, ensure_ascii=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
