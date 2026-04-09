#!/usr/bin/env python3
"""分析 alan_eval_data.json：条数、instruction 中的 section 分布、input 中的 TMP ID / TMP Type 分布。"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# TMP Form Version 2.0 TMP ID: 7041 - Current Version
RE_TMP_ID = re.compile(r"TMP ID:\s*(\d+)", re.IGNORECASE)
# [ Filling Section 4 - Work Zone Strategies ]
RE_INSTRUCTION_SECTION_NO = re.compile(r"Section\s+(\d+)", re.IGNORECASE)
# Markdown 表格行: | TMP Type:           | 2                                                         |
RE_TMP_TYPE = re.compile(
    r"^\|\s*TMP Type:\s*\|\s*([^|]+?)\s*\|",
    re.MULTILINE | re.IGNORECASE,
)
# 部分样本表格损坏，首行仅有数字：| 2                                                |
RE_TMP_TYPE_FALLBACK = re.compile(
    r"### \*\*Section 1A[^\n]*\n\n\|\s*(\d+)\s*\|",
    re.IGNORECASE,
)


def extract_tmp_id(text: str) -> str | None:
    m = RE_TMP_ID.search(text)
    return m.group(1) if m else None


def extract_tmp_type(text: str) -> str | None:
    m = RE_TMP_TYPE.search(text)
    if m:
        return m.group(1).strip() or None
    m = RE_TMP_TYPE_FALLBACK.search(text)
    if m:
        return m.group(1).strip() or None
    return None


def extract_section_no(instruction: str) -> int | None:
    m = RE_INSTRUCTION_SECTION_NO.search(instruction or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def print_counter(title: str, ctr: Counter[str], missing_key: str = "(未解析)") -> None:
    print(f"\n=== {title} ===")
    total = sum(ctr.values())
    for k, n in ctr.most_common():
        pct = 100.0 * n / total if total else 0
        print(f"  {k!r}: {n} ({pct:.1f}%)")
    if missing_key in ctr:
        pass  # already printed
    print(f"  [合计] {total}")
    ids = ctr.keys()
    # ids = [int(id) for id in ids if id.isdigit()]
    ids = sorted(ids)
    print(f"  一共[{len(ids)}]个 ID")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--json",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "alan_eval_data.json",
        help="alan_eval_data.json 路径",
    )
    ap.add_argument(
        "--split",
        action="store_true",
        help="按 instruction 的 Section No. 分割输出为 alan_eval_sec4.json ... alan_eval_sec9.json",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="split 输出目录（默认 ./data）",
    )
    args = ap.parse_args()

    with args.json.open(encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise SystemExit("顶层应为 JSON 数组")

    n = len(data)
    print(f"数据条数: {n}")

    instruction_ctr: Counter[str] = Counter()
    tmp_id_ctr: Counter[str] = Counter()
    tmp_type_ctr: Counter[str] = Counter()
    section_buckets: dict[int, list[dict]] = {k: [] for k in range(4, 10)}
    missing_id = 0
    missing_type = 0

    for i, row in enumerate(data):
        if not isinstance(row, dict):
            print(f"警告: 第 {i} 条不是对象，已跳过")
            continue
        inst = row.get("instruction", "")
        instruction_ctr[str(inst)] += 1
        sec_no = extract_section_no(str(inst))
        if sec_no in section_buckets:
            section_buckets[sec_no].append(row)

        inp = row.get("input") or ""
        if not isinstance(inp, str):
            inp = str(inp)

        tid = extract_tmp_id(inp)
        if tid is None:
            missing_id += 1
            tmp_id_ctr["(未解析 TMP ID)"] += 1
        else:
            tmp_id_ctr[tid] += 1

        ttyp = extract_tmp_type(inp)
        if ttyp is None:
            missing_type += 1
            tmp_type_ctr["(未解析 TMP Type)"] += 1
        else:
            tmp_type_ctr[ttyp] += 1

    print_counter("instruction 分布（完整字符串）", instruction_ctr)
    print_counter("input 中 TMP ID 分布", tmp_id_ctr)
    print_counter("input 中 TMP Type 分布（表格第一列值）", tmp_type_ctr)

    if args.split:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        total_out = 0
        for sec_no in range(4, 10):
            out_path = args.out_dir / f"alan_eval_sec{sec_no}.json"
            rows = section_buckets.get(sec_no, [])
            total_out += len(rows)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)
            print(f"写出: {out_path}  (rows={len(rows)})")
        print(f"split 合计写出条数: {total_out}")

    if missing_id or missing_type:
        print("\n--- 解析说明 ---")
        if missing_id:
            print(f"未能从 input 正则匹配到 TMP ID 的条数: {missing_id}")
        if missing_type:
            print(
                f"未能从 input 匹配到标准表格行 | TMP Type: | ... | 的条数: {missing_type}"
            )


if __name__ == "__main__":
    main()
