#!/usr/bin/env python3
"""
分析 LLaMA-Factory inference_efficiency 输出：
- trainer_log.jsonl: 读取 elapsed_time，做相邻差得到每 step 耗时
- generated_predictions.jsonl: 从 prompt 中解析 section（形如 "[ Filling Section 4 - xxx ]"）
- 丢掉第一个 section，将剩余 section 与 step 耗时按顺序对齐，统计每个 section 总耗时/均值/次数
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


_FILLING_SECTION_RE = re.compile(
    r"\[\s*Filling\s+Section\s+(?P<num>\d+)\s*-\s*(?P<title>.*?)\s*\]",
    re.IGNORECASE,
)


def _parse_elapsed_time_to_seconds(value: object) -> float:
    """
    支持：
    - "H:MM:SS"
    - "MM:SS"
    - "D day, HH:MM:SS" / "D days, HH:MM:SS"
    - 数值（视作秒）
    """
    if value is None:
        raise ValueError("elapsed_time is None")
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise ValueError(f"elapsed_time has unsupported type: {type(value)}")

    s = value.strip()
    # "D day(s), HH:MM:SS"
    m = re.match(r"^(?P<days>\d+)\s+day[s]?,\s+(?P<hms>\d+:\d{2}:\d{2})$", s)
    if m:
        days = int(m.group("days"))
        hms = m.group("hms")
        h, mm, ss = [int(x) for x in hms.split(":")]
        return float(days * 86400 + h * 3600 + mm * 60 + ss)

    parts = s.split(":")
    if len(parts) == 3:
        h, mm, ss = [int(x) for x in parts]
        return float(h * 3600 + mm * 60 + ss)
    if len(parts) == 2:
        mm, ss = [int(x) for x in parts]
        return float(mm * 60 + ss)

    raise ValueError(f"Unrecognized elapsed_time format: {value!r}")


def load_step_times_seconds(trainer_log_path: Path) -> list[float]:
    elapsed_seconds: list[float] = []
    with trainer_log_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {trainer_log_path}:{i}: {e}") from e
            if "elapsed_time" not in obj:
                raise ValueError(f"Missing 'elapsed_time' at {trainer_log_path}:{i}")
            elapsed_seconds.append(_parse_elapsed_time_to_seconds(obj["elapsed_time"]))

    if len(elapsed_seconds) < 2:
        return []

    step_times = []
    for prev, cur in zip(elapsed_seconds[:-1], elapsed_seconds[1:]):
        step_times.append(cur - prev)
    return step_times


def extract_sections_from_predictions(predictions_path: Path) -> list[str]:
    sections: list[str] = []
    with predictions_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {predictions_path}:{i}: {e}") from e
            prompt = obj.get("prompt", "")
            if not isinstance(prompt, str):
                prompt = str(prompt)
            m = _FILLING_SECTION_RE.search(prompt)
            if not m:
                # 兜底：尝试从正文里抓 "Section X - Title"
                m2 = re.search(
                    r"Section\s+(?P<num>\d+)\s*-\s*(?P<title>[^\]\n\r]+)",
                    prompt,
                    flags=re.IGNORECASE,
                )
                if m2:
                    num = m2.group("num")
                    title = m2.group("title").strip()
                    sections.append(f"Section {num} - {title}")
                else:
                    sections.append("UNKNOWN")
                continue
            num = m.group("num")
            title = m.group("title").strip()
            sections.append(f"Section {num} - {title}")
    return sections


@dataclass
class SectionStats:
    count: int = 0
    total_seconds: float = 0.0
    # Welford online variance accumulator
    mean_acc: float = 0.0
    m2: float = 0.0

    @property
    def mean_seconds(self) -> float:
        return self.total_seconds / self.count if self.count else 0.0

    @property
    def std_seconds(self) -> float:
        # 样本标准差（n-1）。若只有 1 个样本，定义为 0。
        if self.count <= 1:
            return 0.0
        return math.sqrt(self.m2 / (self.count - 1))

    def add(self, x: float) -> None:
        self.count += 1
        self.total_seconds += x
        # Welford update
        delta = x - self.mean_acc
        self.mean_acc += delta / self.count
        delta2 = x - self.mean_acc
        self.m2 += delta * delta2


def format_seconds(s: float) -> str:
    # 保留 3 位小数，避免整秒看起来太“硬”
    return f"{s:.3f}s"


def _section_number(section_label: str) -> int | None:
    m = re.match(r"^\s*Section\s+(?P<num>\d+)\b", section_label, flags=re.IGNORECASE)
    if not m:
        return None
    return int(m.group("num"))


def find_consecutive_section_cycles(
    sections: list[str], step_times: list[float], pattern: list[int]
) -> list[float]:
    """
    在对齐后的序列中寻找连续 pattern（例如 [4,5,6,7,8,9]），返回每个匹配片段的总耗时（秒）。
    匹配后默认跳过整个片段，避免重叠计数。
    """
    if len(sections) != len(step_times):
        raise ValueError("sections and step_times must be aligned with same length")
    k = len(pattern)
    if k == 0:
        return []
    totals: list[float] = []
    i = 0
    while i + k <= len(sections):
        nums = [_section_number(s) for s in sections[i : i + k]]
        if all(n is not None for n in nums) and [int(n) for n in nums] == pattern:
            totals.append(float(sum(step_times[i : i + k])))
            i += k
        else:
            i += 1
    return totals


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-dir",
        default="/home/sky-lab/codes/LLaMA-Factory/tmp/inference_efficiency/DeepSeek-R1-Distill-Qwen-14B/",
        help="jsonl 路径",
    )
    ap.add_argument(
        "--drop-first-sections",
        type=int,
        default=1,
        help="丢掉最前面的 section 数（默认 1）",
    )
    ap.add_argument(
        "--strict-length",
        action="store_true",
        help="若 section 数与 step_time 数不一致则直接报错；默认取 min 对齐并给出提示",
    )
    args = ap.parse_args()

    trainer_log_path = Path(args.base_dir) / "trainer_log.jsonl"
    predictions_path = Path(args.base_dir) / "generated_predictions.jsonl"

    step_times = load_step_times_seconds(trainer_log_path)
    sections_all = extract_sections_from_predictions(predictions_path)

    if args.drop_first_sections < 0:
        raise ValueError("--drop-first-sections must be >= 0")
    sections = sections_all[args.drop_first_sections :]

    if args.strict_length and len(sections) != len(step_times):
        raise SystemExit(
            f"Length mismatch: sections(after drop)={len(sections)} vs step_times={len(step_times)}"
        )

    n = min(len(sections), len(step_times))
    if n == 0:
        print("No aligned data. Check inputs.", file=sys.stderr)
        print(
            f"step_times={len(step_times)}, sections_all={len(sections_all)}, sections(after drop)={len(sections)}",
            file=sys.stderr,
        )
        return 2

    if len(sections) != len(step_times):
        print(
            f"[warn] Length mismatch, aligning by order using min: sections(after drop)={len(sections)}, step_times={len(step_times)}, aligned={n}",
            file=sys.stderr,
        )

    stats: dict[str, SectionStats] = defaultdict(SectionStats)
    for sec, dt in zip(sections[:n], step_times[:n]):
        stats[sec].add(float(dt))

    # 输出：按总耗时降序
    # items = sorted(stats.items(), key=lambda kv: kv[1].total_seconds, reverse=True)
    items = sorted(stats.items(), key=lambda kv: kv[0])

    total_aligned = sum(v.total_seconds for _, v in items)
    print(f"aligned_pairs: {n}")
    print(f"total_time(aligned): {format_seconds(total_aligned)}")
    print(f"unique_sections: {len(items)}")
    print("")
    for sec, st in items:
        print(
            f"{sec[:9]}\tcount={st.count}\ttotal={format_seconds(st.total_seconds)}\tmean={format_seconds(st.mean_seconds)}\tstd={format_seconds(st.std_seconds)}"
        )

    # 连续 Section 4-9 cycle 统计
    aligned_sections = sections[:n]
    aligned_step_times = step_times[:n]
    cycle_totals = find_consecutive_section_cycles(
        aligned_sections, aligned_step_times, pattern=[4, 5, 6, 7, 8, 9]
    )
    cycle_mean, cycle_std = mean_std(cycle_totals)
    print("")
    print(
        "total"
        f"\tcount={len(cycle_totals)}"
        f"\tmean={format_seconds(cycle_mean)}"
        f"\tstd={format_seconds(cycle_std)}"
        f"\ttotal={format_seconds(sum(cycle_totals))}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

