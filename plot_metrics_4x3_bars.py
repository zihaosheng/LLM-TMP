#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    metrics = ["Section 4", "Section 5", "Section 6", "Section 7", "Section 8", "Section 9"]
    metrics_row_13 = metrics[:3]  # Section 4,5,6
    metrics_row_24 = metrics[3:]  # Section 7,8,9
    metric_idx_row_13 = [0, 1, 2]
    metric_idx_row_24 = [3, 4, 5]

    # 6 个模型：第一行 3 个，第二行 3 个
    models_row1 = ["Llama-3.1-8B", "Qwen2.5-7B", "DeepSeek-R1-Distill-Llama-8B"]
    models_row2 = ["Qwen2.5-14B", "Qwen3-14B", "DeepSeek-R1-Distill-Llama-14B"]

    model_family = {
        "Llama-3.1-8B": "Llama",
        "Qwen2.5-7B": "Qwen2.5",
        "Qwen2.5-14B": "Qwen2.5",
        "Qwen3-14B": "Qwen3",
        "DeepSeek-R1-Distill-Llama-8B": "DeepSeek",
        "DeepSeek-R1-Distill-Llama-14B": "DeepSeek",
    }

    # fake 数据：每个模型都有 4 个指标，每个指标有 before/after
    # 形状: [num_models, num_metrics, 2]
    rng = np.random.default_rng(8)
    base_row1 = rng.uniform(15, 45, size=(len(models_row1), len(metrics), 1))
    gain_row1 = rng.uniform(2.0, 8.0, size=(len(models_row1), len(metrics), 1))
    scores_row1 = np.concatenate([base_row1, base_row1 + gain_row1], axis=2)

    base_row2 = rng.uniform(15, 45, size=(len(models_row2), len(metrics), 1))
    gain_row2 = rng.uniform(2.0, 8.0, size=(len(models_row2), len(metrics), 1))
    scores_row2 = np.concatenate([base_row2, base_row2 + gain_row2], axis=2)

    # 配色：每个模型一种颜色（legend 用）。用更柔和的 Set2。
    palette = plt.get_cmap("Set2").colors
    all_models = models_row1 + models_row2
    colors_all = {m: palette[i] for i, m in enumerate(all_models)}
    colors_row1 = {m: colors_all[m] for m in models_row1}
    colors_row2 = {m: colors_all[m] for m in models_row2}

    def lighten(rgb: tuple[float, float, float], amount: float) -> tuple[float, float, float]:
        """amount in [0,1], 越大越接近白色。"""
        r, g, b = rgb
        return (r + (1.0 - r) * amount, g + (1.0 - g) * amount, b + (1.0 - b) * amount)

    # before/after 用同色系浅/深表示
    phase_labels = ["Before FT", "After FT"]
    phase_color_amounts = [0.55, 0.0]  # before 更浅；after 用原色

    fig, axes = plt.subplots(
        nrows=4,
        ncols=3,
        figsize=(14.2, 10.6),
        sharey=True,
        constrained_layout=False,
        gridspec_kw={"hspace": 0.75, "wspace": 0.25},
    )
    # 给底部 legend 和子图标号预留空间
    fig.subplots_adjust(bottom=0.22, top=0.90)

    def plot_row(
        row_axes: np.ndarray,
        row_models: list[str],
        row_scores: np.ndarray,
        row_colors: dict[str, tuple[float, float, float]],
        metric_indices: list[int],
        metric_titles: list[str],
        row_model_family: dict[str, str] | None = None,
    ) -> None:
        # 每个子图：3 组（模型）× 2 根（before/after）= 6 根柱
        n_models = len(row_models)
        group_x = np.arange(n_models)  # 0,1,2 对应 3 个模型组
        bar_w = 0.28
        inner_gap = 0.08  # 同一模型 before/after 两根柱之间的间隙（相对 x 轴单位）
        offsets = np.array(
            [-(bar_w / 2 + inner_gap / 2), (bar_w / 2 + inner_gap / 2)]
        )

        for col, metric_j in enumerate(metric_indices):
            ax = row_axes[col]

            for mi, model in enumerate(row_models):
                for pi in range(2):
                    x = group_x[mi] + offsets[pi]
                    y = float(row_scores[mi, metric_j, pi])
                    base_c = row_colors[model]
                    bar_c = lighten(base_c, phase_color_amounts[pi])
                    ax.bar(
                        x,
                        y,
                        width=bar_w,
                        color=bar_c,
                        edgecolor="gray",
                        linewidth=0.6,
                    )

            ax.set_title(metric_titles[col])
            ax.set_xticks(group_x)
            if row_model_family is not None:
                xticklabels = [row_model_family[model] for model in row_models]
            else:
                xticklabels = row_models
            ax.set_xticklabels(xticklabels, rotation=0)
            ax.grid(axis="y", linestyle="--", alpha=0.35)

    # 4x3：第 1/3 行标题 Section 4-6；第 2/4 行标题 Section 7-9
    plot_row(axes[0], models_row1, scores_row1, colors_row1, metric_idx_row_13, metrics_row_13, model_family)
    plot_row(axes[1], models_row1, scores_row1, colors_row1, metric_idx_row_24, metrics_row_24, model_family)
    plot_row(axes[2], models_row2, scores_row2, colors_row2, metric_idx_row_13, metrics_row_13, model_family)
    plot_row(axes[3], models_row2, scores_row2, colors_row2, metric_idx_row_24, metrics_row_24, model_family)

    # 统一纵轴含义
    axes[0, 0].set_ylabel("Score")
    axes[1, 0].set_ylabel("Score")
    axes[2, 0].set_ylabel("Score")
    axes[3, 0].set_ylabel("Score")

    # 子图序号 (a) ... (h)，放在每个子图下方
    letters = "abcdefghijkl"
    for idx, ax in enumerate(axes.flat):
        ax.text(
            0.5,
            -0.28,
            f"({letters[idx]})",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
        )

    # Legend 1：模型（颜色）
    model_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors_all[m], edgecolor="gray") for m in all_models
    ]

    # Legend 2：before/after（浅/深）
    phase_base = (0.35, 0.35, 0.35)
    phase_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=lighten(phase_base, phase_color_amounts[i]),
            edgecolor="gray",
        )
        for i in range(2)
    ]

    # 合并成一行 legend：先放 6 个模型，再放 before/after
    legend_handles = model_handles + phase_handles
    legend_labels = all_models + phase_labels
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=len(legend_labels),
        frameon=False,
        bbox_to_anchor=(0.5, 0.07),
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=1.1,
    )

    out_path = "/home/sky-lab/SHENG_code/LLM-TMP/paper/figs/metrics_4x3_barplot."
    # fig.suptitle("Fake Metrics Before/After Fine-tuning", y=0.98)
    fig.savefig(out_path + "png", dpi=300, bbox_inches="tight")
    fig.savefig(out_path + "pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out_path + "svg", dpi=300, bbox_inches="tight")
    fig.savefig(out_path + "jpg", dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()

