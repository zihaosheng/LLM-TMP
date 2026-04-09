#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

fontsize = 10

def main() -> None:
    # 4 个指标，对应每一列的子图标题（同一行 4 个子图）
    metrics = ["Section 4", "Section 5", "Section 6", "Section 7", "Section 8", "Section 9"]

    # 6 个模型：第一行 3 个，第二行 3 个
    models_row1 = ["Llama-3.1-8B", "DeepSeek-R1-Distill-Llama-8B", "Qwen2.5-7B"]
    models_row2 = ["Qwen3-14B", "DeepSeek-R1-Distill-Qwen-14B", "Qwen2.5-14B"]

    model_family = {
        "Llama-3.1-8B": "Llama",
        "Qwen2.5-7B": "Qwen2.5",
        "Qwen2.5-14B": "Qwen2.5",
        "Qwen3-14B": "Qwen3",
        "DeepSeek-R1-Distill-Llama-8B": "DeepSeek",
        "DeepSeek-R1-Distill-Qwen-14B": "DeepSeek",
    }

    # fake 数据：每个模型都有 4 个指标，每个指标有 before/after
    # 形状: [num_models, num_metrics, 2]
    rng = np.random.default_rng(8)
    row1_before = np.array([
        [7.53, 14.4, 12.5, 6.50, 4.88, 11.3], # Llama-3.1-8B
        [5.12, 10.2, 8.07, 4.04, 2.62, 8.03], # DeepSeek-8B
        [6.98, 13.4, 15.4, 8.34, 4.49, 14.3], # Qwen2.5-7B
    ]).reshape(len(models_row1), len(metrics), 1)
    row1_after = np.array([
        [42.2, 63.2, 56.9, 46.6, 63.9, 53.8], # Llama-3.1-8B
        [45.3, 64.3, 57.2, 51.2, 68.3, 53.8], # DeepSeek-8B
        [42.7, 63.7, 58.3, 56.5, 70.8, 56.4], # Qwen2.5-7B
    ]).reshape(len(models_row1), len(metrics), 1)
    scores_row1 = np.concatenate([row1_before, row1_after], axis=2)

    row2_before = np.array([
        [5.97, 12.0, 10.6, 5.76, 3.49, 11.0], # Qwen3-14B
        [5.49, 10.5, 8.21, 4.46, 2.77, 8.55], # DeepSeek-R1-Distill-Qwen-14B
        [6.05, 13.8, 11.5, 5.62, 3.35, 11.2], # Qwen2.5-14B
    ]).reshape(len(models_row2), len(metrics), 1)
    row2_after = np.array([
        [45.7, 61.9, 57.9, 54.6, 69.9, 54.4], # Qwen3-14B
        [41.4, 63.1, 55.7, 53.8, 70.9, 55.0], # DeepSeek-R1-Distill-Qwen-14B
        [46.8, 63.3, 57.7, 53.8, 68.6, 55.2], # Qwen2.5-14B
    ]).reshape(len(models_row2), len(metrics), 1)
    scores_row2 = np.concatenate([row2_before, row2_after], axis=2)

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
        nrows=2,
        ncols=6,
        figsize=(16, 6),
        sharey=True,
        constrained_layout=False,
        gridspec_kw={"hspace": 0.55, "wspace": 0.25},
    )
    # 给底部 legend 和子图标号预留空间
    fig.subplots_adjust(bottom=0.22, top=0.90)

    def plot_row(
        row_axes: np.ndarray,
        row_models: list[str],
        row_scores: np.ndarray,
        row_colors: dict[str, tuple[float, float, float]],
        row_model_family: dict[str, str]=None,
    ) -> None:
        # 每个子图：3 组（模型）× 2 根（before/after）= 6 根柱
        n_models = len(row_models)
        group_x = np.arange(n_models)  # 0,1,2 对应 3 个模型组
        bar_w = 0.28
        inner_gap = 0.08  # 同一模型 before/after 两根柱之间的间隙（相对 x 轴单位）
        offsets = np.array(
            [-(bar_w / 2 + inner_gap / 2), (bar_w / 2 + inner_gap / 2)]
        )

        for j, metric in enumerate(metrics):
            ax = row_axes[j]

            for mi, model in enumerate(row_models):
                for pi in range(2):
                    x = group_x[mi] + offsets[pi]
                    y = float(row_scores[mi, j, pi])
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

            ax.set_title(metric, fontsize=fontsize)
            ax.set_xticks(group_x)
            if row_model_family is not None:
                xticklabels = [row_model_family[model] for model in row_models]
            else:
                xticklabels = row_models
            ax.set_xticklabels(xticklabels, rotation=0, fontsize=fontsize-1.5)
            ax.grid(axis="y", linestyle="--", alpha=0.35)

    plot_row(axes[0], models_row1, scores_row1, colors_row1, model_family)
    plot_row(axes[1], models_row2, scores_row2, colors_row2, model_family)

    # 统一纵轴含义
    # axes[0, 0].set_ylabel("Score")
    # axes[1, 0].set_ylabel("Score")

    # 子图序号 (a) ... (h)，放在每个子图下方
    letters = "abcdefghijkl"
    for idx, ax in enumerate(axes.flat):
        ax.text(
            0.5,
            -0.2,
            f"({letters[idx]})",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=fontsize,
        )

    # Legend 1：模型（颜色）
    model_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors_all[m], edgecolor="gray") for m in all_models
    ]

    # Legend 2：before/after（浅/深）
    phase_base = (0.35, 0.35, 0.35)
    phase_handles = [
        # plt.Rectangle(
        #     (0, 0),
        #     1,
        #     1,
        #     facecolor=lighten(phase_base, phase_color_amounts[i]),
        #     edgecolor="gray",
        # )
        # for i in range(2)
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
        fontsize=fontsize-1,
    )

    out_path = "/home/sky-lab/SHENG_code/LLM-TMP/paper/figs/metrics_2x6_barplot."
    # fig.suptitle("Fake Metrics Before/After Fine-tuning", y=0.98)
    fig.savefig(out_path + "png", dpi=300, bbox_inches="tight")
    fig.savefig(out_path + "pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out_path + "svg", dpi=300, bbox_inches="tight")
    fig.savefig(out_path + "jpg", dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()

