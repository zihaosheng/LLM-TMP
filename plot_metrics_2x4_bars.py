#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

fontsize = 13

def main() -> None:
    # 4 个指标，对应每一列的子图标题（同一行 4 个子图）
    metrics = ["BLEU-4", "ROUGE-1", "ROUGE-2", "ROUGE-L"]

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
    # 形状: [num_models, num_metrics, 2]  = [3, 4, 2]
    rng = np.random.default_rng(8)
    row1_before = np.array([
        [18.1, 18.9, 3.11, 9.52], # Llama-3.1-8B
        [13.6, 15.1, 2.22, 6.34], # DeepSeek-8B
        [18.7, 18.8, 6.00, 10.5], # Qwen2.5-7B
    ]).reshape(len(models_row1), len(metrics), 1)
    row1_after = np.array([
        [45.9, 61.3, 50.7, 54.5], # Llama-3.1-8B
        [48.0, 62.9, 52.7, 56.7], # DeepSeek-R1-Distill-Llama-8B
        [49.6, 63.8, 53.9, 58.1], # Qwen2.5-7B
    ]).reshape(len(models_row1), len(metrics), 1)
    scores_row1 = np.concatenate([row1_before, row1_after], axis=2)

    row2_before = np.array([
        [15.7, 19.2, 3.83, 8.13], # Qwen3-14B
        [14.0, 15.5, 2.40, 6.67], # DeepSeek-R1-Distill-Qwen-14B
        [15.3, 19.1, 4.27, 8.57], # Qwen2.5-14B
    ]).reshape(len(models_row2), len(metrics), 1)
    row2_after = np.array([
        [49.9, 62.9, 53.1, 57.4], # Qwen3-14B
        [49.6, 62.7, 52.4, 56.7], # DeepSeek-R1-Distill-Qwen-14B
        [50.8, 63.5, 53.3, 57.6], # Qwen2.5-14B
    ]).reshape(len(models_row2), len(metrics), 1)
    scores_row2 = np.concatenate([row2_before, row2_after], axis=2)

    # 配色：每个模型一种颜色（legend 用）。用更柔和的 Set2。
    palette = plt.get_cmap("Set2").colors
    
    all_models = models_row1 + models_row2
    # import seaborn as sns
    # palette = sns.color_palette("husl", len(all_models))
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
        ncols=4,
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
            ax.set_xticklabels(xticklabels, rotation=0, fontsize=fontsize-1)
            ax.set_yticklabels([0, 20, 40, 60], fontsize=fontsize-1)
            ax.grid(axis="y", linestyle="--", alpha=0.35)

    plot_row(axes[0], models_row1, scores_row1, colors_row1, model_family)
    plot_row(axes[1], models_row2, scores_row2, colors_row2, model_family)

    # 统一纵轴含义
    # axes[0, 0].set_ylabel("Score")
    # axes[1, 0].set_ylabel("Score")

    # 子图序号 (a) ... (h)，放在每个子图下方
    letters = "abcdefgh"
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

    out_path = "/home/sky-lab/SHENG_code/LLM-TMP/paper/figs/metrics_2x4_barplot."
    # fig.suptitle("Fake Metrics Before/After Fine-tuning", y=0.98)
    fig.savefig(out_path + "png", dpi=300, bbox_inches="tight")
    fig.savefig(out_path + "pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out_path + "svg", dpi=300, bbox_inches="tight")
    fig.savefig(out_path + "jpg", dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()

