#!/usr/bin/env python3
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    metrics_all = ["Section 4", "Section 5", "Section 6", "Section 7", "Section 8", "Section 9"]

    # 3 行 4 列：每一行 4 个 title（允许重复/重排）
    titles_row1 = ["Section 4", "Section 5", "Section 6", "Section 7"]
    titles_row2 = ["Section 8", "Section 9", "Section 4", "Section 5"]
    titles_row3 = ["Section 7", "Section 7", "Section 8", "Section 9"]

    title_to_idx = {t: i for i, t in enumerate(metrics_all)}
    metric_idx_row1 = [title_to_idx[t] for t in titles_row1]
    metric_idx_row2 = [title_to_idx[t] for t in titles_row2]
    metric_idx_row3 = [title_to_idx[t] for t in titles_row3]

    # 每一行放 3 个模型（示例）
    models_row1 = ["Llama-3.1-8B", "Qwen2.5-7B", "DeepSeek-R1-Distill-Llama-8B"]
    models_row2 = ["Qwen2.5-14B", "Qwen3-14B", "DeepSeek-R1-Distill-Llama-14B"]
    models_row3 = ["Llama-3.1-8B", "Qwen3-14B", "DeepSeek-R1-Distill-Llama-14B"]

    model_family = {
        "Llama-3.1-8B": "Llama",
        "Qwen2.5-7B": "Qwen2.5",
        "Qwen2.5-14B": "Qwen2.5",
        "Qwen3-14B": "Qwen3",
        "DeepSeek-R1-Distill-Llama-8B": "DeepSeek",
        "DeepSeek-R1-Distill-Llama-14B": "DeepSeek",
    }

    rng = np.random.default_rng(8)

    def make_scores(n_models: int, n_metrics: int) -> np.ndarray:
        base = rng.uniform(15, 45, size=(n_models, n_metrics, 1))
        gain = rng.uniform(2.0, 8.0, size=(n_models, n_metrics, 1))
        return np.concatenate([base, base + gain], axis=2)  # [n_models, n_metrics, 2]

    scores_row1 = make_scores(len(models_row1), len(metrics_all))
    scores_row2 = make_scores(len(models_row2), len(metrics_all))
    scores_row3 = make_scores(len(models_row3), len(metrics_all))

    palette = plt.get_cmap("Set2").colors
    all_models = list(dict.fromkeys(models_row1 + models_row2 + models_row3))
    colors_all = {m: palette[i % len(palette)] for i, m in enumerate(all_models)}

    def lighten(rgb: tuple[float, float, float], amount: float) -> tuple[float, float, float]:
        r, g, b = rgb
        return (r + (1.0 - r) * amount, g + (1.0 - g) * amount, b + (1.0 - b) * amount)

    phase_labels = ["Before FT", "After FT"]
    phase_color_amounts = [0.55, 0.0]

    fig, axes = plt.subplots(
        nrows=3,
        ncols=4,
        figsize=(16, 9.2),
        sharey=True,
        constrained_layout=False,
        gridspec_kw={"hspace": 0.85, "wspace": 0.25},
    )
    fig.subplots_adjust(bottom=0.22, top=0.90)

    def plot_row(
        row_axes: np.ndarray,
        row_models: list[str],
        row_scores: np.ndarray,
        metric_indices: list[int],
        metric_titles: list[str],
        row_model_family: dict[str, str] | None = None,
    ) -> None:
        n_models = len(row_models)
        group_x = np.arange(n_models)
        bar_w = 0.28
        inner_gap = 0.08
        offsets = np.array([-(bar_w / 2 + inner_gap / 2), (bar_w / 2 + inner_gap / 2)])

        for col, metric_j in enumerate(metric_indices):
            ax = row_axes[col]
            for mi, model in enumerate(row_models):
                for pi in range(2):
                    x = group_x[mi] + offsets[pi]
                    y = float(row_scores[mi, metric_j, pi])
                    base_c = colors_all[model]
                    bar_c = lighten(base_c, phase_color_amounts[pi])
                    ax.bar(x, y, width=bar_w, color=bar_c, edgecolor="gray", linewidth=0.6)

            ax.set_title(metric_titles[col])
            ax.set_xticks(group_x)
            if row_model_family is not None:
                xticklabels = [row_model_family[m] for m in row_models]
            else:
                xticklabels = row_models
            ax.set_xticklabels(xticklabels, rotation=0)
            ax.grid(axis="y", linestyle="--", alpha=0.35)

    plot_row(axes[0], models_row1, scores_row1, metric_idx_row1, titles_row1, model_family)
    plot_row(axes[1], models_row2, scores_row2, metric_idx_row2, titles_row2, model_family)
    plot_row(axes[2], models_row3, scores_row3, metric_idx_row3, titles_row3, model_family)

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

    model_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors_all[m], edgecolor="gray") for m in all_models
    ]
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

    out_path = "/home/sky-lab/SHENG_code/LLM-TMP/paper/figs/metrics_3x4_barplot."
    fig.savefig(out_path + "png", dpi=300, bbox_inches="tight")
    fig.savefig(out_path + "pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out_path + "svg", dpi=300, bbox_inches="tight")
    fig.savefig(out_path + "jpg", dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()

