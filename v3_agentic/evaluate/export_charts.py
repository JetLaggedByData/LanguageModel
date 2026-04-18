"""
mlflow_runs/export_charts.py
Export all four Model Arena Plotly charts as static JSON files.

These JSON files are committed to the repo so the deployed HF Spaces app
can render rich benchmark charts without any model inference or MLflow DB.

Writes:
  mlflow_runs/charts/perplexity.json
  mlflow_runs/charts/bleu.json
  mlflow_runs/charts/critic_distribution.json
  mlflow_runs/charts/loss_curve.json

Run AFTER benchmark.py:
  python mlflow_runs/export_charts.py
"""

import sys
import json
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

REPORT_PATH = ROOT / "mlflow_runs" / "benchmark_report.json"
LOSS_PATH   = ROOT / "mlflow_runs" / "v2_loss_curve.json"
CHARTS_DIR  = ROOT / "mlflow_runs" / "charts"

# Shared theme
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#090e1a",
    plot_bgcolor="#0f1829",
    font=dict(family="Share Tech Mono, monospace", color="#c8d8e8", size=13),
    title_font=dict(family="Share Tech Mono, monospace", color="#00d4ff", size=16),
    xaxis=dict(gridcolor="#1e3a5f", linecolor="#1e3a5f", zerolinecolor="#1e3a5f"),
    yaxis=dict(gridcolor="#1e3a5f", linecolor="#1e3a5f", zerolinecolor="#1e3a5f"),
    legend=dict(bgcolor="#0f1829", bordercolor="#1e3a5f", borderwidth=1),
    margin=dict(l=60, r=30, t=60, b=50),
)

COLORS = {
    "v1": "#5a7a9a",
    "v2": "#f5a623",
    "v3": "#00d4ff",
}


# ── Chart builders ────────────────────────────────────────────────────────

def build_perplexity_chart(report: dict) -> go.Figure:
    v1 = report.get("v1", {})
    v2 = report.get("v2", {})

    models = ["V1 LSTM\n(char-level)", "V2 QLoRA\n(word-level)"]
    values = [
        v1.get("v1_char_perplexity", 0),
        v2.get("v2_word_perplexity", 0),
    ]
    colors = [COLORS["v1"], COLORS["v2"]]

    fig = go.Figure(go.Bar(
        x=models, y=values,
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
        textfont=dict(color="#c8d8e8"),
        marker_color=colors,
        marker_line_color="#1e3a5f",
        marker_line_width=1,
        width=0.5,
    ))

    # Annotate improvement delta
    deltas = report.get("deltas", {})
    pct = deltas.get("perplexity_pct_change")
    if pct is not None:
        fig.add_annotation(
            x=1, y=values[1] * 1.15,
            text=f"{pct:+.1f}% vs V1",
            showarrow=False,
            font=dict(color=COLORS["v2"], size=12),
        )

    fig.update_layout(
        title="Perplexity — Lower is Better",
        yaxis_title="Perplexity Score",
        showlegend=False,
        **PLOTLY_LAYOUT,
    )
    return fig


def build_bleu_chart(report: dict) -> go.Figure:
    v1 = report.get("v1", {})
    v2 = report.get("v2", {})

    metrics  = ["BLEU-2", "BLEU-4"]
    v1_vals  = [v1.get("v1_bleu2", 0), 0]
    v2_vals  = [v2.get("v2_bleu2", 0), v2.get("v2_bleu4", 0)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="V1 LSTM",  x=metrics, y=v1_vals,
        marker_color=COLORS["v1"], marker_line_color="#1e3a5f",
        text=[f"{v:.3f}" for v in v1_vals], textposition="outside",
        textfont=dict(color="#c8d8e8"),
    ))
    fig.add_trace(go.Bar(
        name="V2 QLoRA", x=metrics, y=v2_vals,
        marker_color=COLORS["v2"], marker_line_color="#1e3a5f",
        text=[f"{v:.3f}" for v in v2_vals], textposition="outside",
        textfont=dict(color="#c8d8e8"),
    ))

    fig.update_layout(
        title="BLEU Scores — Higher is Better",
        yaxis_title="BLEU Score",
        barmode="group",
        **PLOTLY_LAYOUT,
    )
    return fig


def build_critic_distribution_chart(report: dict) -> go.Figure:
    v3 = report.get("v3", {})
    # Reconstruct score distribution from per_story data
    all_scores = []
    for story in report.get("per_story", []):
        score = story.get("avg_score")
        if score is not None:
            all_scores.append(float(score))

    # Fall back to summary stats if individual scores unavailable
    mean   = v3.get("v3_score_mean", 0)
    median = v3.get("v3_score_median", 0)

    fig = go.Figure()

    if all_scores:
        fig.add_trace(go.Histogram(
            x=all_scores, nbinsx=15,
            marker_color=COLORS["v3"],
            marker_line_color="#1e3a5f",
            marker_line_width=1,
            name="Story scores",
        ))
        if mean:
            fig.add_vline(
                x=mean, line_dash="dash", line_color=COLORS["v2"], line_width=2,
                annotation_text=f"mean {mean:.2f}",
                annotation_font=dict(color=COLORS["v2"], size=12),
                annotation_position="top right",
            )
    else:
        # Show score bucket bar chart from summary stats
        buckets = ["Excellent (≥0.8)", "Good (0.6–0.8)", "Poor (<0.6)"]
        counts  = [
            v3.get("v3_chapters_excellent", 0),
            v3.get("v3_chapters_good", 0),
            v3.get("v3_chapters_poor", 0),
        ]
        bucket_colors = ["#39ff14", COLORS["v2"], "#ff4b4b"]
        fig.add_trace(go.Bar(
            x=buckets, y=counts,
            marker_color=bucket_colors,
            text=counts, textposition="outside",
            textfont=dict(color="#c8d8e8"),
        ))

    fig.update_layout(
        title="V3 Critique Score Distribution",
        xaxis_title="Score" if all_scores else "Score Bucket",
        yaxis_title="Count",
        showlegend=False,
        **PLOTLY_LAYOUT,
    )
    return fig


def build_loss_curve_chart(loss_data: list[dict]) -> go.Figure:
    fig = go.Figure()

    if loss_data:
        steps  = [d.get("step", i) for i, d in enumerate(loss_data)]
        losses = [d.get("loss", 0) for d in loss_data]

        # Smoothed line (every 5th point) + raw scatter
        fig.add_trace(go.Scatter(
            x=steps, y=losses,
            mode="lines",
            line=dict(color=COLORS["v2"], width=2),
            name="Train Loss",
            opacity=0.8,
        ))
        # Min loss annotation
        min_loss = min(losses)
        min_step = steps[losses.index(min_loss)]
        fig.add_annotation(
            x=min_step, y=min_loss,
            text=f"best {min_loss:.3f}",
            showarrow=True, arrowcolor=COLORS["v3"],
            font=dict(color=COLORS["v3"], size=11),
        )
    else:
        fig.add_annotation(
            text="Run v2_finetuned/finetune.py<br>then benchmark.py to populate",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False,
            font=dict(color="#5a7a9a", size=14),
        )

    fig.update_layout(
        title="V2 QLoRA Training Loss",
        xaxis_title="Step",
        yaxis_title="Loss",
        **PLOTLY_LAYOUT,
    )
    return fig


# ── Export ────────────────────────────────────────────────────────────────

def export_all_charts() -> None:
    """Build all charts and write as Plotly JSON to mlflow_runs/charts/."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    report    = json.loads(REPORT_PATH.read_text()) if REPORT_PATH.exists() else {}
    loss_data = json.loads(LOSS_PATH.read_text())   if LOSS_PATH.exists()   else []

    charts = {
        "perplexity":           build_perplexity_chart(report),
        "bleu":                 build_bleu_chart(report),
        "critic_distribution":  build_critic_distribution_chart(report),
        "loss_curve":           build_loss_curve_chart(loss_data),
    }

    for name, fig in charts.items():
        out = CHARTS_DIR / f"{name}.json"
        out.write_text(pio.to_json(fig))
        print(f"  Exported → {out}")

    print(f"\n✅ All charts exported to {CHARTS_DIR}")
    print("   Commit this directory to make them available on HF Spaces.")


if __name__ == "__main__":
    export_all_charts()
