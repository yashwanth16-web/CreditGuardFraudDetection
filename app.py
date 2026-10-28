
from pathlib import Path
from io import BytesIO
import json

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "reports"
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

MODEL_PATH = MODEL_DIR / "xgboost.joblib"
RESULTS_PATH = REPORT_DIR / "final_test_results.csv"
MONITORING_HISTORY_PATH = REPORT_DIR / "monitoring_history.json"
COMPARISON_PATH = REPORT_DIR / "threshold_comparison.csv"
TEST_PATH = DATA_DIR / "test.csv"

TARGET = "Class"


st.set_page_config(
    page_title="CreditGuard | Fraud Analytics",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_data
def load_csv(path_string):
    path = Path(path_string)
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_resource
def load_model(path_string):
    path = Path(path_string)
    if not path.exists():
        return None
    return joblib.load(path)


def metric_card(label, value, help_text=None):
    st.metric(label=label, value=value, help=help_text)



def build_quality_pdf(source_name, df, checks, profile, missing_by_column, target_counts=None):
    """Create a self-contained PDF audit report without adding a new dependency."""
    buffer = BytesIO()
    rows, columns = df.shape
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    with PdfPages(buffer) as pdf:
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.patch.set_facecolor("white")
        fig.text(0.07, 0.91, "CreditGuard | Data Quality Audit", fontsize=22, weight="bold", color="#10233f")
        fig.text(0.07, 0.86, f"Dataset: {source_name}", fontsize=11, color="#42526b")
        fig.text(0.07, 0.82, f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", fontsize=9, color="#64748b")
        summary = [
            ["Rows", f"{rows:,}"], ["Columns", f"{columns:,}"],
            ["Missing cells", f"{missing_cells:,}"], ["Exact duplicate rows", f"{duplicate_rows:,}"],
        ]
        ax = fig.add_axes([0.07, 0.57, 0.86, 0.19])
        ax.axis("off")
        table = ax.table(cellText=summary, colLabels=["Metric", "Result"], loc="center", cellLoc="left", colLoc="left", colWidths=[0.55, 0.35])
        table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.65)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("#d8e0ea")
            if r == 0:
                cell.set_facecolor("#eaf1f8"); cell.set_text_props(weight="bold", color="#10233f")
        fig.text(0.07, 0.50, "Validation checks", fontsize=14, weight="bold", color="#10233f")
        check_rows = [[str(row.get("Check", "")), str(row.get("Status", "")), str(row.get("Details", ""))] for row in checks]
        ax2 = fig.add_axes([0.07, 0.12, 0.86, 0.34]); ax2.axis("off")
        t2 = ax2.table(cellText=check_rows, colLabels=["Check", "Status", "Details"], loc="upper left", cellLoc="left", colLoc="left", colWidths=[0.28, 0.13, 0.59], bbox=[0, 0, 1, 1])
        t2.auto_set_font_size(False); t2.set_fontsize(8); t2.scale(1, 1.3)
        for (r, c), cell in t2.get_celld().items():
            cell.set_edgecolor("#d8e0ea")
            if r == 0:
                cell.set_facecolor("#eaf1f8"); cell.set_text_props(weight="bold", color="#10233f")
        fig.text(0.07, 0.045, "This report is diagnostic only. Review duplicates and anomalies before changing source data.", fontsize=8, color="#64748b")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Profile report, split into readable chunks for large schemas.
        page_size = 26
        profile_export = profile.copy()
        if not profile_export.empty:
            for start in range(0, len(profile_export), page_size):
                chunk = profile_export.iloc[start:start + page_size]
                fig, ax = plt.subplots(figsize=(11.7, 8.3))
                fig.patch.set_facecolor("white"); ax.axis("off")
                ax.set_title("Column profile" + (f" ({start + 1}–{min(start + page_size, len(profile_export))})"), loc="left", fontsize=16, fontweight="bold", color="#10233f", pad=18)
                t = ax.table(cellText=chunk.astype(str).values.tolist(), colLabels=list(chunk.columns), loc="center", cellLoc="left", colLoc="left", bbox=[0, 0.02, 1, 0.9])
                t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.25)
                for (r, c), cell in t.get_celld().items():
                    cell.set_edgecolor("#d8e0ea")
                    if r == 0:
                        cell.set_facecolor("#eaf1f8"); cell.set_text_props(weight="bold", color="#10233f")
                pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        if not missing_by_column.empty:
            fig, ax = plt.subplots(figsize=(11.7, 8.3))
            missing_sorted = missing_by_column.sort_values(ascending=True).tail(25)
            ax.barh(missing_sorted.index.astype(str), missing_sorted.values, color="#e85d75")
            ax.set_title("Missing values by column (up to 25 columns)", loc="left", fontsize=16, fontweight="bold")
            ax.set_xlabel("Missing cell count"); ax.grid(axis="x", alpha=0.25)
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        if target_counts is not None and not target_counts.empty:
            fig, ax = plt.subplots(figsize=(11.7, 8.3))
            ax.bar(target_counts["Class"].astype(str), target_counts["Count"], color=["#2f80c9" if str(v) in ("0", "0.0") else "#e85d75" for v in target_counts["Class"]])
            ax.set_title("Target distribution", loc="left", fontsize=16, fontweight="bold")
            ax.set_xlabel("Class label"); ax.set_ylabel("Rows"); ax.grid(axis="y", alpha=0.25)
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    buffer.seek(0)
    return buffer.getvalue()


def build_monitoring_pdf(source_name, drift_df, summary):
    """Create a downloadable PDF summary of feature drift monitoring."""
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.patch.set_facecolor("white")
        fig.text(0.07, 0.92, "CreditGuard | Model Monitoring Report", fontsize=21, weight="bold", color="#10233f")
        fig.text(0.07, 0.875, f"Current batch: {source_name}", fontsize=11, color="#42526b")
        fig.text(0.07, 0.84, f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", fontsize=9, color="#64748b")
        summary_rows = [[str(k), str(v)] for k, v in summary.items()]
        ax = fig.add_axes([0.07, 0.57, 0.86, 0.22]); ax.axis("off")
        table = ax.table(cellText=summary_rows, colLabels=["Metric", "Result"], loc="center", cellLoc="left", colLoc="left", colWidths=[0.55, 0.35])
        table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.5)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("#d8e0ea")
            if r == 0:
                cell.set_facecolor("#eaf1f8"); cell.set_text_props(weight="bold", color="#10233f")
        fig.text(0.07, 0.49, "Feature drift results", fontsize=14, weight="bold", color="#10233f")
        page_size = 24
        if drift_df.empty:
            fig.text(0.07, 0.42, "No feature drift results were available.", fontsize=10, color="#42526b")
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        else:
            first = drift_df.iloc[:page_size]
            ax2 = fig.add_axes([0.07, 0.10, 0.86, 0.34]); ax2.axis("off")
            t2 = ax2.table(cellText=first.round(4).astype(object).where(pd.notna(first), "—").values.tolist(), colLabels=list(first.columns), loc="center", cellLoc="left", colLoc="left", bbox=[0, 0, 1, 1])
            t2.auto_set_font_size(False); t2.set_fontsize(7.5); t2.scale(1, 1.2)
            for (r, c), cell in t2.get_celld().items():
                cell.set_edgecolor("#d8e0ea")
                if r == 0:
                    cell.set_facecolor("#eaf1f8"); cell.set_text_props(weight="bold", color="#10233f")
            fig.text(0.07, 0.045, "PSI is a screening indicator, not proof of model degradation or fraud.", fontsize=8, color="#64748b")
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
            for start in range(page_size, len(drift_df), page_size):
                chunk = drift_df.iloc[start:start + page_size]
                fig, ax = plt.subplots(figsize=(11.7, 8.3)); fig.patch.set_facecolor("white"); ax.axis("off")
                ax.set_title(f"Feature drift results ({start + 1}–{min(start + page_size, len(drift_df))})", loc="left", fontsize=15, fontweight="bold", color="#10233f", pad=18)
                t = ax.table(cellText=chunk.round(4).astype(object).where(pd.notna(chunk), "—").values.tolist(), colLabels=list(chunk.columns), loc="center", cellLoc="left", colLoc="left", bbox=[0, 0.02, 1, 0.9])
                t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.2)
                for (r, c), cell in t.get_celld().items():
                    cell.set_edgecolor("#d8e0ea")
                    if r == 0:
                        cell.set_facecolor("#eaf1f8"); cell.set_text_props(weight="bold", color="#10233f")
                pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def build_table_pdf(title, subtitle, table_df, summary=None, max_rows=80):
    """Create a readable PDF export for a table and its summary metrics."""
    buffer = BytesIO()
    summary = summary or {}
    with PdfPages(buffer) as pdf:
        page_size = max_rows
        chunks = [table_df.iloc[i:i + page_size] for i in range(0, max(len(table_df), 1), page_size)]
        if table_df.empty:
            chunks = [table_df]
        for page_number, chunk in enumerate(chunks, start=1):
            fig, ax = plt.subplots(figsize=(11.7, 8.3))
            fig.patch.set_facecolor("white"); ax.axis("off")
            fig.text(0.07, 0.94, title, fontsize=20, weight="bold", color="#10233f")
            fig.text(0.07, 0.90, subtitle, fontsize=9, color="#64748b")
            top = 0.84
            if page_number == 1 and summary:
                summary_rows = [[str(k), str(v)] for k, v in summary.items()]
                ax_summary = fig.add_axes([0.07, 0.68, 0.86, 0.13]); ax_summary.axis("off")
                stbl = ax_summary.table(cellText=summary_rows, colLabels=["Metric", "Result"], loc="center", cellLoc="left", colLoc="left", colWidths=[0.55, 0.35])
                stbl.auto_set_font_size(False); stbl.set_fontsize(8); stbl.scale(1, 1.15)
                for (r, c), cell in stbl.get_celld().items():
                    cell.set_edgecolor("#d8e0ea")
                    if r == 0:
                        cell.set_facecolor("#eaf1f8"); cell.set_text_props(weight="bold", color="#10233f")
                top = 0.62
            fig.text(0.07, top, f"Details (page {page_number} of {len(chunks)})", fontsize=12, weight="bold", color="#10233f")
            if chunk.empty:
                fig.text(0.07, top - 0.07, "No rows to display.", fontsize=10, color="#42526b")
            else:
                printable = chunk.copy()
                for col in printable.columns:
                    printable[col] = printable[col].map(lambda value: "—" if pd.isna(value) else str(value))
                ax_table = fig.add_axes([0.07, 0.08, 0.86, max(0.15, top - 0.12)]); ax_table.axis("off")
                tbl = ax_table.table(cellText=printable.values.tolist(), colLabels=list(printable.columns), loc="center", cellLoc="left", colLoc="left", bbox=[0, 0, 1, 1])
                tbl.auto_set_font_size(False); tbl.set_fontsize(6.8 if len(printable.columns) > 6 else 8); tbl.scale(1, 1.1)
                for (r, c), cell in tbl.get_celld().items():
                    cell.set_edgecolor("#d8e0ea")
                    if r == 0:
                        cell.set_facecolor("#eaf1f8"); cell.set_text_props(weight="bold", color="#10233f")
            fig.text(0.07, 0.035, "CreditGuard educational prototype — model outputs require independent review.", fontsize=7.5, color="#64748b")
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def calculate_psi(reference, current, bins=10):
    """Population Stability Index using reference quantile bins."""
    ref = pd.to_numeric(pd.Series(reference), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    cur = pd.to_numeric(pd.Series(current), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(ref) < 2 or len(cur) < 2:
        return np.nan
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0 if np.isclose(np.nanmean(ref), np.nanmean(cur)) else np.nan
    edges[0] = -np.inf; edges[-1] = np.inf
    ref_counts = np.histogram(ref, bins=edges)[0].astype(float)
    cur_counts = np.histogram(cur, bins=edges)[0].astype(float)
    ref_pct = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    cur_pct = np.clip(cur_counts / cur_counts.sum(), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

def load_monitoring_history():
    """Load persisted monitoring snapshots from a local JSON file."""
    if not MONITORING_HISTORY_PATH.exists():
        return pd.DataFrame()
    try:
        payload = json.loads(MONITORING_HISTORY_PATH.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            return pd.DataFrame()
        return pd.DataFrame(payload)
    except (OSError, json.JSONDecodeError, ValueError):
        return pd.DataFrame()


def save_monitoring_snapshot(snapshot):
    """Append one monitoring snapshot to durable local history."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    history = load_monitoring_history()
    new_row = pd.DataFrame([snapshot])
    if not history.empty:
        history = pd.concat([history, new_row], ignore_index=True)
    else:
        history = new_row
    MONITORING_HISTORY_PATH.write_text(
        history.replace({np.nan: None}).to_json(orient="records", indent=2, date_format="iso"),
        encoding="utf-8",
    )
    return history


def main():
    # CreditGuard theme and sidebar layout
    st.markdown(
        """
        <style>
        :root {
            --cg-bg: #080b12;
            --cg-panel: #101722;
            --cg-panel-2: #141e2c;
            --cg-line: #263449;
            --cg-muted: #93a3b8;
            --cg-blue: #55b7ff;
            --cg-cyan: #54e1d0;
            --cg-red: #ff5e70;
        }
        .stApp {
            background: #0e1117;
            color: #f4f6fa;
        }
        [data-testid="stMainBlockContainer"] {
            max-width: 1500px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        h1 {
            letter-spacing: -0.045em;
            font-weight: 780 !important;
            line-height: 1.12 !important;
        }
        h2, h3 {
            letter-spacing: -0.025em;
        }
        .stApp [data-testid="stCaptionContainer"] {
            color: var(--cg-muted);
        }
        .cg-page-kicker, .cg-hero-title, .cg-hero-copy, .cg-hero { display: none; }
        [data-testid="stSidebar"] {
            background: #11151d;
            border-right: 1px solid #252c37;
        }
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.2rem;
        }
        .cg-brand {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 1.25rem;
            font-weight: 750;
            color: #f5f7fb;
            padding: 0.35rem 0 1rem 0;
        }
        .cg-brand-icon {
            color: #55aaff;
            font-size: 1.35rem;
            line-height: 1;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.45rem;
            min-width: 1.45rem;
        }
        .cg-subtitle {
            color: #9ca7b8;
            font-size: 0.82rem;
            margin-top: -0.3rem;
            margin-bottom: 1rem;
        }
        /* Sidebar navigation buttons (no radio circles) */
        [data-testid="stSidebar"] div[data-testid="stButton"] {
            margin: 0.18rem 0;
        }
        [data-testid="stSidebar"] div[data-testid="stButton"] button {
            min-height: 2.45rem;
            width: 100%;
            justify-content: flex-start;
            text-align: left;
            border-radius: 9px;
            padding: 0.48rem 0.68rem;
            font-size: 0.90rem;
            font-weight: 500;
            box-shadow: none;
            transition: background 120ms ease, border-color 120ms ease,
                        color 120ms ease;
        }
        [data-testid="stSidebar"] div[data-testid="stButton"] button[kind="secondary"] {
            background: transparent;
            border: 1px solid transparent;
            color: #c5ccd8;
        }
        [data-testid="stSidebar"] div[data-testid="stButton"] button[kind="secondary"]:hover {
            background: #202632;
            border-color: #2b3340;
            color: #f5f7fb;
        }
        [data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {
            background: #2b202a;
            border: 1px solid #382833;
            color: #ff5964;
        }
        [data-testid="stSidebar"] div[data-testid="stButton"] button p {
            font-size: 0.90rem;
            margin: 0;
        }
        [data-testid="stSidebar"] div[data-testid="stButton"] button:focus {
            box-shadow: 0 0 0 1px #384253;
        }
        .cg-footer {
            color: #7f8a9b;
            font-size: 0.78rem;
            line-height: 1.5;
        }
        div[data-testid="stMetric"] {
            background: #151922;
            border: 1px solid #303744;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            box-shadow: none;
            min-height: 100px;
        }
        div[data-testid="stMetric"] label {
            color: #a9b8cc;
            font-size: .82rem;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-weight: 760;
            letter-spacing: -.035em;
        }
        div[data-testid="stAlert"] {
            border-radius: 11px;
            border-width: 1px;
        }
        div[data-testid="stDataFrame"], div[data-testid="stTable"] {
            border: 1px solid #263244;
            border-radius: 12px;
            overflow: hidden;
        }
        div[data-testid="stPlotlyChart"] {
            background: #11151d;
            border: 1px solid #252c37;
            border-radius: 8px;
            padding: .35rem;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 10px;
            font-weight: 650;
            transition: transform .15s ease, border-color .15s ease, filter .15s ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            transform: translateY(-1px);
            filter: brightness(1.08);
        }
        div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
            border-radius: 9px;
            background-color: #111722;
        }
        hr {
            border-color: #263244;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown(
            '<div class="cg-brand"><span class="cg-brand-icon">🛡️</span>'
            '<span>CreditGuard</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="cg-subtitle">Credit Card Fraud Detection<br>'
            'Machine Learning Analytics</div>',
            unsafe_allow_html=True,
        )
        st.divider()

        # Add future sections here, then add the corresponding page renderer
        # below the existing section blocks. Each section gets its own button.
        navigation_items = [
            ("Executive Overview", "▦", "nav_overview"),
            ("Model Comparison", "▥", "nav_comparison"),
            ("Transaction Scoring", "◎", "nav_scoring"),
            ("Fraud Investigation", "⌕", "nav_investigation"),
            ("Data Quality", "▤", "nav_data_quality"),
            ("Model Monitoring", "◷", "nav_monitoring"),
            ("Risk Alerts", "⚑", "nav_risk_alerts"),
            ("Monitoring History", "▤", "nav_monitoring_history"),
            ("Project Information", "ⓘ", "nav_information"),
        ]
        if "creditguard_navigation" not in st.session_state:
            st.session_state["creditguard_navigation"] = "Executive Overview"

        def select_section(section):
            st.session_state["creditguard_navigation"] = section

        for section, icon, button_key in navigation_items:
            is_active = st.session_state["creditguard_navigation"] == section
            st.button(
                f"{icon}   {section}",
                key=button_key,
                type="primary" if is_active else "secondary",
                use_container_width=True,
                on_click=select_section,
                args=(section,),
            )

        selected_section = st.session_state["creditguard_navigation"]

        st.divider()
        st.markdown(
            '<div class="cg-footer">Machine Learning Analytics<br>'
            'Educational prototype</div>',
            unsafe_allow_html=True,
        )

    results = load_csv(str(RESULTS_PATH))
    comparison = load_csv(str(COMPARISON_PATH))
    test = load_csv(str(TEST_PATH))

    # ---------------------------------------------------------
    # TAB 1: Executive overview
    # ---------------------------------------------------------
    if selected_section == "Executive Overview":
        st.title("Executive Overview")
        st.caption("Fraud detection performance at a glance.")
        st.subheader("Final Test Evaluation")

        if results is None:
            st.warning(
                "Final results are missing. Run "
                "python src/evaluate_final_model.py first."
            )
        else:
            result = results.iloc[0]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                metric_card(
                    "Fraud Precision",
                    f"{result['precision']:.2%}",
                )

            with col2:
                metric_card(
                    "Fraud Recall",
                    f"{result['recall']:.2%}",
                )

            with col3:
                metric_card(
                    "Fraud F1-score",
                    f"{result['f1_score']:.3f}",
                )

            with col4:
                metric_card(
                    "Average Precision",
                    f"{result['average_precision']:.4f}",
                )

            st.markdown("### Confusion Matrix")

            matrix = [
                [
                    int(result["true_negatives"]),
                    int(result["false_positives"]),
                ],
                [
                    int(result["false_negatives"]),
                    int(result["true_positives"]),
                ],
            ]

            figure = go.Figure(
                data=go.Heatmap(
                    z=matrix,
                    x=["Predicted Legitimate", "Predicted Fraud"],
                    y=["Actual Legitimate", "Actual Fraud"],
                    text=matrix,
                    texttemplate="%{text}",
                    colorscale="Blues",
                    showscale=False,
                )
            )

            figure.update_layout(
                xaxis_title="Prediction",
                yaxis_title="Actual class",
                height=400,
            )

            st.plotly_chart(figure, use_container_width=True)

            st.info(
                f"Model: {result['model']} | "
                f"Threshold: {result['threshold']:.6f}"
            )

            st.caption(
                "These are results on the reserved test partition. "
                "They are not a guarantee of real-world performance."
            )

        if test is not None and TARGET in test.columns:
            st.markdown("### Test Dataset Distribution")

            counts = (
                test[TARGET]
                .value_counts()
                .reindex([0, 1], fill_value=0)
                .rename(index={0: "Legitimate", 1: "Fraud"})
                .rename_axis("Transaction Type")
                .reset_index(name="Count")
            )

            fig = px.bar(
                counts,
                x="Transaction Type",
                y="Count",
                color="Transaction Type",
                title="Legitimate vs Fraudulent Transactions",
            )

            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Precision–Recall Curve")

            curve_path = (
                ROOT
                / "reports"
                / "figures"
                / "precision_recall_curves.png"
            )

            if curve_path.exists():
                st.image(
                    str(curve_path),
                    caption=(
                        "Precision–recall curves for all six models "
                        "on the validation set."
                    ),
                    use_container_width=True,
                )
            else:
                st.info(
                    "Run python src/evaluate_curves.py to generate "
                    "the precision–recall curve."
                )

            ap_path = ROOT / "reports" / "precision_recall_metrics.csv"

            if ap_path.exists():
                st.markdown("### Average Precision Ranking")
                ap_df = pd.read_csv(ap_path).sort_values(
                    "average_precision",
                    ascending=False,
                )

                st.dataframe(
                    ap_df,
                    use_container_width=True,
                    hide_index=True,
                )


    # ---------------------------------------------------------
    # TAB 2: Model comparison
    # ---------------------------------------------------------
    elif selected_section == "Model Comparison":
        st.title("Model Comparison")
        st.caption("Compare validation performance across all six models.")
        st.subheader("Validation Model Leaderboard")

        if comparison is None:
            st.warning(
                "Comparison report missing. Run "
                "python src/tune_thresholds.py first."
            )
        else:
            metric_columns = [
                "model",
                "average_precision",
                "roc_auc",
                "precision",
                "recall",
                "f1_score",
                "false_positives",
                "false_negatives",
            ]

            available_columns = [
                column for column in metric_columns
                if column in comparison.columns
            ]

            display_df = comparison[available_columns].copy()

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )

            if {"model", "precision", "recall", "f1_score"}.issubset(
                comparison.columns
            ):
                chart_df = comparison.melt(
                    id_vars="model",
                    value_vars=["precision", "recall", "f1_score"],
                    var_name="Metric",
                    value_name="Score",
                )

                fig = px.bar(
                    chart_df,
                    x="model",
                    y="Score",
                    color="Metric",
                    barmode="group",
                    title="Validation Metrics by Model",
                    range_y=[0, 1],
                )

                st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "These are validation results at thresholds selected "
                "on validation data. They are not directly comparable "
                "to the final test results without considering the "
                "threshold-selection process."
            )

    # ---------------------------------------------------------
    # TAB 3: Transaction scoring
    # ---------------------------------------------------------
    elif selected_section == "Transaction Scoring":
        st.title("Transaction Scoring")
        st.caption("Score a CSV of transactions with the selected XGBoost model.")
        st.subheader("Score Transactions")

        st.write(
            "Upload a CSV containing the transaction features used "
            "during training. The file must contain the same feature "
            "columns, such as Time, V1–V28, and Amount. The Class "
            "column is not required."
        )

        model = load_model(str(MODEL_PATH))

        if model is None:
            st.error(
                "Trained XGBoost model not found. "
                "Run python src/train_xgboost.py first."
            )
        else:
            threshold = 0.5

            if results is not None:
                threshold = float(results.iloc[0]["threshold"])

            threshold = st.slider(
                "Fraud decision threshold",
                min_value=0.01,
                max_value=0.99,
                value=max(0.01, min(0.99, threshold)),
                step=0.01,
                help=(
                    "Changing this threshold changes the number of "
                    "transactions flagged as fraud."
                ),
            )

            uploaded_file = st.file_uploader(
                "Upload transaction CSV",
                type=["csv"],
                key="transaction_csv",
            )

            if uploaded_file is not None:
                try:
                    uploaded_df = pd.read_csv(uploaded_file)

                    if TARGET in uploaded_df.columns:
                        st.warning(
                            "The uploaded file contains Class labels. "
                            "They will be excluded from prediction."
                        )
                        X_input = uploaded_df.drop(columns=[TARGET])
                    else:
                        X_input = uploaded_df.copy()

                    expected_columns = list(
                        model.feature_names_in_
                    )

                    missing = [
                        col for col in expected_columns
                        if col not in X_input.columns
                    ]

                    extra = [
                        col for col in X_input.columns
                        if col not in expected_columns
                    ]

                    if missing or extra:
                        st.error(
                            f"Missing columns: {missing}\n\n"
                            f"Unexpected columns: {extra}"
                        )
                    elif X_input.empty:
                        st.warning("The uploaded CSV contains no rows.")
                    elif not all(
                        pd.api.types.is_numeric_dtype(dtype)
                        for dtype in X_input.dtypes
                    ):
                        st.error(
                            "All transaction feature columns must be numeric."
                        )
                    elif X_input.isna().any().any():
                        st.error(
                            "The uploaded data contains missing values."
                        )
                    else:
                        X_input = X_input[expected_columns]

                        numeric_values = X_input.to_numpy(dtype=float)

                        if not np.isfinite(numeric_values).all():
                            st.error(
                                "Transaction features must contain only "
                                "finite numbers (no NaN or infinity)."
                            )
                        else:
                            probabilities = model.predict_proba(
                                X_input
                            )[:, 1]

                            predictions = (
                                probabilities >= threshold
                            ).astype(int)

                            output = uploaded_df.copy()
                            output["Fraud Probability"] = probabilities
                            output["Predicted Class"] = predictions
                            output["Prediction"] = [
                                "Fraud" if value == 1
                                else "Legitimate"
                                for value in predictions
                            ]

                            # Keep the latest scored batch available to the
                            # Fraud Investigation page while this session is open.
                            st.session_state["creditguard_scored_output"] = output.copy()
                            st.session_state["creditguard_scored_threshold"] = threshold

                            st.success(
                                f"Scored {len(output):,} transactions."
                            )

                            col1, col2 = st.columns(2)

                            with col1:
                                metric_card(
                                    "Flagged as Fraud",
                                    f"{int(predictions.sum()):,}",
                                )

                            with col2:
                                metric_card(
                                    "Flag Rate",
                                    f"{predictions.mean():.2%}",
                                )

                            st.dataframe(
                                output.head(100),
                                use_container_width=True,
                            )

                            scoring_pdf = build_table_pdf(
                                "CreditGuard | Transaction Scoring Report",
                                f"Generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} — preview of up to 80 scored transactions",
                                output.head(80),
                                {"Transactions scored": f"{len(output):,}", "Flagged as fraud": f"{int(predictions.sum()):,}", "Flag rate": f"{predictions.mean():.2%}", "Decision threshold": f"{threshold:.2f}"},
                            )
                            st.download_button(
                                "Download scoring report (PDF)",
                                data=scoring_pdf,
                                file_name="creditguard_scoring_report.pdf",
                                mime="application/pdf",
                                key="download_scoring_pdf",
                            )

                except Exception as exc:
                    st.error(
                        f"Unable to process the uploaded CSV: {exc}"
                    )

    # ---------------------------------------------------------
    # TAB 4: Fraud investigation
    # ---------------------------------------------------------
    elif selected_section == "Fraud Investigation":
        st.title("Fraud Investigation")
        st.caption(
            "Review scored transactions, prioritize high-risk cases, "
            "and inspect the model's output for each transaction."
        )

        investigation_df = st.session_state.get("creditguard_scored_output")

        if investigation_df is None or investigation_df.empty:
            st.info(
                "No scored transactions are available in this session yet. "
                "Open Transaction Scoring, upload a CSV, and score it first."
            )
            st.button(
                "Go to Transaction Scoring",
                key="go_to_scoring_from_investigation",
                on_click=select_section,
                args=("Transaction Scoring",),
            )
        elif not {"Fraud Probability", "Predicted Class", "Prediction"}.issubset(
            investigation_df.columns
        ):
            st.error(
                "The saved scoring results do not contain the expected prediction "
                "columns. Please score the CSV again from Transaction Scoring."
            )
        else:
            review_df = investigation_df.copy()
            review_df.insert(0, "Transaction Row", range(len(review_df)))

            fraud_count = int((review_df["Predicted Class"] == 1).sum())
            high_risk_count = int((review_df["Fraud Probability"] >= 0.80).sum())
            average_risk = float(review_df["Fraud Probability"].mean())

            metric_cols = st.columns(3)
            with metric_cols[0]:
                metric_card("Flagged as Fraud", f"{fraud_count:,}")
            with metric_cols[1]:
                metric_card("High-risk (≥ 80%)", f"{high_risk_count:,}")
            with metric_cols[2]:
                metric_card("Mean Fraud Probability", f"{average_risk:.2%}")

            st.markdown("### Investigation Filters")
            filter_cols = st.columns([1, 1, 2])
            with filter_cols[0]:
                prediction_filter = st.selectbox(
                    "Prediction",
                    ["All", "Fraud", "Legitimate"],
                    key="investigation_prediction_filter",
                )
            with filter_cols[1]:
                risk_filter = st.selectbox(
                    "Risk level",
                    ["All risks", "High (≥ 80%)", "Medium (50–79.99%)", "Low (< 50%)"],
                    key="investigation_risk_filter",
                )
            with filter_cols[2]:
                minimum_probability = st.slider(
                    "Minimum fraud probability",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05,
                    key="investigation_min_probability",
                )

            filtered = review_df.copy()
            if prediction_filter != "All":
                filtered = filtered[filtered["Prediction"] == prediction_filter]
            if risk_filter == "High (≥ 80%)":
                filtered = filtered[filtered["Fraud Probability"] >= 0.80]
            elif risk_filter == "Medium (50–79.99%)":
                filtered = filtered[
                    (filtered["Fraud Probability"] >= 0.50)
                    & (filtered["Fraud Probability"] < 0.80)
                ]
            elif risk_filter == "Low (< 50%)":
                filtered = filtered[filtered["Fraud Probability"] < 0.50]
            filtered = filtered[
                filtered["Fraud Probability"] >= minimum_probability
            ]
            filtered = filtered.sort_values(
                "Fraud Probability", ascending=False
            )

            st.caption(
                f"Showing {len(filtered):,} of {len(review_df):,} scored transactions, "
                "sorted from highest to lowest model-estimated fraud probability."
            )

            preferred_columns = [
                "Transaction Row", "Prediction", "Predicted Class", "Fraud Probability"
            ]
            if "Time" in filtered.columns:
                preferred_columns.append("Time")
            if "Amount" in filtered.columns:
                preferred_columns.append("Amount")
            st.dataframe(
                filtered[preferred_columns],
                use_container_width=True,
                hide_index=True,
            )

            investigation_pdf = build_table_pdf(
                "CreditGuard | Fraud Investigation Report",
                f"Generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} — filtered investigation cases",
                filtered.drop(columns=["Transaction Row"], errors="ignore").head(80),
                {"Cases matching filters": f"{len(filtered):,}", "Risk threshold": f"{minimum_probability:.0%}" if "minimum_probability" in locals() else "Configured in investigation filters"},
            )
            st.download_button(
                "Download investigation report (PDF)",
                data=investigation_pdf,
                file_name="creditguard_investigation_report.pdf",
                mime="application/pdf",
                key="download_investigation_pdf",
            )

            st.markdown("### Inspect One Transaction")
            if filtered.empty:
                st.warning("No transactions match the selected filters.")
            else:
                row_options = filtered["Transaction Row"].astype(int).tolist()
                selected_row = st.selectbox(
                    "Transaction row",
                    row_options,
                    format_func=lambda row_id: (
                        f"Row {row_id} — "
                        f"{float(review_df.loc[review_df['Transaction Row'] == row_id, 'Fraud Probability'].iloc[0]):.2%} "
                        "estimated fraud probability"
                    ),
                    key="investigation_selected_row",
                )
                selected_record = review_df.loc[
                    review_df["Transaction Row"] == selected_row
                ].iloc[0]
                detail_cols = st.columns(3)
                with detail_cols[0]:
                    metric_card("Model prediction", str(selected_record["Prediction"]))
                with detail_cols[1]:
                    metric_card(
                        "Fraud probability",
                        f"{float(selected_record['Fraud Probability']):.2%}",
                    )
                with detail_cols[2]:
                    saved_threshold = st.session_state.get(
                        "creditguard_scored_threshold", 0.5
                    )
                    metric_card("Scoring threshold", f"{float(saved_threshold):.2f}")

                st.markdown("**Transaction feature values**")
                feature_record = selected_record.drop(
                    labels=[
                        "Transaction Row", "Fraud Probability",
                        "Predicted Class", "Prediction"
                    ],
                    errors="ignore",
                )
                feature_table = feature_record.rename("Value").to_frame()
                feature_table.index.name = "Feature"
                st.dataframe(feature_table, use_container_width=True)

                st.markdown("### Why did the model assign this score?")
                st.caption(
                    "SHAP estimates each feature's contribution to the model output. "
                    "Positive contributions push the score toward fraud; negative "
                    "contributions push it away from fraud. These are model explanations, "
                    "not proof that a transaction is fraudulent."
                )

                if st.button(
                    "Generate SHAP explanation",
                    key="generate_shap_explanation",
                    type="primary",
                ):
                    try:
                        import shap

                        explanation_model = load_model(str(MODEL_PATH))
                        if explanation_model is None:
                            st.error(
                                "The XGBoost model could not be loaded. "
                                "Check that models/xgboost.joblib exists."
                            )
                        else:
                            # Support both a bare tree model and sklearn Pipeline.
                            # TreeExplainer must receive the underlying tree estimator,
                            # not the Pipeline wrapper itself.
                            pipeline_steps = list(getattr(explanation_model, "steps", []))
                            if pipeline_steps:
                                tree_model = pipeline_steps[-1][1]
                                transformers = [step for _, step in pipeline_steps[:-1]]
                            else:
                                tree_model = explanation_model
                                transformers = []

                            expected_features = list(
                                getattr(explanation_model, "feature_names_in_", [])
                            )
                            if not expected_features:
                                expected_features = list(
                                    getattr(tree_model, "feature_names_in_", [])
                                )

                            source_features = selected_record.drop(
                                labels=[
                                    "Transaction Row", "Fraud Probability",
                                    "Predicted Class", "Prediction", TARGET
                                ],
                                errors="ignore",
                            )
                            if not expected_features:
                                st.error(
                                    "The saved model does not expose its input feature names. "
                                    "Retrain/save the model with named pandas columns to enable safe explanations."
                                )
                            else:
                                missing_features = [
                                    name for name in expected_features
                                    if name not in source_features.index
                                ]
                                if missing_features:
                                    st.error(
                                        "Cannot explain this row because model features "
                                        f"are missing: {missing_features}"
                                    )
                                else:
                                    explain_row = pd.DataFrame(
                                        [{name: float(source_features[name]) for name in expected_features}],
                                        columns=expected_features,
                                    )
                                    transformed_row = explain_row.copy()
                                    for transformer in transformers:
                                        if hasattr(transformer, "transform"):
                                            transformed_row = transformer.transform(transformed_row)

                                    # Keep transformed output as a DataFrame when possible;
                                    # this also lets SHAP labels match preprocessing output.
                                    if isinstance(transformed_row, pd.DataFrame):
                                        transformed_values = transformed_row.to_numpy()
                                        transformed_names = list(transformed_row.columns)
                                    else:
                                        transformed_values = np.asarray(transformed_row)
                                        transformed_names = []
                                        for transformer in transformers:
                                            if hasattr(transformer, "get_feature_names_out"):
                                                try:
                                                    transformed_names = list(
                                                        transformer.get_feature_names_out(expected_features)
                                                    )
                                                except Exception:
                                                    try:
                                                        transformed_names = list(
                                                            transformer.get_feature_names_out()
                                                        )
                                                    except Exception:
                                                        transformed_names = []
                                        if not transformed_names:
                                            model_names = list(
                                                getattr(tree_model, "feature_names_in_", [])
                                            )
                                            transformed_names = model_names or [
                                                f"Feature {i + 1}" for i in range(transformed_values.shape[1])
                                            ]

                                    # TreeExplainer is designed for tree estimators such as XGBoost.
                                    explainer = shap.TreeExplainer(tree_model)
                                    shap_values = explainer.shap_values(
                                        transformed_values,
                                        check_additivity=False,
                                    )

                                    if isinstance(shap_values, list):
                                        contribution_values = np.asarray(shap_values[-1]).reshape(-1)
                                    else:
                                        contribution_array = np.asarray(shap_values)
                                        if contribution_array.ndim == 3:
                                            # Common layouts: (rows, features, classes) or
                                            # (classes, rows, features), depending on SHAP version.
                                            if contribution_array.shape[0] == 1:
                                                contribution_values = contribution_array[0, :, -1]
                                            elif contribution_array.shape[1] == 1:
                                                contribution_values = contribution_array[-1, 0, :]
                                            else:
                                                contribution_values = contribution_array[0, :, -1]
                                        elif contribution_array.ndim == 2:
                                            contribution_values = contribution_array[0]
                                        else:
                                            contribution_values = contribution_array.reshape(-1)

                                    feature_values = transformed_values.reshape(-1)
                                    if len(contribution_values) != len(transformed_names):
                                        raise ValueError(
                                            "SHAP output feature count does not match transformed model inputs."
                                        )

                                    explanation_df = pd.DataFrame({
                                        "Feature": transformed_names,
                                        "SHAP contribution": contribution_values,
                                        "Feature value": feature_values,
                                    })
                                    explanation_df["Direction"] = np.where(
                                        explanation_df["SHAP contribution"] >= 0,
                                        "Pushes toward fraud",
                                        "Pushes away from fraud",
                                    )
                                    explanation_df["Absolute contribution"] = (
                                        explanation_df["SHAP contribution"].abs()
                                    )
                                    top_features = explanation_df.sort_values(
                                        "Absolute contribution", ascending=False
                                    ).head(15).sort_values("SHAP contribution")

                                    fig = px.bar(
                                        top_features,
                                        x="SHAP contribution",
                                        y="Feature",
                                        orientation="h",
                                        color="Direction",
                                        color_discrete_map={
                                            "Pushes toward fraud": "#ff626d",
                                            "Pushes away from fraud": "#55aaff",
                                        },
                                        hover_data=["Feature value"],
                                        title="Top feature contributions for this transaction",
                                    )
                                    fig.update_layout(
                                        template="plotly_dark",
                                        paper_bgcolor="rgba(0,0,0,0)",
                                        plot_bgcolor="rgba(0,0,0,0)",
                                        height=520,
                                        legend_title_text="Contribution direction",
                                        margin=dict(l=10, r=10, t=55, b=10),
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    st.dataframe(
                                        explanation_df.sort_values(
                                            "Absolute contribution", ascending=False
                                        )[[
                                            "Feature", "Feature value",
                                            "SHAP contribution", "Direction",
                                        ]].head(15),
                                        use_container_width=True,
                                        hide_index=True,
                                    )
                    except ImportError:
                        st.error(
                            "SHAP is not installed in this environment. "
                            "Install it with: python -m pip install shap"
                        )
                    except Exception as exc:
                        st.error(
                            "SHAP explanation could not be generated for this transaction. "
                            f"Details: {exc}"
                        )

                st.info(
                    "The fraud probability is the model's estimate, not proof of fraud. "
                    "Use SHAP to understand model behaviour and investigate cases using "
                    "appropriate independent evidence."
                )

    # ---------------------------------------------------------
    # TAB 5: Data quality
    # ---------------------------------------------------------
    elif selected_section == "Data Quality":
        st.title("Data Quality")
        st.caption(
            "Audit transaction files for missing values, duplicates, invalid values, "
            "schema problems, and target-label issues before model scoring."
        )

        st.markdown("### Choose a dataset to audit")
        source_mode = st.radio(
            "Data source",
            ["Project dataset split", "Upload a CSV"],
            horizontal=True,
            key="quality_source_mode",
        )

        quality_df = None
        quality_source_name = ""

        if source_mode == "Project dataset split":
            split_options = {
                "Training split": DATA_DIR / "train.csv.gz",
                "Validation split": DATA_DIR / "validation.csv",
                "Test split": DATA_DIR / "test.csv",
                "Latest scored transactions": None,
            }
            selected_split = st.selectbox(
                "Select project data",
                list(split_options.keys()),
                key="quality_split_select",
            )
            if selected_split == "Latest scored transactions":
                saved_output = st.session_state.get("creditguard_scored_output")
                if saved_output is not None and not saved_output.empty:
                    quality_df = saved_output.copy()
                    quality_source_name = "Latest scored transactions"
                else:
                    st.info(
                        "No scored transactions are available in this session. "
                        "Score a CSV in Transaction Scoring first, or choose a dataset split."
                    )
            else:
                selected_path = split_options[selected_split]
                if selected_path.exists():
                    quality_df = load_csv(str(selected_path))
                    quality_source_name = selected_split
                else:
                    st.warning(
                        f"{selected_split} was not found at `{selected_path}`. "
                        "Run the data preparation step or upload a CSV instead."
                    )
        else:
            quality_upload = st.file_uploader(
                "Upload CSV for quality checks",
                type=["csv"],
                key="quality_csv_upload",
                help="The file is inspected in this Streamlit session and is not modified.",
            )
            if quality_upload is not None:
                try:
                    quality_df = pd.read_csv(quality_upload, low_memory=False)
                    quality_source_name = quality_upload.name
                except Exception as exc:
                    st.error(f"Could not read this CSV: {exc}")

        if quality_df is not None:
            df_quality = quality_df
            row_count, column_count = df_quality.shape
            missing_cells = int(df_quality.isna().sum().sum())
            duplicate_rows = int(df_quality.duplicated().sum())
            duplicate_columns = int(df_quality.columns.duplicated().sum())
            empty_columns = [
                str(column) for column in df_quality.columns
                if df_quality[column].isna().all()
            ]
            constant_columns = [
                str(column) for column in df_quality.columns
                if df_quality[column].nunique(dropna=False) <= 1
            ]
            numeric_columns = df_quality.select_dtypes(include=[np.number]).columns.tolist()
            non_finite_counts = {}
            for column in numeric_columns:
                values = pd.to_numeric(df_quality[column], errors="coerce").to_numpy(dtype=float)
                bad_count = int((~np.isfinite(values) & ~np.isnan(values)).sum())
                if bad_count:
                    non_finite_counts[str(column)] = bad_count
            missing_by_column = df_quality.isna().sum()
            missing_by_column = missing_by_column[missing_by_column > 0].sort_values(ascending=False)

            st.caption(f"Auditing: **{quality_source_name}** · {row_count:,} rows · {column_count:,} columns")
            metric_cols = st.columns(4)
            with metric_cols[0]:
                metric_card("Rows", f"{row_count:,}")
            with metric_cols[1]:
                metric_card("Columns", f"{column_count:,}")
            with metric_cols[2]:
                metric_card("Missing cells", f"{missing_cells:,}")
            with metric_cols[3]:
                metric_card("Exact duplicate rows", f"{duplicate_rows:,}")

            st.markdown("### Validation checks")
            checks = []
            checks.append({
                "Check": "Dataset is not empty",
                "Status": "PASS" if row_count > 0 and column_count > 0 else "FAIL",
                "Details": f"{row_count:,} rows and {column_count:,} columns",
            })
            checks.append({
                "Check": "Missing values",
                "Status": "PASS" if missing_cells == 0 else "REVIEW",
                "Details": "No missing cells" if missing_cells == 0 else f"{missing_cells:,} missing cells",
            })
            checks.append({
                "Check": "Exact duplicate rows",
                "Status": "PASS" if duplicate_rows == 0 else "REVIEW",
                "Details": "No exact duplicate rows" if duplicate_rows == 0 else f"{duplicate_rows:,} repeated rows; review before removing",
            })
            checks.append({
                "Check": "Duplicate column names",
                "Status": "PASS" if duplicate_columns == 0 else "FAIL",
                "Details": "Column names are unique" if duplicate_columns == 0 else f"{duplicate_columns} duplicate column names",
            })
            checks.append({
                "Check": "Infinite numeric values",
                "Status": "PASS" if not non_finite_counts else "FAIL",
                "Details": "No positive or negative infinity" if not non_finite_counts else ", ".join(f"{name}: {count}" for name, count in non_finite_counts.items()),
            })
            if TARGET in df_quality.columns:
                target_values = set(pd.Series(df_quality[TARGET].dropna().unique()).tolist())
                valid_target = target_values.issubset({0, 1, 0.0, 1.0})
                checks.append({
                    "Check": "Class target labels",
                    "Status": "PASS" if valid_target and df_quality[TARGET].notna().all() else "FAIL",
                    "Details": "Class contains only 0/1 labels" if valid_target and df_quality[TARGET].notna().all() else f"Unexpected or missing target labels: {sorted(map(str, target_values))}",
                })
            else:
                checks.append({
                    "Check": "Class target labels",
                    "Status": "INFO",
                    "Details": "No Class column present; acceptable for unlabeled scoring files",
                })
            st.dataframe(pd.DataFrame(checks), use_container_width=True, hide_index=True)

            left_col, right_col = st.columns(2)
            with left_col:
                st.markdown("### Missing values by column")
                if missing_by_column.empty:
                    st.success("No missing values detected.")
                else:
                    missing_df = missing_by_column.rename("Missing count").reset_index()
                    missing_df.columns = ["Column", "Missing count"]
                    st.dataframe(missing_df, use_container_width=True, hide_index=True)
            with right_col:
                st.markdown("### Target distribution")
                if TARGET in df_quality.columns:
                    target_counts = (
                        df_quality[TARGET].value_counts(dropna=False)
                        .rename_axis("Class")
                        .reset_index(name="Count")
                    )
                    target_counts["Percentage"] = target_counts["Count"] / max(row_count, 1) * 100
                    st.dataframe(target_counts, use_container_width=True, hide_index=True)
                    if set(df_quality[TARGET].dropna().unique()).issubset({0, 1, 0.0, 1.0}):
                        fig = px.bar(
                            target_counts,
                            x="Class",
                            y="Count",
                            title="Class balance",
                            color="Class",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No `Class` target column is present in this dataset.")

            st.markdown("### Column profile")
            profile = pd.DataFrame({
                "Column": df_quality.columns.astype(str),
                "Data type": [str(dtype) for dtype in df_quality.dtypes],
                "Missing": [int(value) for value in df_quality.isna().sum().values],
                "Unique values": [int(df_quality[column].nunique(dropna=True)) for column in df_quality.columns],
            })
            profile["Missing %"] = (profile["Missing"] / max(row_count, 1) * 100).round(3)
            st.dataframe(profile, use_container_width=True, hide_index=True)
            target_counts_for_pdf = None
            if TARGET in df_quality.columns:
                target_counts_for_pdf = (
                    df_quality[TARGET].value_counts(dropna=False)
                    .rename_axis("Class").reset_index(name="Count")
                )
                target_counts_for_pdf["Percentage"] = (
                    target_counts_for_pdf["Count"] / max(row_count, 1) * 100
                ).round(4)

            pdf_bytes = build_quality_pdf(
                quality_source_name,
                df_quality,
                checks,
                profile,
                missing_by_column,
                target_counts_for_pdf,
            )
            st.download_button(
                "Download PDF quality report",
                data=pdf_bytes,
                file_name="creditguard_data_quality_report.pdf",
                mime="application/pdf",
                key="download_data_quality_pdf",
                help="Download the audit summary, validation checks, column profile, and available quality charts as a PDF.",
            )
            if empty_columns or constant_columns:
                with st.expander("Review empty or constant columns"):
                    if empty_columns:
                        st.write("**Completely empty columns:**", empty_columns)
                    if constant_columns:
                        st.write("**Columns with one unique value:**", constant_columns)
            st.info(
                "This page reports quality issues but does not automatically delete rows, "
                "impute values, or modify your source files. Duplicate rows may be valid "
                "in transaction data and should be investigated before removal."
            )

    # ---------------------------------------------------------
    # TAB 6: Model monitoring
    # ---------------------------------------------------------
    elif selected_section == "Model Monitoring":
        st.title("Model Monitoring")
        st.caption("Monitor incoming transaction data for distribution changes and track scoring behaviour against the held-out test reference.")
        reference_df = test
        if reference_df is None or reference_df.empty:
            st.warning("The processed test split was not found. Run the data preparation workflow first.")
        else:
            scored_df = st.session_state.get("creditguard_scored_output")
            st.markdown("### Monitoring source")
            monitoring_mode = st.radio(
                "Current batch",
                ["Latest scored transactions", "Upload a CSV batch"],
                horizontal=True,
                key="monitoring_source_mode",
            )
            current_df = None
            current_name = ""
            if monitoring_mode == "Latest scored transactions":
                if scored_df is not None and not scored_df.empty:
                    current_df = scored_df.copy()
                    current_name = "Latest scored transactions"
                else:
                    st.info("No scored batch is available in this session. Score a CSV in Transaction Scoring, or upload a batch here.")
            else:
                monitoring_upload = st.file_uploader("Upload current batch CSV", type=["csv"], key="monitoring_csv_upload")
                if monitoring_upload is not None:
                    try:
                        current_df = pd.read_csv(monitoring_upload, low_memory=False)
                        current_name = monitoring_upload.name
                    except Exception as exc:
                        st.error(f"Could not read the monitoring CSV: {exc}")

            if current_df is not None:
                model = load_model(str(MODEL_PATH))
                feature_columns = list(getattr(model, "feature_names_in_", [])) if model is not None else []
                if not feature_columns:
                    feature_columns = [c for c in reference_df.columns if c != TARGET]
                feature_columns = [c for c in feature_columns if c in reference_df.columns and c in current_df.columns and c != TARGET]
                if not feature_columns:
                    st.error("No shared model feature columns were found between the reference test split and the current batch.")
                else:
                    ref_features = reference_df[feature_columns].copy()
                    cur_features = current_df[feature_columns].copy()
                    for col in feature_columns:
                        ref_features[col] = pd.to_numeric(ref_features[col], errors="coerce")
                        cur_features[col] = pd.to_numeric(cur_features[col], errors="coerce")
                    total_rows = len(cur_features)
                    missing_cells = int(cur_features.isna().sum().sum())
                    infinite_cells = 0
                    for col in feature_columns:
                        vals = cur_features[col].to_numpy(dtype=float)
                        infinite_cells += int(np.isinf(vals).sum())

                    score_col = "Fraud Probability" if "Fraud Probability" in current_df.columns else None
                    pred_col = "Predicted Class" if "Predicted Class" in current_df.columns else None
                    flagged_count = int((pd.to_numeric(current_df[pred_col], errors="coerce") == 1).sum()) if pred_col else None
                    mean_probability = float(pd.to_numeric(current_df[score_col], errors="coerce").mean()) if score_col else None

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Current batch rows", f"{total_rows:,}")
                    m2.metric("Feature columns compared", f"{len(feature_columns):,}")
                    m3.metric("Missing feature cells", f"{missing_cells:,}")
                    m4.metric("Infinite values", f"{infinite_cells:,}")
                    if score_col:
                        a, b, c = st.columns(3)
                        a.metric("Transactions flagged", f"{flagged_count:,}" if flagged_count is not None else "Not available")
                        b.metric("Flag rate", f"{flagged_count / max(total_rows, 1):.2%}" if flagged_count is not None else "Not available")
                        c.metric("Mean fraud probability", f"{mean_probability:.2%}" if mean_probability is not None and np.isfinite(mean_probability) else "Not available")

                    st.markdown("### Feature drift against test reference")
                    st.caption("PSI is a screening indicator: below 0.10 is usually small, 0.10–0.25 merits review, and above 0.25 suggests a larger shift. This is not proof of model degradation.")
                    drift_rows = []
                    for col in feature_columns:
                        ref_vals = ref_features[col].replace([np.inf, -np.inf], np.nan).dropna()
                        cur_vals = cur_features[col].replace([np.inf, -np.inf], np.nan).dropna()
                        psi = calculate_psi(ref_vals, cur_vals)
                        ref_std = float(ref_vals.std()) if len(ref_vals) > 1 else np.nan
                        mean_shift = ((float(cur_vals.mean()) - float(ref_vals.mean())) / ref_std) if len(cur_vals) and len(ref_vals) and np.isfinite(ref_std) and ref_std > 0 else np.nan
                        if not np.isfinite(psi):
                            status = "Insufficient data"
                        elif psi >= 0.25:
                            status = "High drift"
                        elif psi >= 0.10:
                            status = "Review"
                        else:
                            status = "Low drift"
                        drift_rows.append({"Feature": col, "PSI": psi, "Mean shift (reference SD)": mean_shift, "Reference rows": len(ref_vals), "Current rows": len(cur_vals), "Status": status})
                    drift_df = pd.DataFrame(drift_rows).sort_values("PSI", ascending=False, na_position="last")
                    st.dataframe(drift_df, use_container_width=True, hide_index=True)
                    plot_df = drift_df.dropna(subset=["PSI"]).head(20).sort_values("PSI", ascending=True)
                    if not plot_df.empty:
                        fig = px.bar(plot_df, x="PSI", y="Feature", orientation="h", color="Status", title="Highest feature drift scores (top 20)")
                        fig.add_vline(x=0.10, line_dash="dash", line_color="#e8b44b", annotation_text="Review 0.10")
                        fig.add_vline(x=0.25, line_dash="dash", line_color="#ff5e70", annotation_text="High 0.25")
                        st.plotly_chart(fig, use_container_width=True)
                    monitoring_summary = {
                        "Reference dataset": "Processed test split",
                        "Current batch": current_name,
                        "Current batch rows": f"{total_rows:,}",
                        "Feature columns compared": f"{len(feature_columns):,}",
                        "Missing feature cells": f"{missing_cells:,}",
                        "Infinite values": f"{infinite_cells:,}",
                        "High-drift features (PSI ≥ 0.25)": int((drift_df["Status"] == "High drift").sum()),
                        "Features needing review (PSI 0.10–0.25)": int((drift_df["Status"] == "Review").sum()),
                    }
                    if flagged_count is not None:
                        monitoring_summary["Transactions flagged"] = f"{flagged_count:,}"
                        monitoring_summary["Flag rate"] = f"{flagged_count / max(total_rows, 1):.2%}"
                    if mean_probability is not None and np.isfinite(mean_probability):
                        monitoring_summary["Mean fraud probability"] = f"{mean_probability:.2%}"

                    st.markdown("### Monitoring history")
                    st.caption("Save a snapshot to compare this batch with future runs. Snapshots are stored locally in reports/monitoring_history.json and remain available after restarting Streamlit on this machine.")
                    snapshot = {
                        "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Batch": str(current_name),
                        "Rows": int(total_rows),
                        "Feature columns": int(len(feature_columns)),
                        "Missing cells": int(missing_cells),
                        "Infinite values": int(infinite_cells),
                        "Flagged transactions": int(flagged_count) if flagged_count is not None else None,
                        "Flag rate": float(flagged_count / max(total_rows, 1)) if flagged_count is not None else None,
                        "Mean fraud probability": float(mean_probability) if mean_probability is not None and np.isfinite(mean_probability) else None,
                        "High-drift features": int((drift_df["Status"] == "High drift").sum()),
                        "Features needing review": int((drift_df["Status"] == "Review").sum()),
                        "Mean PSI": float(drift_df["PSI"].mean()) if drift_df["PSI"].notna().any() else None,
                        "Maximum PSI": float(drift_df["PSI"].max()) if drift_df["PSI"].notna().any() else None,
                    }
                    if st.button("Save current snapshot to history", key="save_monitoring_snapshot", type="primary"):
                        try:
                            saved_history = save_monitoring_snapshot(snapshot)
                            st.success(f"Monitoring snapshot saved. History now contains {len(saved_history):,} snapshots.")
                        except OSError as exc:
                            st.error(f"Could not save monitoring history: {exc}")

                    monitoring_pdf = build_monitoring_pdf(current_name, drift_df, monitoring_summary)
                    st.download_button(
                        "Download PDF monitoring report",
                        data=monitoring_pdf,
                        file_name="creditguard_model_monitoring_report.pdf",
                        mime="application/pdf",
                        key="download_model_monitoring_pdf",
                    )

                    if TARGET in current_df.columns:
                        st.markdown("### Label-based performance monitoring")
                        actual = pd.to_numeric(current_df[TARGET], errors="coerce")
                        if pred_col and actual.notna().all() and set(actual.unique()).issubset({0, 1}):
                            from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
                            predicted = pd.to_numeric(current_df[pred_col], errors="coerce").fillna(0).astype(int)
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Fraud precision", f"{precision_score(actual, predicted, zero_division=0):.2%}")
                            c2.metric("Fraud recall", f"{recall_score(actual, predicted, zero_division=0):.2%}")
                            c3.metric("Fraud F1-score", f"{f1_score(actual, predicted, zero_division=0):.3f}")
                        else:
                            st.info("Class labels are present, but actual-versus-predicted metrics require a scored batch with both valid Class labels and Predicted Class values.")
                    else:
                        st.info("No actual Class labels are present. This view can monitor data and prediction patterns, but cannot calculate real-world precision or recall without confirmed labels.")
                    st.warning("Monitoring compares the current batch with the processed test split as a reference. Investigate meaningful shifts before retraining; do not treat drift alone as proof of fraud or performance loss.")

    # ---------------------------------------------------------
    # TAB 7: Risk alerts
    # ---------------------------------------------------------
    elif selected_section == "Risk Alerts":
        st.title("Risk Alerts")
        st.caption("Prioritize transactions that need review using the latest scored batch. Alerts are model signals, not proof of fraud.")
        alert_source = st.session_state.get("creditguard_scored_output")
        if alert_source is None or alert_source.empty:
            st.info("No scored transactions are available in this session. Open Transaction Scoring and score a batch first; the latest results will appear here.")
        elif "Fraud Probability" not in alert_source.columns:
            st.warning("The current scored batch does not contain a Fraud Probability column. Score the batch again to create risk alerts.")
        else:
            alerts = alert_source.copy().reset_index(drop=False).rename(columns={"index": "Transaction Row"})
            alerts["Fraud Probability"] = pd.to_numeric(alerts["Fraud Probability"], errors="coerce")
            if "Predicted Class" not in alerts.columns:
                threshold_value = float(st.session_state.get("creditguard_scored_threshold", 0.5))
                alerts["Predicted Class"] = (alerts["Fraud Probability"] >= threshold_value).astype(int)
            alerts["Risk Level"] = pd.cut(
                alerts["Fraud Probability"],
                bins=[-np.inf, 0.50, 0.80, 0.95, np.inf],
                labels=["Low", "Elevated", "High", "Critical"],
                right=False,
            ).astype("object")
            threshold_value = float(st.session_state.get("creditguard_scored_threshold", 0.5))
            flagged = pd.to_numeric(alerts["Predicted Class"], errors="coerce").fillna(0).astype(int) == 1
            critical = alerts["Fraud Probability"] >= 0.95
            high_risk = alerts["Fraud Probability"] >= 0.80
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Transactions scored", f"{len(alerts):,}")
            k2.metric("Flagged by model", f"{int(flagged.sum()):,}")
            k3.metric("High risk (≥ 80%)", f"{int(high_risk.sum()):,}")
            k4.metric("Critical risk (≥ 95%)", f"{int(critical.sum()):,}")

            st.markdown("### Alert filters")
            f1, f2, f3 = st.columns([1, 1, 1.3])
            with f1:
                risk_choice = st.selectbox("Risk level", ["All", "Critical", "High", "Elevated", "Low"], key="risk_alert_level")
            with f2:
                prediction_choice = st.selectbox("Model prediction", ["All", "Fraud", "Legitimate"], key="risk_alert_prediction")
            with f3:
                minimum_probability = st.slider("Minimum fraud probability", 0.0, 1.0, 0.80, 0.01, key="risk_alert_min_probability")

            filtered_alerts = alerts[alerts["Fraud Probability"].fillna(-1) >= minimum_probability].copy()
            if risk_choice != "All":
                filtered_alerts = filtered_alerts[filtered_alerts["Risk Level"] == risk_choice]
            if prediction_choice == "Fraud":
                filtered_alerts = filtered_alerts[pd.to_numeric(filtered_alerts["Predicted Class"], errors="coerce") == 1]
            elif prediction_choice == "Legitimate":
                filtered_alerts = filtered_alerts[pd.to_numeric(filtered_alerts["Predicted Class"], errors="coerce") == 0]
            filtered_alerts = filtered_alerts.sort_values("Fraud Probability", ascending=False, na_position="last")
            display_cols = [c for c in ["Transaction Row", "Prediction", "Predicted Class", "Risk Level", "Fraud Probability", "Time", "Amount"] if c in filtered_alerts.columns]
            st.caption(f"Showing {len(filtered_alerts):,} of {len(alerts):,} scored transactions, sorted by estimated fraud probability.")
            if filtered_alerts.empty:
                st.info("No transactions match these alert filters. Lower the minimum probability or choose a different risk level.")
            else:
                shown = filtered_alerts[display_cols].head(500).copy()
                if "Fraud Probability" in shown:
                    shown["Fraud Probability"] = shown["Fraud Probability"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
                st.dataframe(shown, use_container_width=True, hide_index=True)
                if len(filtered_alerts) > 500:
                    st.caption("For readability, the table shows the top 500 matching alerts. The PDF report includes a summary and the highest-risk cases.")

            alert_summary = {
                "Scored transactions": f"{len(alerts):,}",
                "Flagged by model": f"{int(flagged.sum()):,}",
                "High risk (≥ 80%)": f"{int(high_risk.sum()):,}",
                "Critical risk (≥ 95%)": f"{int(critical.sum()):,}",
                "Scoring threshold": f"{threshold_value:.2f}",
                "Rows matching filters": f"{len(filtered_alerts):,}",
            }
            alert_pdf = build_monitoring_pdf("Risk Alerts — latest scored batch", filtered_alerts[display_cols].head(100), alert_summary)
            st.download_button("Download PDF risk alert report", data=alert_pdf, file_name="creditguard_risk_alert_report.pdf", mime="application/pdf", key="download_risk_alert_pdf")
            st.warning("Risk levels are based on model-estimated probabilities. Investigate high-priority transactions using independent evidence before taking action.")

    # ---------------------------------------------------------
    # TAB 8: Monitoring history
    # ---------------------------------------------------------
    elif selected_section == "Monitoring History":
        st.title("Monitoring History")
        st.caption("Review saved monitoring snapshots over time. History is stored locally on this machine and is not automatically shared or uploaded.")
        history = load_monitoring_history()
        if history.empty:
            st.info("No monitoring snapshots have been saved yet. Open Model Monitoring, review a batch, then select Save current snapshot to history.")
        else:
            history["Timestamp"] = pd.to_datetime(history["Timestamp"], errors="coerce")
            history = history.sort_values("Timestamp", ascending=False, na_position="last").reset_index(drop=True)
            latest = history.iloc[0]
            previous = history.iloc[1] if len(history) > 1 else None
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Saved snapshots", f"{len(history):,}")
            c2.metric("Latest batch rows", f"{int(latest.get('Rows', 0)):,}")
            c3.metric("High-drift features", f"{int(latest.get('High-drift features', 0)):,}")
            max_psi = latest.get("Maximum PSI")
            c4.metric("Maximum PSI", f"{float(max_psi):.3f}" if pd.notna(max_psi) else "N/A")

            st.markdown("### Snapshot timeline")
            timeline = history.copy().sort_values("Timestamp")
            timeline["Snapshot"] = timeline["Timestamp"].dt.strftime("%Y-%m-%d %H:%M")
            if "Flag rate" in timeline and timeline["Flag rate"].notna().any():
                fig = px.line(timeline, x="Snapshot", y="Flag rate", markers=True, title="Fraud flag rate across saved batches")
                fig.update_yaxes(tickformat=".2%")
                st.plotly_chart(fig, use_container_width=True)
            if "Maximum PSI" in timeline and timeline["Maximum PSI"].notna().any():
                fig = px.line(timeline, x="Snapshot", y="Maximum PSI", markers=True, title="Maximum feature drift (PSI) across saved batches")
                fig.add_hline(y=0.10, line_dash="dash", annotation_text="Review 0.10")
                fig.add_hline(y=0.25, line_dash="dash", annotation_text="High drift 0.25")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Saved snapshots")
            display_history = history.copy()
            display_history["Timestamp"] = display_history["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
            if "Flag rate" in display_history:
                display_history["Flag rate"] = display_history["Flag rate"].map(lambda v: f"{float(v):.2%}" if pd.notna(v) else "—")
            if "Mean fraud probability" in display_history:
                display_history["Mean fraud probability"] = display_history["Mean fraud probability"].map(lambda v: f"{float(v):.2%}" if pd.notna(v) else "—")
            st.dataframe(display_history, use_container_width=True, hide_index=True)

            history_pdf = build_table_pdf(
                "CreditGuard | Monitoring History",
                f"Generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} — saved monitoring snapshots",
                display_history,
                summary={
                    "Snapshots stored": f"{len(history):,}",
                    "Latest batch": str(latest.get("Batch", "—")),
                    "Latest batch rows": f"{int(latest.get('Rows', 0)):,}",
                    "Latest maximum PSI": f"{float(max_psi):.4f}" if pd.notna(max_psi) else "Not available",
                },
                max_rows=35,
            )
            st.download_button(
                "Download PDF monitoring history report",
                data=history_pdf,
                file_name="creditguard_monitoring_history.pdf",
                mime="application/pdf",
                key="download_monitoring_history_pdf",
            )
            st.info("Interpret PSI as a screening signal, not proof of model degradation. Flag rates can change because the transaction mix changes; use confirmed labels to evaluate actual model performance.")

    # ---------------------------------------------------------
    # TAB 9: Project information
    # ---------------------------------------------------------
    elif selected_section == "Project Information":
        st.title("Project Information")
        st.caption("Project objective, models, technology, and limitations.")
        st.subheader("About This Project")

        st.markdown(
            """
            **Objective:** Detect potentially fraudulent credit-card
            transactions using machine learning.

            **Models developed:**
            - Logistic Regression
            - Decision Tree
            - Random Forest
            - XGBoost
            - Linear SVM
            - Isolation Forest

            **Technology stack:** Python, pandas, scikit-learn,
            XGBoost, Plotly, and Streamlit.

            **Evaluation design:** Chronological training, validation,
            and test partitions, with preprocessing fitted on training
            data and thresholds selected using validation data.
            """
        )

        st.warning(
            "This is an educational prototype, not a production "
            "banking decision system. Do not use it as the sole basis "
            "for approving, declining, or blocking financial transactions."
        )


if __name__ == "__main__":
    main()
