"""Streamlit AML monitoring dashboard."""
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from src.accounts import load_account_sample
from src.catalog import discover_variants
from src.data import demo_transactions, load_transaction_sample, normalize_transactions
from src.patterns import summarize_patterns
from src.risk import score_transactions

st.set_page_config(page_title="ClearWatch | AML Monitoring", page_icon="◈", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.block-container {padding-top:1.5rem;max-width:1500px}
div[data-testid="stMetric"] {background:#101b29;border:1px solid #1e3042;padding:16px 18px;border-radius:12px}
div[data-testid="stMetricLabel"] {color:#91a2b5}
.eyebrow {font:500 11px 'DM Mono';letter-spacing:2px;color:#45d6ad;text-transform:uppercase}
.muted {color:#91a2b5;font-size:13px}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="eyebrow">CLEARWATCH / TRANSACTION INTELLIGENCE</div>', unsafe_allow_html=True)
st.title("AML monitoring")
st.markdown('<p class="muted">Monitor activity, triage risk signals, and support analyst review.</p>', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def cached_transaction_sample(path: str, nrows: int, file_size: int, modified_ns: int) -> pd.DataFrame:
    return load_transaction_sample(path, nrows=nrows)

@st.cache_data(show_spinner=False)
def cached_account_sample(path: str, nrows: int, file_size: int, modified_ns: int) -> pd.DataFrame:
    return load_account_sample(path, nrows=nrows)

@st.cache_data(show_spinner=False)
def cached_pattern_summary(path: str, modified_ns: int) -> dict:
    return summarize_patterns(path)

variants = discover_variants()
variant_map = {variant.name: variant for variant in variants}

with st.sidebar:
    st.markdown("## ◈ ClearWatch")
    st.caption("TRANSACTION MONITORING")
    st.divider()
    uploaded = st.file_uploader("Load another transaction CSV", type=["csv"], help="Use this for smaller CSVs. The local dataset selector below supports large files without uploading them into the browser.")
    selected_variant = None
    if variants:
        default_index = next((i for i, variant in enumerate(variants) if variant.name == "HI-Large"), 0)
        selected_name = st.selectbox("IBM AML variant", [variant.name for variant in variants], index=default_index)
        selected_variant = variant_map[selected_name]
        local_path = st.text_input("Transaction CSV path", value=str(selected_variant.transaction_path), key=f"transaction_path_{selected_name}")
        available_companions = [label for label, path in (("Accounts", selected_variant.accounts_path), ("Patterns", selected_variant.patterns_path)) if path and path.is_file()]
        st.caption(f"Companion files found: {', '.join(available_companions) if available_companions else 'none'}")
    else:
        local_path = st.text_input("Transaction CSV path", value="data/HI-Small_Trans.csv")
    preview_rows = st.select_slider("Rows to sample across the file", options=[10000, 25000, 50000, 100000, 250000], value=100000, format_func=lambda n: f"{n:,}")
    st.caption("Large files are sampled across the full file, not just from their first rows. Without a local or uploaded CSV, demo records are used.")
    st.divider()
    st.caption("Risk thresholds")
    medium_cutoff = st.slider("Medium risk from", 10, 80, 30)
    high_cutoff = st.slider("High risk from", 20, 100, 65)
    if high_cutoff <= medium_cutoff:
        st.warning("High-risk threshold should exceed medium-risk threshold.")

try:
    if uploaded is not None:
        transactions = normalize_transactions(pd.read_csv(uploaded, nrows=preview_rows)); data_mode = f"Uploaded data · up to {preview_rows:,} initial rows"
    elif Path(local_path).is_file():
        csv_path = Path(local_path)
        transactions = cached_transaction_sample(str(csv_path), preview_rows, csv_path.stat().st_size, csv_path.stat().st_mtime_ns)
        data_mode = f"{selected_variant.name if selected_variant else 'Local data'} · file-wide sample of {len(transactions):,} · {local_path}"
    else:
        transactions = demo_transactions(); data_mode = "Illustrative demo data"
    scored = score_transactions(transactions)
except Exception as exc:
    st.error(f"Could not load or score this data: {exc}")
    st.stop()

if high_cutoff > medium_cutoff:
    scored["risk_level"] = scored["risk_score"].map(lambda score: "High Risk" if score >= high_cutoff else "Medium Risk" if score >= medium_cutoff else "Low Risk")
scored["timestamp"] = pd.to_datetime(scored["timestamp"], errors="coerce")

st.caption(f"{data_mode}  ·  {len(scored):,} transactions  ·  Scoring: {scored['scoring_method'].iloc[0]}")
total = len(scored); high = int((scored.risk_level == "High Risk").sum()); medium = int((scored.risk_level == "Medium Risk").sum())
known_labels = int(scored.is_laundering.sum()); label_rate = f"{known_labels / total:.2%}" if total else "—"
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Transactions", f"{total:,}")
c2.metric("High risk", f"{high:,}", f"{high / total:.1%} of volume" if total else "—")
c3.metric("Medium risk", f"{medium:,}")
c4.metric("Labeled laundering", label_rate, help="Observed dataset labels; demo labels are synthetic and illustrative.")
c5.metric("Total amount", f"{scored.amount_paid.sum():,.0f}")

left, right = st.columns([1.55, 1])
with left:
    st.subheader("Transaction volume")
    dated = scored.dropna(subset=["timestamp"])
    if not dated.empty:
        trend = dated.set_index("timestamp").resample("D").agg(transactions=("amount_paid", "size"), amount=("amount_paid", "sum")).reset_index()
        fig = px.area(trend, x="timestamp", y="transactions", color_discrete_sequence=["#45d6ad"])
        fig.update_traces(line_width=2, fillcolor="rgba(69,214,173,0.12)")
        fig.update_layout(height=310, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#91a2b5", xaxis_title="", yaxis_title="Transactions", hovermode="x unified")
        fig.update_xaxes(showgrid=False); fig.update_yaxes(gridcolor="#223244")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Add a timestamp column to view activity over time.")
with right:
    st.subheader("Risk distribution")
    risk_counts = scored.risk_level.value_counts().reindex(["High Risk", "Medium Risk", "Low Risk"], fill_value=0).rename_axis("Risk").reset_index(name="Transactions")
    fig = px.pie(risk_counts, names="Risk", values="Transactions", hole=.68, color="Risk", color_discrete_map={"High Risk":"#ff667d","Medium Risk":"#f6bd60","Low Risk":"#45d6ad"})
    fig.update_layout(height=310, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor="rgba(0,0,0,0)", font_color="#c8d2de", legend=dict(orientation="h",y=-.1))
    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Payment format")
    fmt = scored.groupby("payment_format", as_index=False).agg(Transactions=("amount_paid", "size"), Amount=("amount_paid", "sum")).sort_values("Transactions", ascending=True)
    fig = px.bar(fmt, x="Transactions", y="payment_format", orientation="h", color_discrete_sequence=["#6688ff"])
    fig.update_layout(height=270, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#91a2b5", xaxis_title="Transactions", yaxis_title="")
    fig.update_xaxes(gridcolor="#223244"); fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Investigation queue")
    queue = scored[scored.risk_level != "Low Risk"].sort_values("risk_score", ascending=False).head(8)
    if queue.empty:
        st.success("No transactions currently meet the selected review thresholds.")
    else:
        for i, row in queue.iterrows():
            badge = "🔴" if row.risk_level == "High Risk" else "🟠"
            st.markdown(f"**{badge} {row.from_bank} → {row.to_bank}**　`{row.risk_score:02d}/100`")
            st.caption(f"{row.amount_paid:,.2f} {row.payment_currency} · {row.payment_format} · {row.timestamp.strftime('%b %d, %H:%M') if pd.notna(row.timestamp) else 'Time unavailable'}")

st.divider()
st.subheader("Transaction explorer")
f1, f2, f3 = st.columns([1,1,2])
levels = f1.multiselect("Risk level", ["High Risk", "Medium Risk", "Low Risk"], default=["High Risk", "Medium Risk"])
formats = sorted(scored.payment_format.dropna().unique())
selected_formats = f2.multiselect("Payment format", formats, default=formats)
search = f3.text_input("Search bank", placeholder="Enter bank name or identifier")
filtered = scored[scored.risk_level.isin(levels) & scored.payment_format.isin(selected_formats)]
if search:
    filtered = filtered[filtered.from_bank.str.contains(search, case=False, na=False) | filtered.to_bank.str.contains(search, case=False, na=False)]
view = filtered.sort_values("risk_score", ascending=False)[["timestamp","from_bank","from_account","to_bank","to_account","amount_paid","amount_received","payment_currency","payment_format","risk_score","risk_level"]]
st.dataframe(view, use_container_width=True, hide_index=True, column_config={"timestamp":st.column_config.DatetimeColumn("Timestamp", format="MMM D, YYYY HH:mm"), "amount_paid":st.column_config.NumberColumn("Amount paid", format="%.2f"), "amount_received":st.column_config.NumberColumn("Amount received", format="%.2f"), "risk_score":st.column_config.ProgressColumn("Risk score", min_value=0, max_value=100, format="%d")})
st.download_button("Download filtered transactions", data=filtered.to_csv(index=False).encode("utf-8"), file_name="aml_review_queue.csv", mime="text/csv")

if selected_variant is not None:
    accounts_tab, patterns_tab = st.tabs(["Account reference", "Laundering typologies"])
    with accounts_tab:
        st.subheader(f"{selected_variant.name} account reference")
        account_path = selected_variant.accounts_path
        if account_path and account_path.is_file():
            account_sample = cached_account_sample(str(account_path), min(preview_rows, 25_000), account_path.stat().st_size, account_path.stat().st_mtime_ns)
            entity_count = account_sample["Entity Name"].nunique()
            bank_count = account_sample["Bank ID"].nunique()
            a1, a2, a3 = st.columns(3)
            a1.metric("Sampled accounts", f"{len(account_sample):,}")
            a2.metric("Banks in sample", f"{bank_count:,}")
            a3.metric("Entity names in sample", f"{entity_count:,}")
            st.caption("Reference-file preview sampled across the file; counts shown here describe this preview, not a full-file aggregate.")
            st.dataframe(account_sample, use_container_width=True, hide_index=True)
        else:
            st.info("No companion accounts file was found for this variant.")
    with patterns_tab:
        st.subheader(f"{selected_variant.name} laundering typologies")
        patterns_path = selected_variant.patterns_path
        if patterns_path and patterns_path.is_file():
            summary = cached_pattern_summary(str(patterns_path), patterns_path.stat().st_mtime_ns)
            p1, p2 = st.columns(2)
            p1.metric("Typology blocks", f"{summary['pattern_blocks']:,}")
            p2.metric("Listed pattern transactions", f"{summary['listed_transactions']:,}")
            categories = pd.DataFrame(summary["categories"].items(), columns=["Typology", "Pattern blocks"]).sort_values("Pattern blocks", ascending=False)
            if not categories.empty:
                fig = px.bar(categories, x="Pattern blocks", y="Typology", orientation="h", color_discrete_sequence=["#ff667d"])
                fig.update_layout(height=320, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#91a2b5")
                fig.update_xaxes(gridcolor="#223244"); fig.update_yaxes(showgrid=False)
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(pd.DataFrame(summary["examples"]), use_container_width=True, hide_index=True)
        else:
            st.info("No companion patterns file was found for this variant.")

st.divider()
st.subheader("Quick risk assessment")
st.caption("Score an individual transaction. A flagged result is a review signal, not a determination of wrongdoing.")
with st.form("quick_score"):
    a, b, c, d = st.columns(4)
    amount_paid = a.number_input("Amount paid", min_value=0.0, value=12000.0, step=500.0)
    amount_received = b.number_input("Amount received", min_value=0.0, value=11800.0, step=500.0)
    from_bank = c.text_input("From bank", value="BANK-001")
    to_bank = d.text_input("To bank", value="BANK-014")
    submit = st.form_submit_button("Assess transaction", type="primary")
if submit:
    sample = pd.DataFrame([{"amount_paid":amount_paid,"amount_received":amount_received,"from_bank":from_bank,"to_bank":to_bank}])
    assessed = score_transactions(sample).iloc[0]
    st.metric("Assessment", assessed.risk_level, f"Risk score {assessed.risk_score}/100")
    st.caption(f"Method: {assessed.scoring_method}")

st.caption("For analyst support only. Investigate alerts using applicable institutional procedures and independently verified information.")
