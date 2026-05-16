"""
app_fixed.py  —  KPI Trend Analysis Dashboard
No sentence-transformers. No chromadb. No ONNX.
Uses TF-IDF for vector search + Groq for AI narrative.
Run: streamlit run app_fixed.py
"""
from __future__ import annotations
import io, os, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
warnings.filterwarnings("ignore")
load_dotenv()

st.set_page_config(page_title="KPI Trend Analysis · RAG", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

for k, v in {"df": None, "store": None, "narratives": {}, "forecasts": {}}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── data helpers ────────────────────────────────────────────────
def _load_file(uploaded):
    ext = Path(uploaded.name).suffix.lower()
    raw = uploaded.read()
    if ext in (".csv", ".tsv"):
        sample = raw[:2048].decode("utf-8", errors="replace")
        delim  = max([",", ";", "\t"], key=lambda d: sample.count(d))
        return pd.read_csv(io.BytesIO(raw), delimiter=delim)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(io.BytesIO(raw), engine="openpyxl")
    raise ValueError(f"Unsupported file type: {ext}")

def _normalise(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    def _find(aliases):
        for c in df.columns:
            if any(a in c for a in aliases): return c
        return None
    date_col = _find(["date","period","month","week","time","timestamp"])
    kpi_col  = _find(["kpi_name","kpi","metric","indicator","measure","name"])
    val_col  = _find(["value","actual","amount","val","result","score"])
    dim_col  = _find(["dimension","dim","category","segment","region","department"])
    unit_col = _find(["unit"])
    if not date_col: raise ValueError("❌ Cannot find date column. Name it 'date'.")
    if not val_col:  raise ValueError("❌ Cannot find value column. Name it 'value'.")
    out = pd.DataFrame()
    out["date"]      = pd.to_datetime(df[date_col], errors="coerce")
    out["kpi_name"]  = df[kpi_col].astype(str).str.strip() if kpi_col else "KPI"
    out["value"]     = pd.to_numeric(df[val_col], errors="coerce")
    out["dimension"] = df[dim_col].astype(str).str.strip() if dim_col else "Overall"
    out["unit"]      = df[unit_col].astype(str).str.strip() if unit_col else ""
    return out.dropna(subset=["date","value"]).sort_values(
        ["kpi_name","dimension","date"]).reset_index(drop=True)

def _enrich(df):
    df = df.copy()
    df["pop_pct"]       = df.groupby(["kpi_name","dimension"])["value"].pct_change()*100
    df["roll_mean_12p"] = df.groupby(["kpi_name","dimension"])["value"].transform(
        lambda s: s.rolling(12, min_periods=1).mean())
    df["yoy_pct"]       = df.groupby(["kpi_name","dimension"])["value"].pct_change(52)*100
    def _flag(s):
        q1,q3 = s.quantile(.25), s.quantile(.75); iqr = q3-q1
        return (s < q1-1.5*iqr)|(s > q3+1.5*iqr)
    df["anomaly"] = df.groupby(["kpi_name","dimension"])["value"].transform(_flag)
    return df

def _get_sample():
    rng   = np.random.default_rng(42)
    dates = pd.date_range("2022-01-03", periods=104, freq="W")
    kpis  = {"Monthly Revenue":(500000,10000,"USD"),
             "Customer Churn Rate":(5.2,.3,"%"),
             "NPS Score":(42,3,"pts"),
             "Support Tickets":(320,30,"count"),
             "Gross Margin":(68,2,"%")}
    rows  = []
    for kpi,(base,noise,unit) in kpis.items():
        for dim in ["EMEA","APAC","Americas"]:
            trend  = np.linspace(0,base*.15,104)
            season = base*.05*np.sin(np.linspace(0,4*np.pi,104))
            vals   = base+trend+season+rng.normal(0,noise,104)
            for d,v in zip(dates,vals):
                rows.append({"date":d,"kpi_name":kpi,"dimension":dim,
                             "value":round(float(v),2),"unit":unit})
    return _enrich(pd.DataFrame(rows))

# ── TF-IDF vector store (no external deps) ──────────────────────
def _build_store(df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs, metas = [], []
    for r in df.to_dict(orient="records"):
        d  = r.get("date")
        ds = d.isoformat()[:10] if hasattr(d,"isoformat") else str(d)[:10]
        pop = f" PoP: {r['pop_pct']:+.1f}%." if pd.notna(r.get("pop_pct")) else ""
        an  = " Anomaly detected." if r.get("anomaly") else ""
        docs.append(f"On {ds}, {r['kpi_name']} ({r['dimension']}) "
                    f"= {r['value']:.2f} {r.get('unit','')}.{pop}{an}")
        metas.append({"kpi_name":str(r["kpi_name"]),
                      "dimension":str(r["dimension"]), "date_str":ds})
    with st.spinner("Building search index…"):
        vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        mat = vec.fit_transform(docs)
    return {"vectorizer": vec, "matrix": mat, "docs": docs, "metas": metas}

def _query_store(store, question, kpi, dim, n=8):
    from sklearn.metrics.pairwise import cosine_similarity
    q_vec  = store["vectorizer"].transform([question])
    scores = cosine_similarity(q_vec, store["matrix"])[0]
    out    = []
    for idx in np.argsort(scores)[::-1]:
        m = store["metas"][idx]
        if m["kpi_name"] != kpi: continue
        if dim != "Overall" and m["dimension"] != dim: continue
        out.append({"document": store["docs"][idx],
                    "distance": float(1 - scores[idx])})
        if len(out) >= n: break
    return out

# ── Groq narrative ──────────────────────────────────────────────
def _groq_narrative(question, kpi, dim, docs):
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY",""))
    ctx    = "\n".join(f"[{i+1}] {d['document']}" for i,d in enumerate(docs))
    prompt = f"""KPI: {kpi}\nDimension: {dim}\nQuestion: {question}

--- DATA CONTEXT ---
{ctx}
--- END ---

NARRATIVE:
<2-3 paragraph analysis>

KEY INSIGHTS:
1. <insight>
2. <insight>
3. <insight>

TREND SIGNAL: <UP|DOWN|STABLE|VOLATILE|UNKNOWN>
CONFIDENCE: <HIGH|MEDIUM|LOW>"""
    resp = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL","llama-3.3-70b-versatile"),
        messages=[{"role":"system","content":"You are a senior BI analyst."},
                  {"role":"user","content":prompt}],
        temperature=0.3, max_tokens=1024)
    return resp.choices[0].message.content, resp.usage.total_tokens if resp.usage else 0

def _parse(text):
    narr = re.search(r"NARRATIVE:\s*(.+?)(?=KEY INSIGHTS:|TREND SIGNAL:|$)", text, re.DOTALL)
    ins  = re.search(r"KEY INSIGHTS:\s*(.+?)(?=TREND SIGNAL:|CONFIDENCE:|$)", text, re.DOTALL)
    sig  = re.search(r"TREND SIGNAL:\s*(UP|DOWN|STABLE|VOLATILE|UNKNOWN)", text)
    conf = re.search(r"CONFIDENCE:\s*(HIGH|MEDIUM|LOW)", text)
    insights = []
    if ins:
        insights = [re.sub(r"^\d+\.\s*","",l).strip()
                    for l in ins.group(1).splitlines()
                    if l.strip() and re.match(r"^\d+\.",l.strip())]
    return (narr.group(1).strip() if narr else text, insights,
            sig.group(1) if sig else "UNKNOWN",
            conf.group(1) if conf else "LOW")

# ── SIDEBAR ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📂 Data Source")
    uploaded   = st.file_uploader("Upload CSV / Excel",
                                  type=["csv","xlsx","xls","tsv"], key="uploader")
    sample_btn = st.button("⚡ Load sample data", use_container_width=True)
    st.markdown("---")
    st.markdown("## ⚙️ Settings")
    fhorizon = st.slider("Forecast horizon (periods)", 4, 52, 12)
    st.markdown("---")
    api_key = st.text_input("Groq API Key (gsk_...)",
                            value=os.getenv("GROQ_API_KEY",""), type="password")
    if api_key: os.environ["GROQ_API_KEY"] = api_key
    groq_model = st.selectbox("Groq model",
                    ["llama-3.3-70b-versatile","llama-3.1-8b-instant",
                     "mixtral-8x7b-32768","gemma2-9b-it"])
    os.environ["GROQ_MODEL"] = groq_model

# ── LOAD DATA ───────────────────────────────────────────────────
if sample_btn:
    df_loaded = _get_sample()
    st.session_state.update({"df": df_loaded,
                              "store": _build_store(df_loaded),
                              "forecasts": {}, "narratives": {}})
    st.rerun()

if uploaded is not None and st.session_state["df"] is None:
    try:
        df_loaded = _enrich(_normalise(_load_file(uploaded)))
        st.session_state.update({"df": df_loaded,
                                  "store": _build_store(df_loaded),
                                  "forecasts": {}, "narratives": {}})
        st.rerun()
    except Exception as e:
        st.error(str(e))

if st.session_state["df"] is not None and uploaded is None:
    with st.sidebar:
        if st.button("🔄 Load new file", use_container_width=True):
            st.session_state.update({"df":None,"store":None})
            st.rerun()

# ── MAIN ────────────────────────────────────────────────────────
st.markdown("# 📈 KPI Trend Analysis with RAG")
df = st.session_state.get("df")

if df is None:
    st.info("👈 Upload a CSV or click **⚡ Load sample data** to begin.")
    st.markdown("""
**Expected CSV columns:**

| Column | Example |
|---|---|
| `date` | 2024-01-07 |
| `kpi_name` | Monthly Revenue |
| `dimension` | EMEA |
| `value` | 512000.50 |
| `unit` | USD *(optional)* |
""")
    st.stop()

st.success(f"✅ {len(df):,} rows · {df['kpi_name'].nunique()} KPIs · "
           f"{df['dimension'].nunique()} dimensions · "
           f"{df['date'].min().date()} → {df['date'].max().date()}")

kpi_names  = sorted(df["kpi_name"].unique())
dimensions = sorted(df["dimension"].unique())
c1,c2 = st.columns([2,1])
selected_kpi = c1.selectbox("Select KPI", kpi_names)
selected_dim = c2.selectbox("Select Dimension", ["All"]+dimensions)

if selected_dim == "All":
    view = df[df["kpi_name"]==selected_kpi].groupby("date",as_index=False)["value"].mean()
    view["kpi_name"] = selected_kpi; view["dimension"] = "All"
    dim_key = "Overall"
else:
    view    = df[(df["kpi_name"]==selected_kpi)&(df["dimension"]==selected_dim)].copy()
    dim_key = selected_dim

latest = view.sort_values("date")["value"].iloc[-1]
prev   = view.sort_values("date")["value"].iloc[-2] if len(view)>1 else latest
delta  = latest-prev; pct = delta/prev*100 if prev else 0
m1,m2,m3,m4 = st.columns(4)
m1.metric("Latest Value", f"{latest:,.2f}", f"{delta:+,.2f} ({pct:+.1f}%)")
m2.metric("Data Points",  f"{len(view):,}")
m3.metric("Min",          f"{view['value'].min():,.2f}")
m4.metric("Max",          f"{view['value'].max():,.2f}")
st.markdown("---")

tab1,tab2,tab3 = st.tabs(["📊 Trend Analysis","🔮 Forecast","🤖 AI Narrative (RAG)"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view["date"],y=view["value"],mode="lines+markers",
                             name="Actual",line=dict(color="#1D9E75",width=2),
                             marker=dict(size=4)))
    if "roll_mean_12p" in view.columns:
        fig.add_trace(go.Scatter(x=view["date"],y=view["roll_mean_12p"],mode="lines",
                                 name="12-Period MA",
                                 line=dict(color="#378ADD",width=1.5,dash="dash")))
    if "anomaly" in view.columns:
        an = view[view["anomaly"]==True]
        if not an.empty:
            fig.add_trace(go.Scatter(x=an["date"],y=an["value"],mode="markers",
                                     name="Anomaly",
                                     marker=dict(color="#D85A30",size=10,symbol="x")))
    fig.update_layout(title=f"{selected_kpi} — {selected_dim}",height=420,
                      hovermode="x unified",margin=dict(l=20,r=20,t=50,b=20))
    st.plotly_chart(fig,use_container_width=True)
    if "pop_pct" in view.columns:
        st.markdown("#### Period-over-period change (%)")
        fig2 = go.Figure(go.Bar(x=view["date"],y=view["pop_pct"],
                                marker_color=view["pop_pct"].apply(
                                    lambda v:"#1D9E75" if v>=0 else "#D85A30")))
        fig2.update_layout(height=200,margin=dict(l=20,r=20,t=10,b=20))
        st.plotly_chart(fig2,use_container_width=True)

with tab2:
    ck = f"{selected_kpi}|{dim_key}"
    if ck not in st.session_state["forecasts"]:
        if st.button("▶ Run Forecast", type="primary"):
            try:
                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                series = view.sort_values("date").set_index("date")["value"]
                model  = ExponentialSmoothing(series, trend="add",
                                              seasonal="add", seasonal_periods=52
                                              if len(series)>=104 else None).fit()
                fitted    = model.fittedvalues
                forecast  = model.forecast(fhorizon)
                resid_std = (series - fitted).std()
                fcast_idx = pd.date_range(series.index[-1], periods=fhorizon+1, freq="W")[1:]
                hist_df   = pd.DataFrame({"date":series.index,"value":series.values,
                                          "predicted":fitted.values})
                fcast_df  = pd.DataFrame({"date":fcast_idx,"predicted":forecast.values,
                                          "upper_ci":forecast.values+1.645*resid_std,
                                          "lower_ci":forecast.values-1.645*resid_std})
                from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
                mape = mean_absolute_percentage_error(series, fitted)*100
                rmse = np.sqrt(mean_squared_error(series, fitted))
                st.session_state["forecasts"][ck] = {
                    "hist":hist_df,"fcast":fcast_df,
                    "mape":mape,"rmse":rmse}
                st.rerun()
            except Exception as e:
                st.error(f"Forecast error: {e}")
    else:
        r = st.session_state["forecasts"][ck]
        mc1,mc2 = st.columns(2)
        mc1.metric("MAPE", f"{r['mape']:.2f}%")
        mc2.metric("RMSE", f"{r['rmse']:.2f}")
        hist=r["hist"]; fcast=r["fcast"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["date"],y=hist["value"],
                                 name="Actual",line=dict(color="#1D9E75",width=2)))
        fig.add_trace(go.Scatter(x=hist["date"],y=hist["predicted"],
                                 name="Fitted",line=dict(color="#378ADD",width=1.5,dash="dot")))
        fig.add_trace(go.Scatter(x=fcast["date"],y=fcast["predicted"],
                                 name="Forecast",line=dict(color="#BA7517",width=2,dash="dash")))
        fig.add_trace(go.Scatter(
            x=pd.concat([fcast["date"],fcast["date"][::-1]]),
            y=pd.concat([fcast["upper_ci"],fcast["lower_ci"][::-1]]),
            fill="toself",fillcolor="rgba(186,117,23,0.15)",
            line=dict(color="rgba(255,255,255,0)"),name="90% CI"))
        fig.update_layout(height=450,hovermode="x unified",
                          margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig,use_container_width=True)

with tab3:
    store = st.session_state.get("store")
    if not store:
        st.warning("Load data first.")
        st.stop()
    presets  = ["What is driving the current trend?",
                "Are there any anomalies I should be aware of?",
                "How does this KPI compare year-over-year?",
                "What is the forecast outlook for the next quarter?"]
    preset   = st.selectbox("Preset questions", ["— custom —"]+presets)
    question = st.text_input("Your question",
                             value="" if preset=="— custom —" else preset)
    if st.button("🤖 Generate AI Narrative", type="primary"):
        if not question.strip():
            st.warning("Enter a question.")
        elif not os.getenv("GROQ_API_KEY","").startswith("gsk_"):
            st.error("Add your Groq API key (starts with gsk_) in the sidebar.")
        else:
            nk = f"{selected_kpi}|{dim_key}|{question}"
            if nk not in st.session_state["narratives"]:
                with st.spinner("Querying Groq…"):
                    try:
                        docs = _query_store(store,question,selected_kpi,dim_key)
                        raw,tokens = _groq_narrative(question,selected_kpi,dim_key,docs)
                        narr,insights,signal,conf = _parse(raw)
                        st.session_state["narratives"][nk] = {
                            "narr":narr,"insights":insights,"signal":signal,
                            "conf":conf,"tokens":tokens,"docs":docs}
                    except Exception as e:
                        st.error(f"Groq error: {e}")
            if nk in st.session_state["narratives"]:
                r  = st.session_state["narratives"][nk]
                sc = {"UP":"🟢","DOWN":"🔴","STABLE":"🔵","VOLATILE":"🟠","UNKNOWN":"⚪"}
                cc = {"HIGH":"🟢","MEDIUM":"🟡","LOW":"🔴"}
                a,b,c = st.columns(3)
                a.metric("Trend Signal",f"{sc.get(r['signal'],'')} {r['signal']}")
                b.metric("Confidence",  f"{cc.get(r['conf'],'')} {r['conf']}")
                c.metric("Tokens used", r["tokens"])
                st.markdown("#### Narrative")
                st.markdown(r["narr"])
                if r["insights"]:
                    st.markdown("#### Key Insights")
                    for i,ins in enumerate(r["insights"],1):
                        st.markdown(f"**{i}.** {ins}")
                with st.expander(f"Retrieved context ({len(r['docs'])} chunks)"):
                    for i,d in enumerate(r["docs"],1):
                        st.markdown(f"**[{i}]** *(dist: {d['distance']:.3f})*  \n{d['document']}")
