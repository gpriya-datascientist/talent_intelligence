"""
app.py  —  ContextLens · Streamlit Dashboard
Uses Groq API (groq.com) — free and fast.
Run with:  streamlit run app.py
"""
from __future__ import annotations
import os, json
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from src.groq_client import analyze_context
from src.visualizer import (
    cosine_gauge,
    hidden_state_comparison,
    attention_heatmap,
    layer_progression_chart,
    cosine_similarity_bar,
)
from src.examples import EXAMPLES

# ── page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="ContextLens",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── dark theme CSS ─────────────────────────────────────────────
st.markdown("""
<style>
body, .stApp { background-color: #0d1117; color: #c9d1d9; }
.stTabs [data-baseweb="tab"] { color: #8b949e; }
.stTabs [aria-selected="true"] { color: #00E5CC !important; border-bottom: 2px solid #00E5CC; }
div[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 16px;
}
.stTextInput input, .stTextArea textarea {
    background: #161b22 !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
}
.focus-word-badge {
    display: inline-block;
    background: #1a2a1a;
    border: 1px solid #00E5CC;
    color: #00E5CC;
    border-radius: 20px;
    padding: 4px 16px;
    font-family: monospace;
    font-size: 18px;
    font-weight: bold;
    margin: 8px 0;
}
.domain-tag {
    display: inline-block;
    background: #1c1400;
    border: 1px solid #FFD700;
    color: #FFD700;
    border-radius: 4px;
    padding: 2px 10px;
    font-size: 13px;
    margin: 4px 2px;
}
.layer-card {
    background: #161b22;
    border-left: 3px solid #9B59FF;
    padding: 8px 12px;
    border-radius: 0 6px 6px 0;
    margin: 6px 0;
    font-size: 13px;
    color: #c9d1d9;
}
.meaning-box {
    background: #0d2b1a;
    border: 1px solid #1D9E75;
    border-radius: 8px;
    padding: 12px 16px;
    font-style: italic;
    color: #7ee8a2;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ── session state ──────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state["result"] = None
if "history" not in st.session_state:
    st.session_state["history"] = []


# ── sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 ContextLens")
    st.markdown("*How the same word gets a different vector in different contexts*")
    st.markdown("---")

    # API key
    api_key_input = st.text_input(
        "Groq API Key (gsk_...)",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Get your free key at console.groq.com",
    )
    if api_key_input:
        os.environ["GROQ_API_KEY"] = api_key_input

    model_choice = st.selectbox("Groq model",
        ["llama-3.3-70b-versatile","llama-3.1-8b-instant","mixtral-8x7b-32768","gemma2-9b-it"],
        index=0)
    os.environ["GROQ_MODEL"] = model_choice

    st.markdown("---")
    st.markdown("### 📚 Example pairs")
    example_labels = ["— custom input —"] + [e["label"] for e in EXAMPLES]
    chosen = st.selectbox("Load example", example_labels)

    st.markdown("---")
    st.markdown("### ℹ️ How it works")
    st.markdown("""
1. Enter two sentences sharing a focus word
2. Grok analyzes contextual embeddings
3. Dashboard shows:
   - Per-layer meaning shift
   - Attention heatmap
   - Hidden state vectors
   - Cosine similarity & shift score
    """)

# ── main ───────────────────────────────────────────────────────
st.markdown("# 🔬 ContextLens")
st.markdown("**Full Transformer Architecture** — how the same word gets a different vector in different contexts")
st.markdown("---")

# ── input section ──────────────────────────────────────────────
col_a, col_focus, col_b = st.columns([5, 2, 5])

# Pre-fill from example
ex = next((e for e in EXAMPLES if e["label"] == chosen), None)
default_a     = ex["sentence_a"]   if ex else "The student showed rapid learning after tutoring"
default_b     = ex["sentence_b"]   if ex else "The learning rate controls model updates"
default_focus = ex["focus_word"]   if ex else "learning"

with col_a:
    st.markdown("#### Sentence A")
    sentence_a = st.text_area("", value=default_a, height=100, key="sa",
                              placeholder="Enter first sentence…")

with col_focus:
    st.markdown("#### Focus word")
    focus_word = st.text_input("", value=default_focus, key="fw",
                               placeholder="word")
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

with col_b:
    st.markdown("#### Sentence B")
    sentence_b = st.text_area("", value=default_b, height=100, key="sb",
                              placeholder="Enter second sentence…")


# ── run analysis ───────────────────────────────────────────────
if analyze_btn:
    if not os.getenv("GROQ_API_KEY","").startswith("gsk_"):
        st.error("⚠️ Please enter your Groq API key (starts with gsk_) in the sidebar.")
    elif not focus_word.strip():
        st.warning("Please enter a focus word.")
    elif focus_word.lower() not in sentence_a.lower() or focus_word.lower() not in sentence_b.lower():
        st.warning(f"Focus word **'{focus_word}'** must appear in both sentences.")
    else:
        with st.spinner(f"Groq is analyzing '{focus_word}' in both contexts…"):
            try:
                result = analyze_context(sentence_a, sentence_b, focus_word, model=model_choice)
                st.session_state["result"] = result
                st.session_state["history"].insert(0, {
                    "focus": focus_word,
                    "a": sentence_a[:40] + "…",
                    "b": sentence_b[:40] + "…",
                    "shift": result["embedding_comparison"]["shift_score"],
                    "cosine": result["embedding_comparison"]["cosine_similarity"],
                })
            except Exception as e:
                st.error(f"API error: {e}")

# ── results ────────────────────────────────────────────────────
result = st.session_state.get("result")

if result:
    comp   = result["embedding_comparison"]
    sa_res = result["sentence_a"]
    sb_res = result["sentence_b"]
    cosine = comp["cosine_similarity"]
    shift  = comp["shift_score"]

    st.markdown("---")

    # ── top metrics row ────────────────────────────────────────
    st.markdown(f"### Focus word: <span class='focus-word-badge'>{result['focus_word']}</span>", unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Cosine Similarity",  f"{cosine:.4f}")
    m2.metric("Shift Score",        f"{shift:.4f}", delta=f"{'HIGH' if shift > 0.5 else 'LOW'} divergence")
    m3.metric("Domain A",           sa_res["domain"])
    m4.metric("Domain B",           sb_res["domain"])
    m5.metric("Divergence starts",  comp.get("divergence_starts_at_layer", "—").replace("_", " "))

    st.markdown("---")

    # ── tabs ───────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏗️ Architecture Flow",
        "📊 Vectors & Similarity",
        "🔥 Attention Heatmaps",
        "🧬 Layer Analysis",
        "📜 Raw JSON",
    ])


    # ── Tab 1: Architecture Flow ───────────────────────────────
    with tab1:
        st.markdown("#### Transformer pipeline for both sentences")

        # INPUT SENTENCE
        st.markdown(f"""
<div style='background:#0d2b1a;border:2px solid #00E5CC;border-radius:8px;padding:14px;text-align:center;margin-bottom:8px'>
  <span style='color:#00E5CC;font-weight:bold;letter-spacing:2px'>INPUT SENTENCES</span><br>
  <span style='color:#888;font-size:13px'>A: "{sa_res['text']}"</span><br>
  <span style='color:#888;font-size:13px'>B: "{sb_res['text']}"</span>
</div>""", unsafe_allow_html=True)

        # TOKENIZER
        st.markdown(f"""
<div style='background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;text-align:center;margin-bottom:8px'>
  <span style='color:#9B59FF;font-weight:bold'>⊕ TOKENIZER</span><br>
  <span style='color:#888;font-size:12px'>A: {" ".join(sa_res["tokens"])}</span><br>
  <span style='color:#888;font-size:12px'>B: {" ".join(sb_res["tokens"])}</span>
</div>""", unsafe_allow_html=True)

        # EMBEDDING LAYER
        st.markdown(f"""
<div style='background:#1a1a2e;border:1px solid #30363d;border-radius:8px;padding:12px;text-align:center;margin-bottom:8px'>
  <span style='color:#9B59FF;font-weight:bold'>⊕ EMBEDDING LAYER (context-free)</span><br>
  <span style='color:#888;font-size:12px'>"{focus_word}" token → same vector in BOTH sentences at this stage</span><br>
  <span style='color:#FFD700;font-size:12px'>+ Positional Encoding added (word order information)</span>
</div>""", unsafe_allow_html=True)

        # ATTENTION LAYERS
        c1, c2 = st.columns(2)
        layer_colors = [
            ("#003366", "#378ADD", "ATTENTION LAYERS 1–5", "Syntax forming · Slight context shift begins"),
            ("#1a0033", "#9B59FF", "LAYERS 6–8", f"Semantic: '{'+'.join(comp['key_differentiating_words_a'][:2])}' vs '{'+'.join(comp['key_differentiating_words_b'][:2])}' pulling meaning apart"),
            ("#1a0a00", "#FF6B35", "LAYERS 9–11", "Domain encoded · Divergence linearly decodable by probing classifier"),
            ("#0d1a00", "#00E5CC", "LAYER 12 (FINAL)", f"HIDDEN STATE — DIFFERENT for each sentence"),
        ]
        for bg, border, title, desc in layer_colors:
            st.markdown(f"""
<div style='background:{bg};border:1px solid {border};border-radius:6px;padding:10px 14px;margin:4px 0'>
  <span style='color:{border};font-weight:bold'>⊕ {title}</span>
  <span style='color:#888;font-size:12px;margin-left:12px'>{desc}</span>
</div>""", unsafe_allow_html=True)

        # FINAL OUTPUTS
        col_out_a, col_out_b = st.columns(2)
        with col_out_a:
            st.markdown(f"""
<div style='background:#0d1a00;border:2px solid #FFD700;border-radius:8px;padding:12px;text-align:center'>
  <span style='color:#FFD700;font-weight:bold'>⊕ OUTPUT — {sa_res['domain']} context</span><br>
  <span style='color:#888;font-size:13px;font-style:italic'>"{sa_res['meaning']}"</span><br>
  <span style='color:#FFD700;font-family:monospace;font-size:11px'>{sa_res['hidden_state_mock'][:4]}</span>
</div>""", unsafe_allow_html=True)

        with col_out_b:
            st.markdown(f"""
<div style='background:#001a2e;border:2px solid #00E5CC;border-radius:8px;padding:12px;text-align:center'>
  <span style='color:#00E5CC;font-weight:bold'>⊕ OUTPUT — {sb_res['domain']} context</span><br>
  <span style='color:#888;font-size:13px;font-style:italic'>"{sb_res['meaning']}"</span><br>
  <span style='color:#00E5CC;font-family:monospace;font-size:11px'>{sb_res['hidden_state_mock'][:4]}</span>
</div>""", unsafe_allow_html=True)

        # CONTEXTLENS MEASURES
        st.markdown(f"""
<div style='background:#0d2b1a;border:2px solid #00E5CC;border-radius:8px;padding:12px;text-align:center;margin-top:8px'>
  <span style='color:#00E5CC;font-weight:bold;letter-spacing:2px'>⊕ CONTEXTLENS MEASURES HERE</span><br>
  <span style='color:#888;font-size:12px'>Cosine similarity on output vectors · Shift score = 1 − cosine · Heatmap visualization</span>
</div>""", unsafe_allow_html=True)

        # Fun fact
        if result.get("fun_fact"):
            st.info(f"💡 **Fun fact:** {result['fun_fact']}")


    # ── Tab 2: Vectors & Similarity ────────────────────────────
    with tab2:
        col_g, col_bar = st.columns([1, 2])
        with col_g:
            st.plotly_chart(cosine_gauge(shift), use_container_width=True)
        with col_bar:
            st.plotly_chart(cosine_similarity_bar(cosine, shift), use_container_width=True)
            st.markdown(f"""
<div style='background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;margin-top:8px;font-size:13px'>
  {comp.get('explanation', '')}
</div>""", unsafe_allow_html=True)

        st.plotly_chart(
            hidden_state_comparison(
                sa_res["hidden_state_mock"], sb_res["hidden_state_mock"],
                sa_res["domain"], sb_res["domain"]
            ),
            use_container_width=True
        )

        col_ctx_a, col_ctx_b = st.columns(2)
        with col_ctx_a:
            st.markdown("**Key context words in Sentence A**")
            for w in comp.get("key_differentiating_words_a", []):
                st.markdown(f"<span class='domain-tag'>{w}</span>", unsafe_allow_html=True)
        with col_ctx_b:
            st.markdown("**Key context words in Sentence B**")
            for w in comp.get("key_differentiating_words_b", []):
                st.markdown(f"<span class='domain-tag'>{w}</span>", unsafe_allow_html=True)

    # ── Tab 3: Attention Heatmaps ──────────────────────────────
    with tab3:
        hm_a = comp.get("attention_heatmap_a", [[]])[0]
        hm_b = comp.get("attention_heatmap_b", [[]])[0]
        tokens_a = sa_res.get("tokens", [])
        tokens_b = sb_res.get("tokens", [])
        idx_a    = sa_res.get("focus_token_index", 0)
        idx_b    = sb_res.get("focus_token_index", 0)

        if hm_a and tokens_a:
            st.plotly_chart(
                attention_heatmap(hm_a, tokens_a,
                                  f"Attention weights — Sentence A ({sa_res['domain']})", idx_a),
                use_container_width=True
            )
        if hm_b and tokens_b:
            st.plotly_chart(
                attention_heatmap(hm_b, tokens_b,
                                  f"Attention weights — Sentence B ({sb_res['domain']})", idx_b),
                use_container_width=True
            )

        st.caption("🟠 Orange box = focus token position. Color intensity = attention weight.")

    # ── Tab 4: Layer Analysis ──────────────────────────────────
    with tab4:
        st.plotly_chart(
            layer_progression_chart(sa_res["layer_analysis"], sb_res["layer_analysis"]),
            use_container_width=True
        )

        col_la, col_lb = st.columns(2)
        layer_keys = [
            ("layers_1_5",  "Layers 1–5  (Syntax)"),
            ("layers_6_8",  "Layers 6–8  (Semantic)"),
            ("layers_9_11", "Layers 9–11  (Domain)"),
            ("layer_12",    "Layer 12  (Final)"),
        ]
        with col_la:
            st.markdown(f"#### Sentence A — *{sa_res['domain']}*")
            for key, label in layer_keys:
                text = sa_res["layer_analysis"].get(key, "—")
                st.markdown(f"<div class='layer-card'><b>{label}</b><br>{text}</div>", unsafe_allow_html=True)
        with col_lb:
            st.markdown(f"#### Sentence B — *{sb_res['domain']}*")
            for key, label in layer_keys:
                text = sb_res["layer_analysis"].get(key, "—")
                st.markdown(f"<div class='layer-card'><b>{label}</b><br>{text}</div>", unsafe_allow_html=True)

    # ── Tab 5: Raw JSON ────────────────────────────────────────
    with tab5:
        st.code(json.dumps(result, indent=2), language="json")


# ── empty state ────────────────────────────────────────────────
else:
    st.markdown("---")
    st.markdown("#### 👈 Enter two sentences and a focus word, then click **Analyze**")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🔍 What it shows**")
        st.markdown("How transformer layers progressively shift a word's meaning based on surrounding context — from syntax (layers 1–5) to final semantic representation (layer 12).")
    with col2:
        st.markdown("**📊 Metrics**")
        st.markdown("Cosine similarity between the two output vectors. Shift score = 1 − cosine. Values near 0 = same meaning. Values near 1 = completely different meaning.")
    with col3:
        st.markdown("**💡 Try these examples**")
        for e in EXAMPLES[:4]:
            st.markdown(f"- **{e['focus_word']}**: {e['label']}")

# ── history sidebar ────────────────────────────────────────────
history = st.session_state.get("history", [])
if history:
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🕓 Analysis history")
        for h in history[:5]:
            st.markdown(
                f"**{h['focus']}** · shift `{h['shift']:.3f}`\n"
                f"A: *{h['a']}*\n"
                f"B: *{h['b']}*"
            )
            st.markdown("---")
