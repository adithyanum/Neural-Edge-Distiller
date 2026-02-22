import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from mlx_lm import load, generate
import time
import html
from scripts.utils import has_structure, format_prompt, clean_response, get_sampler

st.set_page_config(layout="wide", page_title="Neural Edge Distiller", page_icon="⚡")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg:        #080b10;
    --bg2:       #0d1117;
    --bg3:       #111820;
    --border:    #1a2236;
    --border2:   #243049;
    --text:      #c9d4e8;
    --text-dim:  #4a5878;
    --accent:    #3b82f6;
    --accent-dim:#1e3a5f;
    --green:     #22c55e;
    --green-dim: #0f3320;
    --green-b:   #166534;
    --red:       #ef4444;
    --amber:     #f59e0b;
    --mono:      'JetBrains Mono', monospace;
    --display:   'Syne', sans-serif;
}

* { box-sizing: border-box; }
.stApp { background-color: var(--bg); color: var(--text); font-family: var(--mono); }

.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
}

.ned-header { padding: 2rem 0 1.5rem; border-bottom: 1px solid var(--border); margin-bottom: 2rem; animation: fadeDown 0.6s ease both; }
.ned-title { font-family: var(--display); font-size: 1.75rem; font-weight: 800; color: #f0f4ff; letter-spacing: -0.03em; line-height: 1; margin: 0; }
.ned-title span { color: var(--accent); }
.ned-subtitle { font-size: 0.72rem; color: var(--text-dim); font-weight: 300; letter-spacing: 0.04em; text-transform: uppercase; margin-top: 0.4rem; }
.input-label { font-size: 0.68rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem; }

.col-header { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 1rem; padding-bottom: 0.8rem; border-bottom: 1px solid var(--border); }
.col-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--text-dim); flex-shrink: 0; }
.col-dot-green { width: 8px; height: 8px; border-radius: 50%; background: var(--green); flex-shrink: 0; box-shadow: 0 0 6px var(--green); animation: pulse-dot 2s ease-in-out infinite; }
@keyframes pulse-dot { 0%, 100% { opacity: 1; box-shadow: 0 0 6px var(--green); } 50% { opacity: 0.6; box-shadow: 0 0 14px var(--green); } }
.col-model-name { font-family: var(--mono); font-size: 0.78rem; font-weight: 500; color: var(--text); }
.col-model-tag { font-size: 0.65rem; color: var(--text-dim); margin-left: auto; text-transform: uppercase; letter-spacing: 0.05em; }
.col-model-tag-green { font-size: 0.65rem; color: var(--green); margin-left: auto; text-transform: uppercase; letter-spacing: 0.05em; }

.response-wrap { position: relative; border-radius: 6px; overflow: hidden; }
.response-box { background: var(--bg2); border: 1px solid var(--border); border-radius: 6px; padding: 1.2rem 1.4rem; font-family: var(--mono); font-size: 0.78rem; line-height: 1.8; color: var(--text); white-space: pre-wrap; word-break: break-word; min-height: 320px; max-height: 560px; overflow-y: auto; scrollbar-width: thin; scrollbar-color: var(--border2) transparent; }
.response-box-finetuned { background: #090f0c; border: 1px solid var(--green-b); border-radius: 6px; padding: 1.2rem 1.4rem; font-family: var(--mono); font-size: 0.78rem; line-height: 1.8; color: #a7f3c8; white-space: pre-wrap; word-break: break-word; min-height: 320px; max-height: 560px; overflow-y: auto; scrollbar-width: thin; scrollbar-color: var(--green-b) transparent; }
.response-box-finetuned::after { content: ''; position: absolute; top: -40%; left: 0; right: 0; height: 40%; background: linear-gradient(to bottom, transparent, rgba(34,197,94,0.04), transparent); animation: scan 1.8s ease-out both; pointer-events: none; }
@keyframes scan { 0% { top: -40%; } 100% { top: 110%; } }

.metrics-row { display: flex; align-items: center; gap: 1rem; margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid var(--border); }
.metric-item { display: flex; flex-direction: column; gap: 1px; }
.metric-value { font-family: var(--mono); font-size: 0.85rem; font-weight: 600; color: var(--text); }
.metric-value-green { font-family: var(--mono); font-size: 0.85rem; font-weight: 600; color: var(--green); }
.metric-label { font-size: 0.6rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.06em; }
.struct-pass { margin-left: auto; font-size: 0.7rem; color: var(--green); font-family: var(--mono); background: var(--green-dim); border: 1px solid var(--green-b); padding: 2px 10px; border-radius: 3px; }
.struct-fail { margin-left: auto; font-size: 0.7rem; color: var(--red); font-family: var(--mono); background: #1a0808; border: 1px solid #7f1d1d; padding: 2px 10px; border-radius: 3px; }

.comparison-bar { background: var(--bg3); border: 1px solid var(--border); border-radius: 6px; padding: 1rem 1.4rem; display: flex; align-items: center; gap: 2rem; margin-top: 1.5rem; animation: fadeUp 0.5s ease both; }
.cbar-label { font-size: 0.65rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.08em; white-space: nowrap; }
.cbar-delta { font-family: var(--display); font-size: 1.1rem; font-weight: 700; }
.cbar-delta-pos { color: var(--green); }
.cbar-delta-neg { color: var(--red); }
.cbar-divider { width: 1px; height: 28px; background: var(--border); flex-shrink: 0; }

@keyframes fadeDown { from { opacity: 0; transform: translateY(-10px); } to { opacity: 1; transform: translateY(0); } }
@keyframes fadeUp   { from { opacity: 0; transform: translateY(8px);  } to { opacity: 1; transform: translateY(0); } }

.stTextArea textarea { background: var(--bg2) !important; color: var(--text) !important; border: 1px solid var(--border2) !important; border-radius: 6px !important; font-family: var(--mono) !important; font-size: 0.82rem !important; }
.stTextArea textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 1px var(--accent-dim) !important; }
.stButton > button { background: var(--accent) !important; color: #fff !important; border: none !important; font-family: var(--mono) !important; font-size: 0.8rem !important; font-weight: 500 !important; padding: 0.5rem 1.4rem !important; border-radius: 4px !important; transition: opacity 0.15s !important; }
.stButton > button:hover { opacity: 0.85 !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }
#MainMenu, footer, header, .stDeployButton { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all_models():
    v_model, v_tok = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
    n_model, n_tok = load("models/neural-edge-3b/")
    return (v_model, v_tok), (n_model, n_tok)


st.markdown("""
<div class="ned-header">
    <p class="ned-title">Neural <span>Edge</span></p>
    <p class="ned-subtitle">CoT Distillation Benchmark · Llama 3.2 3B · Apple Silicon</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Hydrating models into unified memory..."):
    (v_model, v_tok), (n_model, n_tok) = load_all_models()

st.markdown('<div class="input-label">System Design Scenario</div>', unsafe_allow_html=True)
scenario = st.text_area("scenario", placeholder="e.g. Kafka consumer lag spiking to 48 hours under peak load with strict per-user event ordering. Recover throughput without violating ordering constraints.", height=85, label_visibility="collapsed")
run = st.button("Run Dual Inference →", type="primary")
st.markdown("<hr style='border:none;border-top:1px solid #1a2236;margin:1.5rem 0'>", unsafe_allow_html=True)

if run and scenario.strip():
    sampler = get_sampler()
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="col-header"><div class="col-dot"></div><span class="col-model-name">Llama-3.2-3B-Instruct</span><span class="col-model-tag">Vanilla Baseline</span></div>', unsafe_allow_html=True)
        with st.spinner("Generating..."):
            t0 = time.time()
            v_res = generate(v_model, v_tok, prompt=format_prompt(v_tok, scenario), max_tokens=500, sampler=sampler)
            v_elapsed = time.time() - t0
            v_res = clean_response(v_res)
        v_tps = round(len(v_res.split()) / v_elapsed, 1)
        v_struct = has_structure(v_res)
        st.markdown(f'<div class="response-wrap"><div class="response-box">{html.escape(v_res)}</div></div>', unsafe_allow_html=True)
        sl = '<span class="struct-pass">✓ structured</span>' if v_struct else '<span class="struct-fail">✗ unstructured</span>'
        st.markdown(f'<div class="metrics-row"><div class="metric-item"><span class="metric-value">{v_elapsed:.2f}s</span><span class="metric-label">Latency</span></div><div class="metric-item"><span class="metric-value">{v_tps}</span><span class="metric-label">Tok / sec</span></div>{sl}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="col-header"><div class="col-dot-green"></div><span class="col-model-name">Neural Edge 3B</span><span class="col-model-tag-green">LoRA Distilled</span></div>', unsafe_allow_html=True)
        with st.spinner("Generating..."):
            t0 = time.time()
            n_res = generate(n_model, n_tok, prompt=format_prompt(n_tok, scenario), max_tokens=500, sampler=sampler)
            n_elapsed = time.time() - t0
            n_res = clean_response(n_res)
        n_tps = round(len(n_res.split()) / n_elapsed, 1)
        n_struct = has_structure(n_res)
        st.markdown(f'<div class="response-wrap"><div class="response-box-finetuned">{html.escape(n_res)}</div></div>', unsafe_allow_html=True)
        sl = '<span class="struct-pass">✓ structured</span>' if n_struct else '<span class="struct-fail">✗ unstructured</span>'
        st.markdown(f'<div class="metrics-row"><div class="metric-item"><span class="metric-value-green">{n_elapsed:.2f}s</span><span class="metric-label">Latency</span></div><div class="metric-item"><span class="metric-value-green">{n_tps}</span><span class="metric-label">Tok / sec</span></div>{sl}</div>', unsafe_allow_html=True)

    tps_d = round(n_tps - v_tps, 1)
    lat_d = round(n_elapsed - v_elapsed, 2)
    tps_cls = "cbar-delta-pos" if tps_d >= 0 else "cbar-delta-neg"
    lat_cls = "cbar-delta-neg" if lat_d >= 0 else "cbar-delta-pos"
    tps_sign = "+" if tps_d >= 0 else ""
    lat_sign = "+" if lat_d >= 0 else ""
    fmt = ("✓ Both structured" if v_struct and n_struct else "⚡ Neural Edge only" if n_struct else "↔ Neither")
    fmt_color = "var(--green)" if n_struct else "var(--amber)"

    st.markdown(f"""
    <div class="comparison-bar">
        <div class="metric-item"><span class="cbar-label">Δ Throughput</span><span class="cbar-delta {tps_cls}">{tps_sign}{tps_d} tok/s</span></div>
        <div class="cbar-divider"></div>
        <div class="metric-item"><span class="cbar-label">Δ Latency</span><span class="cbar-delta {lat_cls}">{lat_sign}{lat_d}s</span></div>
        <div class="cbar-divider"></div>
        <div class="metric-item"><span class="cbar-label">Format Quality</span><span class="cbar-delta" style="color:{fmt_color};font-size:0.9rem">{fmt}</span></div>
        <div class="cbar-divider"></div>
        <div class="metric-item"><span class="cbar-label">Memory</span><span class="cbar-delta" style="color:var(--text-dim);font-size:0.9rem">Identical</span></div>
    </div>
    """, unsafe_allow_html=True)

elif run and not scenario.strip():
    st.warning("Enter a scenario to run inference.")