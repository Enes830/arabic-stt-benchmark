import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_TRANSCRIPTION_URL = "https://openrouter.ai/api/v1/audio/transcriptions"

# Curated catalog of OpenRouter transcription models
AVAILABLE_MODELS: Dict[str, str] = {
    "elevenlabs/scribe-v2": "ElevenLabs Scribe v2",
    "openai/gpt-4o-transcribe": "OpenAI GPT-4o Transcribe",
    "google/gemini-3.5-transcribe": "Google Gemini 3.5 Transcribe",
    "deepgram/nova-3": "Deepgram Nova-3",
    "openai/whisper-1": "OpenAI Whisper-1 (Prod Baseline)",
    "openai/whisper-large-v3": "OpenAI Whisper Large v3",
    "openai/whisper-large-v3-turbo": "OpenAI Whisper Large v3 Turbo",
    "openai/gpt-transcribe": "OpenAI GPT Transcribe",
    "mistralai/voxtral-small-transcribe-3-0": "Mistral Voxtral Small 3.0",
    "assemblyai/universal-3-6-pro": "AssemblyAI Universal 3.6 Pro",
    "x-ai/grok-stt-2.0": "xAI Grok STT 2.0",
    "qwen/qwen3-asr-1.7b": "Qwen3 ASR 1.7B",
    "google/chirp-3": "Google Chirp 3",
}

# Approximate OpenRouter pricing (USD per audio hour) for reference
MODEL_PRICE_PER_HOUR: Dict[str, str] = {
    "elevenlabs/scribe-v2": "$0.22 / hr ($0.0037/min)",
    "openai/gpt-4o-transcribe": "~$0.36 / hr ($0.0060/min)",
    "google/gemini-3.5-transcribe": "~$0.25 / hr (token-based)",
    "deepgram/nova-3": "$0.26 / hr ($0.0043/min)",
    "openai/whisper-1": "$0.36 / hr ($0.0060/min)",
    "openai/whisper-large-v3": "$0.03 / hr ($0.0005/min)",
    "openai/whisper-large-v3-turbo": "$0.01 / hr ($0.0002/min)",
    "openai/gpt-transcribe": "$0.27 / hr ($0.0045/min)",
    "mistralai/voxtral-small-transcribe-3-0": "$0.24 / hr ($0.0040/min)",
    "assemblyai/universal-3-6-pro": "$0.30 / hr ($0.0050/min)",
    "x-ai/grok-stt-2.0": "$0.10 / hr ($0.0017/min)",
    "qwen/qwen3-asr-1.7b": "$0.03 / hr ($0.0005/min)",
    "google/chirp-3": "$0.96 / hr ($0.0160/min)",
}

DEFAULT_MODELS: List[str] = [
    "elevenlabs/scribe-v2",
    "openai/gpt-4o-transcribe",
    "google/gemini-3.5-transcribe",
    "deepgram/nova-3",
    "openai/whisper-1",
]

AUDIO_SAMPLES_DIR = Path(__file__).resolve().parent.parent / "audio_samples"

PRESET_SAMPLES: Dict[str, Dict[str, str]] = {
    "1_gas_station_fire_1397383171882247.mp3": {
        "label": "1. حريق بنزينة العاشر من رمضان (ضوضاء شديدة + عامية مصرية)",
        "ground_truth": (
            "العربية دي هي السبب صح "
            "اخلوا المكان اخلوا المكان "
            "اخلي يا دكتور "
            "العربيات دي العربيات ده في عربية باظت أهي يلا"
        ),
    },
    "2_german_islam_1736720096996251.mp3": {
        "label": "2. ألماني يشهر إسلامه بالغردقة (صدى صوت + متحدث غير عربي)",
        "ground_truth": (
            "أشهد أشهد أن لا إله إلا الله إلا الله إلا الله إلا الله "
            "وأشهد وأشهد أن محمدا أن محمدا أن محمدا أن محمدا "
            "عبد الله عبد الله ورسوله ورسوله "
            "الله أكبر الله أكبر الله أكبر"
        ),
    },
    "3_gaza_girl_Df7Jkvgs5Pq.mp3": {
        "label": "3. رسالة طفلة غزاوية إلى العالم (صوت طفلة + موسيقى خلفية)",
        "ground_truth": (
            "يا أيها العالم هل تشعرون بنا النزوح مش مجرد انتقال من مكان لمكان "
            "هو هروب من الموت للموت ومن خوف لخوف أكبر ومن وجع لوجع أكبر "
            "ورا كل خيمة عيلة وورا كل عيلة حكاية وورا كل طفل حلم سلب منه "
            "لا تجعلوا معاناتنا ونزوحنا وألمنا مجرد مشهد عابر"
        ),
    },
}


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for fair WER/CER scoring across models."""
    if not text:
        return ""
    # Remove Arabic diacritics (tashkeel) and tatweel
    text = re.sub(r"[\u064B-\u065F\u0670\u0640]", "", text)
    # Normalize Alef, Taa Marbuta, Yaa
    text = re.sub(r"[إأآا]", "ا", text)
    text = text.replace("ة", "ه").replace("ى", "ي")
    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse whitespace
    return re.sub(r"\s+", " ", text).strip().lower()


def levenshtein_distance(seq1: List[str], seq2: List[str]) -> int:
    m, n = len(seq1), len(seq2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return dp[n]


def compute_metrics(reference: str, hypothesis: str) -> Tuple[Optional[float], Optional[float]]:
    """Return (WER %, CER %) after Arabic normalization."""
    ref_norm = normalize_arabic(reference)
    hyp_norm = normalize_arabic(hypothesis)
    if not ref_norm:
        return None, None

    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()
    wer = (levenshtein_distance(ref_words, hyp_words) / max(1, len(ref_words))) * 100.0

    ref_chars = list(ref_norm.replace(" ", ""))
    hyp_chars = list(hyp_norm.replace(" ", ""))
    cer = (levenshtein_distance(ref_chars, hyp_chars) / max(1, len(ref_chars))) * 100.0
    return round(wer, 1), round(cer, 1)


def transcribe_via_openrouter(
    api_key: str,
    model_id: str,
    audio_b64: str,
    audio_format: str,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> Dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://kashida.io",
        "X-Title": "Kashida Arabic STT Benchmark",
    }
    payload: Dict = {
        "model": model_id,
        "input_audio": {
            "data": audio_b64,
            "format": audio_format,
        },
    }
    if language and language != "auto":
        payload["language"] = language
    if prompt and prompt.strip():
        payload["prompt"] = prompt.strip()

    t0 = time.perf_counter()
    try:
        resp = requests.post(
            OPENROUTER_TRANSCRIPTION_URL,
            headers=headers,
            json=payload,
            timeout=120,
        )
        latency = round(time.perf_counter() - t0, 2)
        if resp.status_code != 200:
            return {
                "model_id": model_id,
                "model_name": AVAILABLE_MODELS.get(model_id, model_id),
                "ok": False,
                "latency_s": latency,
                "text": "",
                "error": f"HTTP {resp.status_code}: {resp.text[:300]}",
            }
        data = resp.json()
        text = data.get("text") or data.get("transcript") or ""
        return {
            "model_id": model_id,
            "model_name": AVAILABLE_MODELS.get(model_id, model_id),
            "ok": True,
            "latency_s": latency,
            "text": text.strip(),
            "raw": data,
            "error": None,
        }
    except Exception as exc:
        return {
            "model_id": model_id,
            "model_name": AVAILABLE_MODELS.get(model_id, model_id),
            "ok": False,
            "latency_s": round(time.perf_counter() - t0, 2),
            "text": "",
            "error": str(exc),
        }


# ---------------- Streamlit UI ----------------
st.set_page_config(
    page_title="Kashida Arabic STT Benchmark",
    page_icon="🎙️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .rtl-box {
        direction: rtl;
        text-align: right;
        font-size: 1.05rem;
        line-height: 1.75;
        background-color: rgba(128, 128, 128, 0.08);
        padding: 1rem;
        border-radius: 0.6rem;
        border-right: 4px solid #1283F9;
        min-height: 110px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎙️ Kashida Arabic STT Benchmark")
st.caption("Compare top Speech-to-Text models on noisy & dialectal Arabic audio via OpenRouter")

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "OpenRouter API Key",
        value=os.getenv("OPENROUTER_API_KEY", ""),
        type="password",
        help="Loaded automatically from .env if set.",
    )
    language = st.selectbox(
        "Language Hint",
        options=["ar", "auto"],
        index=0,
        help="'ar' forces Arabic; 'auto' lets each model auto-detect.",
    )
    prompt_hint = st.text_input(
        "Optional Prompt Hint",
        value="",
        placeholder="Leave empty to avoid Whisper prompt hallucination",
    )
    st.divider()
    st.markdown("**Tip:** Leave prompt empty first to test pure acoustic robustness in noise.")

# 1. Audio Input
col_left, col_right = st.columns([1, 1])

audio_bytes: Optional[bytes] = None
audio_format: str = "mp3"
audio_name: str = ""
default_ground_truth: str = ""

with col_left:
    st.subheader("1. Audio Source")
    source_mode = st.radio(
        "Choose input method",
        ["Pre-loaded Prod Samples", "Upload Audio File"],
        horizontal=True,
    )

    if source_mode == "Pre-loaded Prod Samples":
        available_presets = {
            fname: meta
            for fname, meta in PRESET_SAMPLES.items()
            if (AUDIO_SAMPLES_DIR / fname).exists()
        }
        if not available_presets:
            st.warning(f"No sample MP3s found in `{AUDIO_SAMPLES_DIR}`. Switch to Upload.")
        else:
            preset_keys = list(available_presets.keys())
            q_sample = st.query_params.get("sample", "")
            default_idx = next((i for i, k in enumerate(preset_keys) if q_sample and q_sample in k), 0)
            selected_fname = st.selectbox(
                "Select Sample",
                options=preset_keys,
                index=default_idx,
                format_func=lambda k: available_presets[k]["label"],
            )
            sample_path = AUDIO_SAMPLES_DIR / selected_fname
            audio_bytes = sample_path.read_bytes()
            audio_format = sample_path.suffix.lstrip(".").lower() or "mp3"
            audio_name = selected_fname
            default_ground_truth = available_presets[selected_fname]["ground_truth"]
            st.audio(audio_bytes, format=f"audio/{audio_format}")
    else:
        uploaded = st.file_uploader("Upload MP3, WAV, M4A, OGG, or WEBM", type=["mp3", "wav", "m4a", "ogg", "webm"])
        if uploaded is not None:
            audio_bytes = uploaded.read()
            audio_format = Path(uploaded.name).suffix.lstrip(".").lower() or "mp3"
            audio_name = uploaded.name
            st.audio(audio_bytes, format=f"audio/{audio_format}")

with col_right:
    st.subheader("2. Reference Text (Optional Ground Truth)")
    ground_truth = st.text_area(
        "Write the real Arabic transcript here to auto-calculate WER & CER",
        value=default_ground_truth,
        height=145,
        key=f"gt_{audio_name}",
    )

# 2. Model Selection
st.subheader("3. Select Models to Benchmark")
selected_models = st.multiselect(
    "Models (4 strong defaults pre-selected)",
    options=list(AVAILABLE_MODELS.keys()),
    default=DEFAULT_MODELS,
    format_func=lambda m: f"{AVAILABLE_MODELS[m]}  ({m})",
)

run_btn = st.button("🚀 Generate & Compare Models", type="primary", use_container_width=True)

if run_btn:
    if not api_key:
        st.error("Please enter your `OPENROUTER_API_KEY` in the sidebar (or `.env`).")
    elif not audio_bytes:
        st.error("Please select or upload an audio file first.")
    elif not selected_models:
        st.error("Please select at least one model.")
    else:
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        results: List[Dict] = []

        with st.spinner(f"Running {len(selected_models)} models in parallel via OpenRouter..."):
            with ThreadPoolExecutor(max_workers=min(8, len(selected_models))) as pool:
                futures = {
                    pool.submit(
                        transcribe_via_openrouter,
                        api_key,
                        mid,
                        audio_b64,
                        audio_format,
                        language,
                        prompt_hint,
                    ): mid
                    for mid in selected_models
                }
                for fut in as_completed(futures):
                    results.append(fut.result())

        # Keep original selection order
        order_map = {mid: idx for idx, mid in enumerate(selected_models)}
        results.sort(key=lambda r: order_map.get(r["model_id"], 999))

        # Compute WER/CER
        rows = []
        for r in results:
            wer, cer = compute_metrics(ground_truth, r["text"]) if r["ok"] else (None, None)
            r["wer"] = wer
            r["cer"] = cer
            r["word_count"] = len(r["text"].split()) if r["text"] else 0
            rows.append(
                {
                    "Model": r["model_name"],
                    "Model ID": r["model_id"],
                    "Status": "✅ OK" if r["ok"] else "❌ Error",
                    "Latency (s)": r["latency_s"],
                    "Price (Est.)": MODEL_PRICE_PER_HOUR.get(r["model_id"], "-"),
                    "Words": r["word_count"],
                    "WER (%) ↓": wer if wer is not None else "-",
                    "CER (%) ↓": cer if cer is not None else "-",
                }
            )

        st.session_state["last_results"] = results
        st.session_state["last_table"] = rows
        st.session_state["last_audio"] = audio_name

# Optional preload from local benchmark_results.json via ?load=<key>
q_load = st.query_params.get("load", "")
results_cache_path = Path(__file__).resolve().parent / "benchmark_results.json"
if q_load and results_cache_path.exists() and st.session_state.get("last_audio") != audio_name:
    cached_all = json.loads(results_cache_path.read_text(encoding="utf-8"))
    if q_load in cached_all:
        loaded_results = []
        loaded_rows = []
        for item in cached_all[q_load]:
            mid = item["model"]
            txt = item.get("text", "")
            ok = item.get("ok", True)
            wer, cer = compute_metrics(ground_truth, txt) if ok else (None, None)
            rec = {
                "model_id": mid,
                "model_name": AVAILABLE_MODELS.get(mid, mid),
                "ok": ok,
                "latency_s": item.get("latency", 0),
                "text": txt,
                "error": item.get("error"),
                "wer": wer,
                "cer": cer,
                "word_count": len(txt.split()) if txt else 0,
            }
            loaded_results.append(rec)
            loaded_rows.append(
                {
                    "Model": rec["model_name"],
                    "Model ID": mid,
                    "Status": "✅ OK" if ok else "❌ Error",
                    "Latency (s)": rec["latency_s"],
                    "Price (Est.)": MODEL_PRICE_PER_HOUR.get(mid, "-"),
                    "Words": rec["word_count"],
                    "WER (%) ↓": wer if wer is not None else "-",
                    "CER (%) ↓": cer if cer is not None else "-",
                }
            )
        st.session_state["last_results"] = loaded_results
        st.session_state["last_table"] = loaded_rows
        st.session_state["last_audio"] = audio_name

if "last_results" in st.session_state:
    st.divider()
    st.subheader(f"📊 Benchmark Results — `{st.session_state['last_audio']}`")

    df = pd.DataFrame(st.session_state["last_table"])
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Side-by-side cards
    results = st.session_state["last_results"]
    cols = st.columns(2)
    for idx, r in enumerate(results):
        with cols[idx % 2]:
            metric_badge = f"⏱️ {r['latency_s']}s"
            if r.get("wer") is not None:
                metric_badge += f"  |  WER: **{r['wer']}%**  |  CER: **{r['cer']}%**"
            st.markdown(f"#### {r['model_name']}  \n{metric_badge}")
            if r["ok"]:
                text_display = r["text"] if r["text"] else "*(Empty transcription)*"
                st.markdown(f'<div class="rtl-box">{text_display}</div>', unsafe_allow_html=True)
            else:
                st.error(r["error"])
            st.write("")

    # Export for blog post
    st.subheader("📝 Export for Blog Post (`blog.kashida.io`)")
    exp_col1, exp_col2 = st.columns(2)
    md_lines = [
        "| Model | Latency (s) | WER (%) | Transcript |",
        "| :--- | :---: | :---: | :--- |",
    ]
    for r in results:
        safe_txt = (r["text"] or r["error"] or "").replace("\n", " ").replace("|", "\\|")
        wer_str = f"{r['wer']}%" if r.get("wer") is not None else "N/A"
        md_lines.append(f"| {r['model_name']} | {r['latency_s']}s | {wer_str} | {safe_txt} |")
    md_export = "\n".join(md_lines)

    with exp_col1:
        st.download_button(
            "⬇️ Download Markdown Table",
            data=md_export.encode("utf-8"),
            file_name=f"benchmark_{st.session_state['last_audio']}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with exp_col2:
        clean_json = [
            {k: v for k, v in r.items() if k != "raw"}
            for r in results
        ]
        st.download_button(
            "⬇️ Download JSON Results",
            data=json.dumps(clean_json, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"benchmark_{st.session_state['last_audio']}.json",
            mime="application/json",
            use_container_width=True,
        )
