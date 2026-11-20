# Arabic STT Benchmark (OpenRouter + Streamlit)

Interactive benchmark tool for comparing Speech-to-Text (STT) models on real-world noisy & dialectal Arabic audio via **OpenRouter**'s unified `/api/v1/audio/transcriptions` endpoint.

## Quick Start

```bash
cd arabic-stt-benchmark
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your OPENROUTER_API_KEY to .env (or paste it directly in the UI sidebar)
streamlit run app.py
```

## Features
- **4 Strong Default Models**: ElevenLabs Scribe v2, OpenAI GPT-4o Transcribe, Google Gemini 3.5 Transcribe, Deepgram Nova-3 (plus Whisper-1 baseline, Whisper Large v3, Mistral Voxtral, AssemblyAI, Qwen3 ASR, Grok STT).
- **Parallel Execution**: Calls all selected models concurrently and tracks exact latency.
- **Arabic-Aware WER & CER**: Normalizes alef variants, taa marbuta, yaa, diacritics, and punctuation for fair accuracy scoring against your ground-truth transcript.
- **Pre-loaded Prod Samples**: Includes the 3 real-world noisy/dialect MP3 samples from `../audio_samples/`.
- **Blog-Ready Export**: Export results directly as a Markdown table or JSON for `blog.kashida.io`.
