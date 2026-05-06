# ASR Benchmark — India Voice Platform

Benchmark pipeline evaluating four Speech-to-Text systems on **Bangalore locality names** spoken in Hindi/Hinglish/Kannada, across 20 real-world noise conditions.

**Core question:** Can ASR models reliably capture place names like *Koramangala*, *Doddanekundi*, or *Kothanur Dinne* from blue-collar workers speaking naturally over a phone?

---

## Models Tested

| Model | Type | Language Config |
|-------|------|----------------|
| **Deepgram** Nova-3 | Cloud API | `language=hi`, smart_format |
| **OpenAI Whisper** | Local (CPU) | Medium model, `language=hi` |
| **Sarvam AI** | Cloud API | `language_code=hi-IN`, Indic-native |
| **AssemblyAI** | Cloud API | universal-3-pro, `language=hi` |

> Sarvam AI was chosen over IndicWhisper for its production-ready API and native Indic language support, which is more realistic for a deployment scenario.

---

## Project Structure

```
asr_benchmark/
├── audio/                  ← 20 .wav recordings (Hindi/Hinglish/Kannada)
├── results/
│   ├── benchmark_results.json   ← Full results (80 evaluations)
│   └── benchmark_results.csv    ← Same data as CSV
├── benchmark.py            ← Main pipeline
├── ground_truth.json       ← Reference transcripts + condition labels
├── dashboard.html          ← Interactive visual report (open in browser)
├── requirements.txt
└── .env.template           ← Copy to .env and add API keys
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt

# Also install ffmpeg (needed by Whisper):
# macOS:  brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg
```

### 2. Add API keys
```bash
cp .env.template .env
# Edit .env — add DEEPGRAM_API_KEY, SARVAM_API_KEY, ASSEMBLYAI_API_KEY
```

### 3. Add your recordings
Place `.wav` files in the `audio/` folder. Naming convention used:
```
01_Sarjapur_crowded.wav
02_Koramangala_clear.wav
...
20_Whitefield_Whatsapp.wav
```

### 4. Fill in ground truth
`ground_truth.json` maps each filename to its reference transcript, condition, and language. Edit to match your actual recordings exactly — including filler words, restarts, or whatever was actually said.

---

## Running the Benchmark

```bash
# Run all four models
python benchmark.py

# Run specific models only
python benchmark.py --models deepgram sarvam
python benchmark.py --models whisper assemblyai
```

Output saved to `results/benchmark_results.json` and `.csv`.

---

## Viewing the Dashboard

1. Open `dashboard.html` in any browser
2. Drag and drop `results/benchmark_results.json` onto the page
3. All charts populate automatically — WER, entity accuracy, latency, noise condition breakdown, and locality heatmap

---

## Metrics

| Metric | What it measures | Why it matters |
|--------|-----------------|----------------|
| **WER** | Word-level edit distance vs ground truth | Standard ASR benchmark metric |
| **CER** | Character-level edit distance | More meaningful when scripts differ |
| **Entity Capture** | Did the locality name appear in output? | The actual business requirement |
| **Latency** | API response time in ms | Production feasibility |

> **Note on WER values:** All models output **Devanagari script** while ground truth is romanized Hindi. Since normalization cannot reconcile scripts, WER values exceed 100% across the board — this is a known evaluation pipeline limitation, not a model failure. Entity capture is similarly affected: only localities with English-origin words (HSR Layout, BTM Layout, Electronic City) score above 0%.

---

## Key Findings

- **Deepgram** leads on entity capture (15%) — occasionally produces mixed-script output that partially matches romanized ground truth
- **Sarvam AI** is the fastest by far (442ms avg) — 31× faster than Whisper, suited for real-time voice flows
- **Whisper** has the lowest WER (128.8%) but CPU inference is too slow for production (up to 117s per file)
- **Hinglish code-switching** (183% WER) hurt models more than most noise conditions — linguistic ambiguity > acoustic degradation
- **Barking dogs** and **rushed speech** were the hardest noise conditions (243% and 237% WER respectively)

---

## Recording Conditions Covered

`clear` · `crowded` · `car` · `pocket` · `construction` · `music` · `whispered` · `hinglish` · `kannada` · `distance` · `mumbled` · `shouted` · `barking` · `fan` · `traffic` · `rushed` · `slowed` · `market` · `news` · `whatsapp`

---

## Next Steps

1. Fix `entity_accuracy()` in `benchmark.py` to handle Devanagari via `indic-transliteration`
2. Re-evaluate Sarvam AI with corrected ground truth — its native Indic output likely performs significantly better
3. Test GPU-accelerated Whisper to bring latency from ~14s to ~1s
4. Expand dataset to 100+ samples for statistical significance

---

## Dependencies

See `requirements.txt`. Key packages: `deepgram-sdk`, `openai-whisper`, `requests` (Sarvam), `assemblyai`, `torch`, `python-dotenv`.
