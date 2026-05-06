"""
ASR Benchmarking Pipeline
Tests: Deepgram, OpenAI Whisper (local), IndicWhisper (AI4Bharat), AssemblyAI Speech-to-Text
"""

import os
import json
import time
import csv
import re
import ssl
from pathlib import Path
from dotenv import load_dotenv

# Fix SSL certificate issue on macOS for Whisper
try:
    import certifi
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    ssl._create_default_https_context = lambda: ssl_context
except Exception:
    pass

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────────────────────────

AUDIO_DIR = Path("audio")
GROUND_TRUTH_FILE = Path("ground_truth.json")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

LOCALITIES = [
    "Koramangala", "Indiranagar", "Whitefield", "Electronic City", "Marathahalli",
    "Jayanagar", "Rajajinagar", "Hebbal", "Yelahanka", "Banashankari",
    "HSR Layout", "BTM Layout", "Majestic", "Silk Board", "Bellandur",
    "Sarjapur", "Bommanahalli", "KR Puram", "Peenya", "Yeshwanthpur",
    "Byatarayanapura", "Kadugondanahalli", "Hesaraghatta", "Chikkabanavara",
    "Rajarajeshwarinagar", "Kothanur Dinne", "Thanisandra", "Doddanekundi",
    "Kengeri Upanagara", "Thalaghattapura"
]

# ─── METRICS ──────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase, strip punctuation for fair comparison."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text

def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute WER using dynamic programming."""
    ref = normalize(reference).split()
    hyp = normalize(hypothesis).split()
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0

    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i-1] == hyp[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])

    return d[len(ref)][len(hyp)] / len(ref)

def entity_accuracy(reference: str, hypothesis: str, localities: list) -> dict:
    """
    Check if the key locality name was captured correctly.
    Returns: { "found_in_ref": str, "captured": bool }
    """
    ref_norm = normalize(reference)
    hyp_norm = normalize(hypothesis)

    for loc in localities:
        loc_norm = normalize(loc)
        if loc_norm in ref_norm:
            captured = loc_norm in hyp_norm
            # also check partial match (first word of multi-word locality)
            if not captured and " " in loc_norm:
                first_word = loc_norm.split()[0]
                captured = first_word in hyp_norm
            return {"locality": loc, "captured": captured}

    return {"locality": None, "captured": False}

def character_error_rate(reference: str, hypothesis: str) -> float:
    """CER — useful for non-English scripts."""
    ref = normalize(reference).replace(" ", "")
    hyp = normalize(hypothesis).replace(" ", "")
    if len(ref) == 0:
        return 0.0
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1): d[i][0] = i
    for j in range(len(hyp) + 1): d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return d[len(ref)][len(hyp)] / len(ref)

# ─── MODEL RUNNERS ────────────────────────────────────────────────────────────

def run_deepgram(audio_path: str) -> dict:
    """Run Deepgram Nova-3 API."""
    try:
        from deepgram import DeepgramClient
        
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            return {"transcript": "", "error": "DEEPGRAM_API_KEY not set", "latency_ms": 0}

        client = DeepgramClient(api_key=api_key)
        
        start = time.time()
        with open(audio_path, "rb") as f:
            audio_data = f.read()

        # Updated method path for SDK v6+
        response = client.listen.v1.media.transcribe_file(
            request=audio_data,
            model="nova-3",
            language="hi",
            smart_format=True
        )
        
        latency = (time.time() - start) * 1000
        transcript = response.results.channels[0].alternatives[0].transcript
        return {"transcript": transcript, "error": None, "latency_ms": round(latency, 1)}

    except Exception as e:
        return {"transcript": "", "error": str(e), "latency_ms": 0}


def run_whisper(audio_path: str, model_size: str = "medium") -> dict:
    """Run OpenAI Whisper locally."""
    try:
        import whisper
        model = whisper.load_model(model_size)
        start = time.time()
        result = model.transcribe(audio_path, language="hi", task="transcribe")
        latency = (time.time() - start) * 1000
        return {"transcript": result["text"].strip(), "error": None, "latency_ms": round(latency, 1)}
    except Exception as e:
        return {"transcript": "", "error": str(e), "latency_ms": 0}


def run_sarvam(audio_path: str) -> dict:
    """Run Sarvam AI Speech-to-Text API."""
    try:
        import requests
        api_key = os.getenv("SARVAM_API_KEY")
        if not api_key:
            return {"transcript": "", "error": "SARVAM_API_KEY not set", "latency_ms": 0}

        with open(audio_path, "rb") as f:
            files = {"file": (audio_path, f, "audio/wav")}
            payload = {"language_code": "hi-IN"}
            headers = {"api-subscription-key": api_key}

            start = time.time()
            response = requests.post(
                "https://api.sarvam.ai/speech-to-text",
                headers=headers,
                files=files,
                data=payload
            )
            latency = (time.time() - start) * 1000

        result = response.json()
        transcript = result.get("transcript", "")
        return {"transcript": transcript, "error": None, "latency_ms": round(latency, 1)}

    except Exception as e:
        return {"transcript": "", "error": str(e), "latency_ms": 0}


def run_assemblyai(audio_path: str) -> dict:
    """Run AssemblyAI Speech-to-Text API."""
    try:
        import assemblyai as aai

        api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not api_key:
            return {"transcript": "", "error": "ASSEMBLYAI_API_KEY not set", "latency_ms": 0}

        aai.settings.api_key = api_key
        transcriber = aai.Transcriber()

        config = aai.TranscriptionConfig(
            language_code="hi",
            language_detection=False,
            speech_models=["universal-3-pro", "universal-2"],
        )

        start = time.time()
        transcript_response = transcriber.transcribe(audio_path, config=config)
        latency = (time.time() - start) * 1000

        if transcript_response.status == aai.TranscriptStatus.error:
            return {"transcript": "", "error": transcript_response.error, "latency_ms": round(latency, 1)}

        transcript = transcript_response.text.strip() if transcript_response.text else ""
        return {"transcript": transcript, "error": None, "latency_ms": round(latency, 1)}
    except Exception as e:
        return {"transcript": "", "error": str(e), "latency_ms": 0}


# ─── MODEL REGISTRY ───────────────────────────────────────────────────────────

MODELS = {
    "deepgram":      run_deepgram,
    "whisper":       run_whisper,
    "sarvam":        run_sarvam,
    "assemblyai":    run_assemblyai,
}

# ─── MAIN BENCHMARK ───────────────────────────────────────────────────────────

def load_ground_truth() -> dict:
    """Load ground truth transcripts from JSON file."""
    if not GROUND_TRUTH_FILE.exists():
        print(f"[WARNING] {GROUND_TRUTH_FILE} not found. Creating template...")
        template = {}
        for audio_file in sorted(AUDIO_DIR.glob("*.wav")):
            template[audio_file.name] = {
                "transcript": "FILL IN GROUND TRUTH HERE",
                "condition": "quiet|noisy|phone|whisper",
                "language": "hindi|hinglish|kannada"
            }
        with open(GROUND_TRUTH_FILE, "w") as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
        print(f"Template created at {GROUND_TRUTH_FILE}. Please fill in transcripts and re-run.")
        return {}
    with open(GROUND_TRUTH_FILE) as f:
        return json.load(f)


def run_benchmark(models_to_run: list = None):
    """Run the full benchmark pipeline."""
    if models_to_run is None:
        models_to_run = list(MODELS.keys())

    ground_truth = load_ground_truth()
    if not ground_truth:
        return

    audio_files = sorted(AUDIO_DIR.glob("*.wav")) + sorted(AUDIO_DIR.glob("*.mp3")) + sorted(AUDIO_DIR.glob("*.m4a"))
    if not audio_files:
        print(f"[ERROR] No audio files found in {AUDIO_DIR}/")
        print("Please add your recordings as .wav, .mp3, or .m4a files.")
        return

    results = []
    total = len(audio_files) * len(models_to_run)
    done = 0

    print(f"\n{'='*60}")
    print(f"ASR BENCHMARK — {len(audio_files)} files × {len(models_to_run)} models")
    print(f"{'='*60}\n")

    for audio_file in audio_files:
        fname = audio_file.name
        gt = ground_truth.get(fname, {})
        reference = gt.get("transcript", "")
        condition = gt.get("condition", "unknown")
        language = gt.get("language", "unknown")

        if not reference or reference == "FILL IN GROUND TRUTH HERE":
            print(f"[SKIP] {fname} — no ground truth")
            continue

        for model_name in models_to_run:
            done += 1
            print(f"[{done}/{total}] {model_name} ← {fname} ...", end=" ", flush=True)

            model_fn = MODELS[model_name]
            output = model_fn(str(audio_file))

            transcript = output["transcript"]
            error = output["error"]
            latency = output["latency_ms"]

            if error:
                print(f"ERROR: {error}")
                wer = None
                cer = None
                entity = {"locality": None, "captured": False}
            else:
                wer = round(word_error_rate(reference, transcript), 4)
                cer = round(character_error_rate(reference, transcript), 4)
                entity = entity_accuracy(reference, transcript, LOCALITIES)
                print(f"WER={wer:.2%}  Entity={'✅' if entity['captured'] else '❌'}  Latency={latency}ms")

            results.append({
                "file": fname,
                "model": model_name,
                "condition": condition,
                "language": language,
                "reference": reference,
                "hypothesis": transcript,
                "wer": wer,
                "cer": cer,
                "locality": entity["locality"],
                "entity_captured": entity["captured"],
                "latency_ms": latency,
                "error": error,
            })

    # Save CSV
    csv_path = RESULTS_DIR / "benchmark_results.csv"
    if results:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n✅ Results saved to {csv_path}")

    # Save JSON
    json_path = RESULTS_DIR / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Results saved to {json_path}")

    # Print summary
    print_summary(results)
    return results


def print_summary(results: list):
    """Print a quick summary table."""
    from collections import defaultdict

    model_stats = defaultdict(lambda: {"wer_sum": 0, "cer_sum": 0, "entity_hits": 0,
                                        "latency_sum": 0, "count": 0, "errors": 0})

    for r in results:
        m = r["model"]
        if r["error"]:
            model_stats[m]["errors"] += 1
            continue
        model_stats[m]["wer_sum"] += r["wer"] or 0
        model_stats[m]["cer_sum"] += r["cer"] or 0
        model_stats[m]["entity_hits"] += 1 if r["entity_captured"] else 0
        model_stats[m]["latency_sum"] += r["latency_ms"] or 0
        model_stats[m]["count"] += 1

    print(f"\n{'='*75}")
    print(f"{'MODEL':<18} {'WER':>8} {'CER':>8} {'ENTITY ACC':>12} {'AVG LATENCY':>13} {'ERRORS':>8}")
    print(f"{'─'*75}")
    for model, s in model_stats.items():
        n = s["count"]
        if n == 0:
            print(f"{model:<18} {'N/A':>8} {'N/A':>8} {'N/A':>12} {'N/A':>13} {s['errors']:>8}")
        else:
            avg_wer = s["wer_sum"] / n
            avg_cer = s["cer_sum"] / n
            entity_acc = s["entity_hits"] / n
            avg_lat = s["latency_sum"] / n
            print(f"{model:<18} {avg_wer:>7.2%} {avg_cer:>7.2%} {entity_acc:>11.2%} {avg_lat:>11.0f}ms {s['errors']:>8}")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ASR Benchmark Pipeline")
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()),
                        default=list(MODELS.keys()),
                        help="Which models to run (default: all)")
    args = parser.parse_args()
    run_benchmark(models_to_run=args.models)
