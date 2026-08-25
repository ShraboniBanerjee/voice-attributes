# Voice Attribute Inference Service

This service listens to a short piece of caller audio and gives back a best guess
at the caller's **gender** and **age bracket**, along with a confidence score for
each and a flag saying how usable the audio actually was. It's meant to sit
inside a voice-agent stack for logistics calls, where the line is often noisy and
you know nothing about the person before they start talking.

The three endpoints:

```
POST /analyze   -> gender + age bracket + confidence + audio_quality + processing_ms
WS   /stream    -> live predictions as audio streams in (bonus)
GET  /healthz   -> is the service up, and which backend is loaded
```

---

## Running it

### With Docker (this is what I'd reach for)

```bash
docker compose up --build
```

The first run is slow, and I want to be upfront about why: the real model weights
(~1 GB) download from the Hugging Face Hub the first time, and everything gets
installed into a clean image. After that it's cached in a named volume, so
restarts are quick.

If you just want to see the service work without waiting on the model download,
run the mock backend instead:

```bash
MODEL_BACKEND=mock docker compose up --build
```

### Without Docker

You'll need `ffmpeg` on your machine (`brew install ffmpeg` on a Mac,
`apt install ffmpeg` on Linux). Then:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

MODEL_BACKEND=mock python -m uvicorn app.main:app --port 8000     # fast, no download
# or MODEL_BACKEND=wav2vec2 for the real model
```

One gotcha I hit on macOS: if you have Homebrew's own `uvicorn` installed, plain
`uvicorn ...` may run the wrong Python and complain it can't find `numpy`. Using
`python -m uvicorn ...` forces it through your virtual environment and avoids that.

---

## Trying it out

There's a script that writes a synthetic sample so you don't have to find a clip
first:

```bash
python scripts/make_sample.py                 # writes sample.wav
curl -s -F "file=@sample.wav" -F "contact_id=abc-123" \
     http://localhost:8000/analyze | python -m json.tool
```

A response looks like this:

```json
{
  "contact_id": "abc-123",
  "gender": { "prediction": "male", "confidence": 0.95 },
  "age_bracket": { "prediction": "31-45", "confidence": 0.8 },
  "processing_ms": 86,
  "audio_quality": "good"
}
```

You can also skip multipart and send the raw bytes:

```bash
curl -s --data-binary "@sample.wav" -H "Content-Type: audio/wav" \
     "http://localhost:8000/analyze?contact_id=abc-123"
```

Note that `sample.wav` is a plain synthetic tone. It's good for checking the
pipeline runs, but it isn't a real voice, so don't read too much into the
prediction. For an actual result, feed it a real recording, e.g. an `.mp3` from
[Mozilla Common Voice](https://commonvoice.mozilla.org/).

If you'd rather click than curl, open `http://localhost:8000/docs` for the
auto-generated Swagger UI and upload a file there.

### Streaming

The WebSocket endpoint expects raw 16-bit PCM, mono, 16 kHz. It sends back a
prediction roughly once a second as audio arrives, then a final one tagged
`"final": true` when the stream ends. `tests/test_stream.py` has a working client
if you want to see how it's driven.

---

## How it works

The whole thing is a short pipeline, and every stage is deliberately simple:

1. **Decode.** Incoming bytes go through `ffmpeg`, which handles pretty much any
   codec and hands back a clean 16 kHz mono signal. Doing this once up front means
   nothing downstream has to care what format the caller sent.
2. **Quality check.** Before spending any time on the model, a quick energy-based
   pass estimates how much actual speech is present and how far the voice rises
   above the background. That becomes the `audio_quality` flag.
3. **Inference.** The audio goes to the model, which returns gender probabilities
   and an estimated age.
4. **Shape the answer.** Gender is the highest-probability class; age gets bucketed
   into a bracket; both get a confidence, and the JSON goes back.

The important behaviour lives in step 2. If there isn't enough speech, the service
returns `"unknown"` and never bothers the model. If the audio is noisy but usable,
it still predicts but trims the confidence down. The point is to never hand back a
confident-sounding answer that's really just noise.

### Why this model

I used [`audeering/wav2vec2-large-robust-24-ft-age-gender`](https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender).
A few reasons: it does age and gender in a single pass rather than needing two
models, it was fine-tuned to hold up under noise (which is the whole point in a
warehouse or truck cab), and the weights are public so the "runs with only public
model weights" constraint is satisfied. wav2vec2 also captures the things that
actually carry gender and age in a voice (pitch, vocal-tract resonance, voice
quality) far better than features I'd hand-pick myself.

### Why a swappable backend

The model sits behind a small `Backend` interface with two implementations: the
real one, and a `MockBackend` that fakes a plausible answer from a rough pitch
estimate. That let me test the entire pipeline in CI without pulling a 1 GB model,
and it means switching models later is a config change rather than a rewrite.

### The age-confidence caveat

The age head outputs a single number, not a probability per bracket. So the bracket
confidence is something I derived: it's high when the predicted age sits in the
middle of a bracket and low when it's near an edge. It's a reasonable stand-in, not
a real probability, and I've called that out rather than dressing it up.

---

## Privacy

Caller audio is treated as PII and never stored:

- It's never written to disk. Decoding streams through ffmpeg in memory, and the
  streaming buffer only ever lives in RAM.
- It's dropped as soon as the response is built. The audio only exists for the
  duration of a single request.
- Nothing raw is logged. The logs record the contact id, the predicted labels,
  the quality flag and the timing, but never the audio itself.

---

## Latency: what I was aiming for vs. what I got

The target was under 500 ms end to end on a 5-second clip. **I did not hit that on
my hardware.**

I developed this on a MacBook Air (Apple Silicon, CPU only, no GPU). On that
machine the large wav2vec2 model dominates everything else in the pipeline, and a
single request lands somewhere around **1.4 to 3.3 seconds** depending on clip
length and whether things are warmed up. The decode and quality steps are cheap
(tens of milliseconds); it's the model forward pass on CPU that blows the budget.

What I did do to keep things as fast as they reasonably could be:

- The model loads once at startup and is reused, never reloaded per request.
- The quality gate runs before inference, so genuinely unusable audio skips the
  expensive step entirely.
- The mock backend responds in a few milliseconds, which is handy for
  load-testing the web layer on its own.

**How I'd actually get under 500 ms with more time**, roughly in the order I'd try
them:

1. **Run it on a GPU.** The single biggest win. The model was never going to be
   fast on a laptop CPU; on a GPU this comfortably clears the target.
2. **Export to ONNX Runtime and quantize to int8.** This keeps the same model and
   accuracy but typically gives a 2–4x speedup on CPU, which is the right fix when
   a GPU isn't available.
3. **Batch concurrent requests** through the model so overlapping calls share a
   forward pass instead of queueing.

I deliberately did *not* just swap in a smaller, less accurate model to game the
number on my laptop. The honest position is: the architecture is right, the
optimisation work (GPU / ONNX / batching) is what's left, and I'd rather show I
know where the time goes than hide it.

---

## What I didn't get to

A few things I'd pick up next if I kept going:

- **A proper learned VAD.** My quality gate is a simple energy heuristic. It works,
  but it assumes the clip has some quiet gaps to measure the noise floor against,
  so perfectly continuous speech can throw it off. Dropping in Silero or webrtcvad
  behind the same interface would make speech detection much sharper.
- **Confidence calibration.** The model's raw probabilities aren't guaranteed to
  mean what they say (a "0.9" isn't reliably right 90% of the time). The eval
  harness measures this; the fix is temperature scaling, which I didn't implement.
- **The latency work above** (GPU / ONNX / quantization).
- **Language / accent detection.** One of the bonus fields. I scoped it out to keep
  the core solid, but it would slot in as another head or a second lightweight model.
- **A real distributional age head**, so age-bracket confidence is a true
  probability instead of the boundary-distance approximation I'm using now.

---

## Known limitations

- **Age from voice is just hard.** Adjacent brackets often sound almost identical,
  so age confidence should be read as a soft hint, not a guarantee. Gender is much
  more reliable.
- **Gender is binary here** to match the required contract. The model also has a
  `child` class, which I map to `"unknown"`, and any predicted age under 18 also
  maps to `"unknown"` since the contract has no under-18 bracket.
- **Streaming expects raw PCM**, not a compressed stream. In a real deployment a
  telephony gateway would transcode the call leg to PCM before forwarding it here.
- **transformers is pinned to `<5`.** The model's loader code was written for the
  4.x line and breaks on the 5.x rewrite, so `requirements.txt` caps it. Worth
  knowing if you ever bump dependencies.

---

## Tests and evaluation

```bash
MODEL_BACKEND=mock python -m pytest -v
```

The tests cover the label and bracket mapping, the quality gate on synthetic
clean / noisy / silent / too-short audio, the full `/analyze` contract, and the
streaming endpoint. They all run on the mock backend, so they're fast and need no
model download.

The eval harness runs the model against a labelled dataset and reports accuracy
plus a confidence-calibration table:

```bash
python eval/evaluate.py --self-test                          # demo output, no data
python eval/evaluate.py --data /path/to/cv-corpus/en --limit 300   # real run
```