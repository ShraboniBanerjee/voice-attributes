# Design write-up

**Approach & model choice.** I chose `audeering/wav2vec2-large-robust-24-ft-age-gender`
because one model predicts both attributes in a single forward pass — lower
latency than two models — and it was fine-tuned to be robust to noise, matching
the truck-and-warehouse call setting. wav2vec2 embeddings capture speaker traits
(pitch, formants, voice quality) far better than hand-crafted features, and the
weights are public, satisfying the portability constraint. ffmpeg handles codec
decoding so the pipeline is format-agnostic. The model sits behind a swappable
`Backend` interface with a mock implementation, so the whole service is testable
without the 1 GB download.

**Further Improvements:** Replace the energy-based quality gate with a learned VAD
(Silero); add temperature scaling to calibrate confidence; export the model to
ONNX Runtime for 2–4× faster CPU inference; add the language/accent field; and
train a distributional age head so bracket confidence is principled rather than
boundary-derived.

**Scaling to 1,000 concurrent calls.** Run stateless replicas behind a load
balancer with autoscaling. Serve the model via a batching inference server
(Triton/vLLM-style) on GPUs so concurrent 5 s chunks share GPU passes; keep the
FastAPI layer thin and async. Cache attributes per `contact_id`, cap audio
duration, and shed load with backpressure when the quality gate or queue depth
signals overload.
