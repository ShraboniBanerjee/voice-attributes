import numpy as np

from app.quality import assess

SR = 16_000


def _voiced(seconds=2.0, tone_amp=0.3, noise_amp=0.0005, sr=SR):
    """Alternating 0.2 s voiced tone / 0.1 s gap, plus a noise floor.

    The gaps give the estimator quiet frames to measure the noise floor against,
    the way real speech has pauses between words.
    """
    rng = np.random.default_rng(0)
    out = []
    t_tone = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
    tone = tone_amp * np.sin(2 * np.pi * 120 * t_tone)
    gap = np.zeros(int(sr * 0.1))
    while sum(len(c) for c in out) < sr * seconds:
        out.append(tone)
        out.append(gap)
    x = np.concatenate(out)[: int(sr * seconds)]
    x = x + noise_amp * rng.standard_normal(len(x))
    return x.astype(np.float32)


def test_clean_speech_is_good():
    report = assess(_voiced(noise_amp=0.0005), SR)
    assert report.flag == "good"
    assert report.speech_seconds >= 0.5


def test_noisy_speech_is_degraded():
    report = assess(_voiced(tone_amp=0.3, noise_amp=0.06), SR)
    assert report.flag == "degraded"


def test_silence_is_insufficient():
    x = (0.0003 * np.random.default_rng(1).standard_normal(SR * 2)).astype(np.float32)
    report = assess(x, SR)
    assert report.flag == "insufficient"


def test_too_short_is_insufficient():
    report = assess(_voiced(seconds=0.2), SR)
    assert report.flag == "insufficient"
