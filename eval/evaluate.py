"""Evaluation harness (bonus).

Runs the configured backend against a labelled speech dataset and reports:

  * gender accuracy
  * age-bracket accuracy (exact, and within-one-bracket)
  * mean confidence
  * a confidence-calibration table (is "0.8 confidence" right ~80% of the time?)

Designed for Mozilla Common Voice, whose validated.tsv has `gender` and `age`
(as decades) columns and mp3 clips in a `clips/` folder.

    # real run against Common Voice:
    python eval/evaluate.py --data /path/to/cv-corpus/en --limit 300

    # no dataset needed - shows the report format on fabricated predictions:
    python eval/evaluate.py --self-test

Note: Common Voice ages are decades; the boundaries (30/45/60) don't align with
decades, so decade->bracket is a documented approximation (by decade midpoint).
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys

# Make `app` importable when run as `python eval/evaluate.py` from repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Common Voice decade label -> our bracket, chosen by the decade's midpoint.
_CV_AGE_TO_BRACKET = {
    "twenties": "18-30",
    "thirties": "31-45",
    "fourties": "31-45",
    "fifties": "46-60",
    "sixties": "60+",
    "seventies": "60+",
    "eighties": "60+",
    "nineties": "60+",
}
_BRACKET_ORDER = ["18-30", "31-45", "46-60", "60+"]


def _within_one(pred: str, truth: str) -> bool:
    if pred not in _BRACKET_ORDER or truth not in _BRACKET_ORDER:
        return pred == truth
    return abs(_BRACKET_ORDER.index(pred) - _BRACKET_ORDER.index(truth)) <= 1


def calibration_table(confidences, correct, n_bins=5):
    """Bin predictions by confidence and compare mean confidence to accuracy."""
    rows = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, c in enumerate(confidences) if (lo <= c < hi or (hi == 1.0 and c == 1.0))]
        if not idx:
            continue
        avg_conf = sum(confidences[i] for i in idx) / len(idx)
        acc = sum(correct[i] for i in idx) / len(idx)
        rows.append((f"{lo:.1f}-{hi:.1f}", len(idx), round(avg_conf, 3), round(acc, 3)))
    return rows


def _print_report(gender_ok, age_exact, age_near, total, confs, correct):
    print("\n=== Evaluation report ===")
    print(f"samples scored        : {total}")
    if total == 0:
        print("no scorable samples found.")
        return
    print(f"gender accuracy       : {gender_ok / total:.3f}")
    print(f"age bracket (exact)   : {age_exact / total:.3f}")
    print(f"age bracket (±1)      : {age_near / total:.3f}")
    print(f"mean gender confidence: {sum(confs) / len(confs):.3f}")
    print("\nconfidence calibration (gender):")
    print(f"  {'bin':<10}{'n':>6}{'avg_conf':>10}{'accuracy':>10}")
    for name, n, avg_conf, acc in calibration_table(confs, correct):
        print(f"  {name:<10}{n:>6}{avg_conf:>10}{acc:>10}")


def run_dataset(data_dir: str, tsv: str, limit: int) -> None:
    from app.audio import decode_to_mono_f32
    from app.brackets import age_bracket_from_years, gender_from_probs
    from app.inference import build_backend

    backend = build_backend()
    tsv_path = os.path.join(data_dir, tsv)
    clips_dir = os.path.join(data_dir, "clips")

    with open(tsv_path, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t") if r.get("gender") and r.get("age")]
    random.shuffle(rows)

    gender_ok = age_exact = age_near = total = 0
    confs, correct = [], []
    for row in rows:
        if total >= limit:
            break
        truth_gender = row["gender"].strip().lower()
        truth_bracket = _CV_AGE_TO_BRACKET.get(row["age"].strip().lower())
        if truth_gender not in ("male", "female") or truth_bracket is None:
            continue
        clip = os.path.join(clips_dir, row["path"])
        try:
            with open(clip, "rb") as f:
                samples, sr = decode_to_mono_f32(f.read())
            pred = backend.predict(samples, sr)
        except Exception as exc:  # skip unreadable clips, keep going
            print(f"skip {row['path']}: {exc}", file=sys.stderr)
            continue

        g_label, g_conf = gender_from_probs(pred.gender_probs)
        a_label, _ = age_bracket_from_years(pred.age_years)
        is_g_correct = g_label == truth_gender
        gender_ok += int(is_g_correct)
        age_exact += int(a_label == truth_bracket)
        age_near += int(_within_one(a_label, truth_bracket))
        confs.append(g_conf)
        correct.append(int(is_g_correct))
        total += 1

    _print_report(gender_ok, age_exact, age_near, total, confs, correct)


def run_self_test() -> None:
    """Fabricate predictions to demonstrate the report format without data."""
    rng = random.Random(0)
    n = 200
    confs, correct = [], []
    gender_ok = age_exact = age_near = 0
    for _ in range(n):
        c = rng.uniform(0.5, 1.0)
        # Simulate a roughly-calibrated model: higher confidence -> more often right.
        is_correct = rng.random() < c
        confs.append(round(c, 3))
        correct.append(int(is_correct))
        gender_ok += int(is_correct)
        age_exact += int(rng.random() < 0.45)
        age_near += int(rng.random() < 0.85)
    _print_report(gender_ok, age_exact, age_near, n, confs, correct)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the voice-attribute model.")
    ap.add_argument("--data", help="Common Voice dir (contains the tsv and clips/)")
    ap.add_argument("--tsv", default="validated.tsv", help="TSV file name")
    ap.add_argument("--limit", type=int, default=200, help="max clips to score")
    ap.add_argument("--self-test", action="store_true", help="demo report, no data")
    args = ap.parse_args()

    if args.self_test or not args.data:
        if not args.self_test:
            print("no --data given; running --self-test demo.\n", file=sys.stderr)
        run_self_test()
    else:
        run_dataset(args.data, args.tsv, args.limit)


if __name__ == "__main__":
    main()
