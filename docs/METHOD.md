# METHOD — start to end

## 1. Problem

Deployed safety filters fail silently. New attacks, shifted traffic, and model updates break them. Labels do not arrive in real time. The first alarm today comes from outside the company.

## 2. Product

A watcher, not a filter. One guardrail sits in the path. A canary scores the same traffic in parallel and never decides. A monitor reads the disagreement between them. A human approves the repair. Evidence is hash-chained.

## 3. Models

- Guardrail: TF-IDF char n-grams (3-5) + logistic regression.
- Canary: TF-IDF word n-grams (1-2) + logistic regression. Different family, different failures.
- Both trained on ToxicChat train, frozen, digested, carded. Error correlation measured and reported.

## 4. Detection

- Signal: score disagreement over a sliding window.
- Test: KS statistic against a frozen benign reference window.
- Alarm: scan-martingale threshold, calibrated on the negative control.
- Detection proves change, not correctness. Holdout labels decide harm.

## 5. Response

- Triage: data fault, harmless drift, or degradation.
- Repair: conformal recalibration, projection to <=32 dims (sweep 8/16/32/64).
- Authority: a human approves. The repair is verified after. Rollback version recorded.

## 6. Benchmark

- Streams: negative control, attack shift, injection shift. Domain shift and over-refusal only if time remains.
- Run A: guardrail alone. Run B: guardrail + watcher. Paired seeds, identical streams.
- Metrics: detection latency in steps, detection rate, false alarms on negative control, harmful pass-through A vs B, coverage before/after repair.

## 7. Demo

Live slow replay on the VM. Benign traffic, attacks at T, alarm at T + N, repair, restored coverage, scorecard. LinuxONE instance visible in the video.

## 8. Pitch

Guardrails fail silently. A canary watches them. Statistics catch the drift. Conformal math repairs it. Same attacks twice: alone, the guardrail never noticed; watched, the failure was caught in N steps and repaired, with proof.
