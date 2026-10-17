# DECISIONS — locked, with reasons

1. **Train both classifiers ourselves.** DeBERTa was evaluated and excluded: task mismatch (injection-only), unknowable training data, s390x risk. Owner decision, 2026-10-16.
2. **Label:** unsafe = toxicity OR jailbreaking. Severity = sigmoid probability 0-1. Binary decides, severity informs.
3. **Training data:** full ToxicChat train (owner decision). Headline metrics on human-annotated rows only.
4. **Environment:** train and run inside the container (sklearn 1.9.0). No cross-environment pickles.
5. **Threshold:** frozen near 5% FPR on a validation slice of train. Test touched once.
6. **No transformers anywhere in the decision path.** CPU-only, deterministic, reproducible.
7. **WhyLabs excluded.** whylogs-sketching ships no s390x wheel; langkit and openllmtelemetry fail to install. Their content is reference-only under `vendor/whylabs/` on the VM.
8. **The product is the watcher.** Never pitch a new guardrail, a firewall, or injection detection. The comparison set stays empty; the firewall market stays out.
9. **Replay-study framing.** Real prompts, constructed stream order. Never claim production traces.
10. **Detection proves change, not correctness.** Common-mode failure is a declared limitation in every write-up.

Supersedes: any earlier chat, pitch, or doc that conflicts with this list.
