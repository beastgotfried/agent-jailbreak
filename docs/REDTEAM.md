# REDTEAM — the attacks we already survived, and the honest answers

## The five hardest questions

**1. Why will the second model not fail the same way?**
It is not a backup gate. It never decides. It supplies a reference signal from a different feature family. The alarm reads change in the relationship, not correctness. Common-mode failure remains possible and is declared.

**2. Can the guardrail degrade without any score drift?**
Yes. A score-preserving attack is invisible to any score-based monitor. Mitigation: scheduled holdout re-evaluation and continuous random human sampling, not alarm-triggered only.

**3. Does repair not need the labels you said never arrive?**
In production, yes: repair waits for a small human review batch. We say so. The demo verifies repair on a locked holdout and labels that as the demo mechanism.

**4. Is this not just drift monitoring?**
Drift tooling monitors model accuracy. Our layer is safety-control integrity with a repair loop and evidence. Method from arXiv 2606.11949 (cited). The combination is unshipped; the components are not.

**5. Is the broken-update demo theater?**
Partly. Real silent failures come from attack evolution and traffic drift. That is why the benchmark includes attack-injection and injection shifts, and why the write-up calls the version swap a simulation.

## Kill-shot criteria (what would make us stop)

- A product already ships guardrail-integrity monitoring with repair and evidence. Link required.
- The canary signal fails to correlate with holdout-measured harm in our own runs.
- The negative control alarms. Then the statistic is broken for our traffic.

## Falsification experiments we run ourselves

- Benign domain shift: high alarm rate weakens the claim.
- Score-preserving attack: silent failure proves the blind spot. We report it.
- Null-run audit: alarm rate far above design breaks the statistic.
