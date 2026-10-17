# Safety Monitor Integrity Watch

**A health monitor for deployed AI safety filters.** It watches a guardrail with a canary model, detects silent drift, coordinates a human-approved repair, and writes verifiable evidence. Built for IBM LinuxONE (s390x).

IBM Z Datathon 2026 · Theme: Real-Time AI for Critical Decisions · Detect. Decide. Respond.

![Architecture](docs/img/architecture.png)

## The problem

Companies deploy a safety filter and stop watching it. New attacks, changed traffic, and model updates break it silently. It keeps saying "safe" on harmful traffic. The first alarm today comes from users or the press.

## What we build

A watcher. A cheap canary model scores the same traffic as the guardrail, in parallel, never blocking. A statistical monitor reads the disagreement between them and alarms when the pattern changes. A human approves the repair. Every step is logged with a hash-chained evidence record.

## Read these first

- `AGENTS.md` — global context for any AI agent working in this repo.
- `docs/METHOD.md` — the method, start to end.
- `docs/PLAN.md` — the serial to-do list.
- `docs/DECISIONS.md` — every locked decision and why.
- `agents/` — one context file per team role. Point your AI agent at your role file.

## Hard rules

- Event code is written during the event window only. Research, data prep, and environment setup happen before.
- Real data only. No synthetic augmentation. Attack text is data, never executed.
- No secrets, keys, tokens, or raw harmful samples in this repository.
- The method comes from published research. We cite it. We do not claim the algorithm.
