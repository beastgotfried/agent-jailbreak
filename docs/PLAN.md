# PLAN — serial to-do list

## Before the event

1. Train and freeze guardrail + canary on the VM. (Model lead, 2-4 h)
2. Build prompt pools as JSONL: benign + attacks. (Model + Systems, 1 h)
3. Convert the two parquets to CSV on a laptop, upload. (Systems, 30 min)
4. Write the one-page interface contract: stream, log, score formats. (All, 30 min)
5. Smoke test: 100 prompts, guardrail scores, KS computes, alarm fires. (Systems, 1 h)

## Event, hours 0-8

6. Agree interfaces aloud. No code before this. (All, 30 min)
7. Build the monitor: disagreement, KS window, martingale alarm. (Monitor lead, 4 h)
8. Finish replay harness + agent stub. (Systems lead, 4 h)
9. Build dashboard reading the logs. (Platform lead, 4 h)

## Event, hours 8-16

10. Build conformal repair, projection sweep. (Monitor + Model, 3 h)
11. First end-to-end run: harness -> monitor -> dashboard. (All, 1 h)
12. Run A: guardrail alone, silent failure recorded. (Systems, 1 h)
13. Run B: guardrail + watcher, alarm + repair recorded. (Systems, 1 h)
14. Negative control: benign only, watcher quiet. (Systems, 30 min)
15. Scorecard printed from logs. (Systems, 1 h)

## Event, hours 16-24

16. Freeze results. Cut anything broken. (Platform lead owns the cut line, 30 min)
17. Record the 3-minute video. LinuxONE visible. (Platform lead, 2 h)
18. Finish the pitch. (Platform lead, 1 h)
19. Submit. Keep one hour of buffer. (All)

Critical path: 1, 5, 7, 11, 13, 15, 17.
