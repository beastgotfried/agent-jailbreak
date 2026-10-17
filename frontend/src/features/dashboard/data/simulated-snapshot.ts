import type { DashboardSnapshot } from "../types";

/** Illustrative UI fixture only. These values are not benchmark results. */
export const simulatedSnapshot: DashboardSnapshot = {
  dataMode: "simulated",
  runId: "PREVIEW-042",
  scenario: "Injection shift preview",
  state: "awaiting-review",
  scoredRequests: 480,
  ksStatistic: 0.26,
  ksThreshold: 0.19,
  martingaleValue: 23.8,
  martingaleThreshold: 20,
  recommendation: "recalibrate",
  signal: [
    { step: 40, ksStatistic: 0.025, threshold: 0.19 },
    { step: 80, ksStatistic: 0.031, threshold: 0.19 },
    { step: 120, ksStatistic: 0.028, threshold: 0.19 },
    { step: 160, ksStatistic: 0.034, threshold: 0.19 },
    { step: 200, ksStatistic: 0.03, threshold: 0.19 },
    { step: 240, ksStatistic: 0.041, threshold: 0.19 },
    { step: 280, ksStatistic: 0.046, threshold: 0.19 },
    { step: 320, ksStatistic: 0.092, threshold: 0.19 },
    { step: 360, ksStatistic: 0.154, threshold: 0.19 },
    { step: 400, ksStatistic: 0.211, threshold: 0.19 },
    { step: 440, ksStatistic: 0.238, threshold: 0.19 },
    { step: 480, ksStatistic: 0.26, threshold: 0.19 },
  ],
  events: [
    { id: "preview-start", label: "Replay started", occurredAt: "Step 000", detail: "Illustrative injection-shift scenario · seed 42" },
    { id: "preview-shift", label: "Shift introduced", occurredAt: "Step 320", detail: "Score distribution begins to move from the reference" },
    { id: "preview-alarm", label: "Shift alarm", occurredAt: "Step 424", detail: "KS statistic crossed the preview threshold" },
    { id: "preview-action", label: "Review required", occurredAt: "Step 480", detail: "Recalibration recommendation is awaiting human approval" },
  ],
  comparison: {
    detectionLatency: { runA: "—", runB: "24 steps" },
    falseAlarms: { runA: "—", runB: "0 / 10" },
    harmfulPassThrough: { runA: "34%", runB: "18%" },
    holdoutCoverage: { runA: "84%", runB: "91%" },
  },
};
