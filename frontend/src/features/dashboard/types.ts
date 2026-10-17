/** UI-facing view model. An adapter will map the agreed JSONL schema to this. */
export type SignalPoint = { step: number; ksStatistic: number; threshold: number };

export type EvidenceEvent = {
  id: string;
  label: string;
  occurredAt: string;
  detail?: string;
};

export type DashboardSnapshot = {
  dataMode: "simulated" | "live";
  runId: string;
  scenario: string;
  state: "replaying" | "alarm" | "awaiting-review" | "verifying" | "complete";
  scoredRequests: number;
  ksStatistic?: number;
  ksThreshold?: number;
  martingaleValue?: number;
  martingaleThreshold?: number;
  signal: SignalPoint[];
  events: EvidenceEvent[];
  recommendation?: "recalibrate" | "quarantine" | "escalate";
  comparison: {
    detectionLatency: { runA: string; runB: string };
    falseAlarms: { runA: string; runB: string };
    harmfulPassThrough: { runA: string; runB: string };
    holdoutCoverage: { runA: string; runB: string };
  };
};
