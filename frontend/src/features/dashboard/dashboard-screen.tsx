import type { CSSProperties } from "react";
import { ActionPanel } from "./components/action-panel";
import { ComparisonPanel } from "./components/comparison-panel";
import { EvidencePanel } from "./components/evidence-panel";
import { Icon } from "./components/icon";
import { MetricCard } from "./components/metric-card";
import { SignalPanel } from "./components/signal-panel";
import { simulatedSnapshot } from "./data/simulated-snapshot";
import { WorkspaceHeader } from "../workspace/workspace-header";
import styles from "./dashboard.module.css";

export function DashboardScreen() {
  const snapshot = simulatedSnapshot;
  const ksRatio = snapshot.ksStatistic && snapshot.ksThreshold
    ? snapshot.ksStatistic / snapshot.ksThreshold
    : 0;
  const evidenceRatio = snapshot.martingaleValue && snapshot.martingaleThreshold
    ? snapshot.martingaleValue / snapshot.martingaleThreshold
    : 0;

  return (
    <div className={styles.appShell} id="overview">
      <div className={styles.workspace}>
        <WorkspaceHeader activePage="dashboard" />
        <main className={styles.content}>
          <section className={styles.pageIntro}>
            <div>
              <div className={styles.overline}>GUARDRAIL INTEGRITY <span>·</span> OVERVIEW</div>
              <h1>Safety overview</h1>
            </div>
            <div className={styles.toolbar}>
              <span className={styles.dateControl}><Icon name="clock" /> Last 15 minutes <span>⌄</span></span>
              <span className={styles.filterControl}><Icon name="overview" /> Filter</span>
              <span className={styles.previewChip}><i /> Simulated preview</span>
            </div>
          </section>

          {snapshot.dataMode === "simulated" && (
            <aside className={styles.simulationNotice} aria-label="Simulated data notice">
              <strong>PREVIEW DATA</strong>
              <span>Illustrative replay values; no live guardrail feed is connected.</span>
            </aside>
          )}

          <section className={styles.metricGrid} aria-label="Current run metrics">
            <MetricCard tone="dark" label="Scored requests" value={snapshot.scoredRequests.toLocaleString()} hint="Requests in this replay" icon="runs" />
            <MetricCard tone="plain" label="KS statistic" value={snapshot.ksStatistic?.toFixed(3)} hint={`Threshold ${snapshot.ksThreshold?.toFixed(2)} · crossed`} icon="pulse" />
            <MetricCard tone="plain" label="Martingale evidence" value={snapshot.martingaleValue?.toFixed(1)} hint={`Alarm threshold ${snapshot.martingaleThreshold?.toFixed(0)}`} icon="overview" />
          </section>

          <div className={styles.primaryGrid}>
            <SignalPanel points={snapshot.signal} />
            <section className={`${styles.panel} ${styles.statusPanel}`} aria-labelledby="status-heading">
              <div className={styles.panelHeading}>
                <div><span className={styles.eyebrow}>DETECTION STATUS</span><h2 id="status-heading">Drift detected</h2><p>Review the signal before changing the guardrail.</p></div>
                <span className={styles.alertTag}><i /> Review</span>
              </div>
              <div className={styles.thresholdRing} style={{ "--ring-progress": `${Math.min(ksRatio, 1) * 100}%` } as CSSProperties}>
                <div><strong>{ksRatio.toFixed(2)}×</strong><span>KS threshold</span></div>
              </div>
              <div className={styles.statusStats}>
                <div><span>KS statistic</span><strong>{snapshot.ksStatistic?.toFixed(3)} <small>/ {snapshot.ksThreshold?.toFixed(2)}</small></strong></div>
                <div><span>Martingale evidence</span><strong>{snapshot.martingaleValue?.toFixed(1)} <small>/ {snapshot.martingaleThreshold?.toFixed(0)}</small></strong></div>
              </div>
              <div className={styles.statusCallout}>
                <span className={styles.calloutIcon}><Icon name="clock" /></span>
                <div><span className={styles.eyebrow}>RECOMMENDED ACTION</span><strong>Recalibrate</strong><p>{evidenceRatio.toFixed(2)}× evidence threshold · operator approval required</p></div>
              </div>
            </section>
          </div>

          <div className={styles.secondaryGrid}>
            <ActionPanel snapshot={snapshot} />
            <ComparisonPanel comparison={snapshot.comparison} />
          </div>
          <div className={styles.evidenceRow}><EvidencePanel events={snapshot.events} /></div>

          <footer className={styles.footerNote}>
            <span><Icon name="shield" />Monitoring detects change, not correctness.</span>
            <span>Prompt text is intentionally excluded from this view.</span>
          </footer>
        </main>
      </div>
    </div>
  );
}
