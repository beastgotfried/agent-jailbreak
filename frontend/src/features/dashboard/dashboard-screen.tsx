import { GeneralizationPanel } from "./components/generalization-panel";
import { MetricCard } from "./components/metric-card";
import { PairedOutcomePanel } from "./components/paired-outcome-panel";
import { ProvenancePanel } from "./components/provenance-panel";
import { RuntimeOutcomePanel } from "./components/runtime-outcome-panel";
import { releaseBaseline } from "./data/release-baseline";
import { WorkspaceHeader } from "../workspace/workspace-header";
import styles from "./dashboard.module.css";

export function DashboardScreen() {
  const report = releaseBaseline;
  const harmfulAllowReduction = (report.baseline.harmfulAllowRate - report.supervised.harmfulAllowRate) * 100;

  return (
    <div className={styles.appShell} id="overview">
      <div className={styles.workspace}>
        <WorkspaceHeader activePage="dashboard" />
        <main className={styles.content} id="main-content" tabIndex={-1}>
          <section className={styles.pageIntro}>
            <div>
              <div className={styles.overline}>VEYRA <span>·</span> RELEASE EVIDENCE</div>
              <h1>Safety monitor baseline</h1>
              <p>Paired replay summary · {report.studyId}</p>
            </div>
            <div className={styles.reportMeta}>
              <span className={styles.reportBadge}><i /> Read-only snapshot</span>
              <a className={styles.reportLink} href="https://github.com/beastgotfried/agent-jailbreak/tree/release/safety-watch-baseline-20261017" target="_blank" rel="noreferrer">View release ↗</a>
            </div>
          </section>

          <section className={styles.metricGrid} aria-label="Release summary">
            <MetricCard tone="dark" label="Harmful allow rate" value={`${(report.supervised.harmfulAllowRate * 100).toFixed(1)}%`} hint={`${harmfulAllowReduction.toFixed(1)} point decrease vs guardrail only`} icon="pulse" />
            <MetricCard tone="plain" label="Episodes restored" value={`${report.runtime.restoredEpisodes} / ${report.pairedEpisodes}`} hint={`${report.runtime.heldEpisodes} contained · hold is not recovery`} icon="runs" />
            <MetricCard tone="plain" label="External ROC-AUC" value={report.externalRegression.guardrailRocAuc.toFixed(3)} hint="ToxicChat · exposed regression" icon="overview" />
          </section>

          <div className={styles.primaryGrid}>
            <PairedOutcomePanel baseline={report.baseline} supervised={report.supervised} />
            <RuntimeOutcomePanel runtime={report.runtime} episodes={report.pairedEpisodes} />
          </div>

          <details className={styles.evidenceDetails}>
            <summary><span><b>Release validation & provenance</b><small>External regression metrics, qualification record, and source hashes</small></span><i aria-hidden="true">+</i></summary>
            <div className={styles.evidenceGrid}>
              <GeneralizationPanel regression={report.externalRegression} />
              <ProvenancePanel report={report} />
            </div>
          </details>

          <footer className={styles.footerNote}>
            <span>30,720 requests · 60 paired episodes · 5 reused seeds</span>
            <span>No live service or action connector is configured.</span>
          </footer>
        </main>
      </div>
    </div>
  );
}
