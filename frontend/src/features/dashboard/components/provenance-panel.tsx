import styles from "../dashboard.module.css";
import type { ReleaseReport } from "../types";
import { PanelHeading } from "./panel-heading";

const releaseBase = "https://github.com/beastgotfried/agent-jailbreak/blob/release/safety-watch-baseline-20261017";
const shortHash = (hash: string) => `${hash.slice(0, 12)}…${hash.slice(-8)}`;

export function ProvenancePanel({ report }: { report: ReleaseReport }) {
  return (
    <section className={`${styles.panel} ${styles.provenancePanel}`} aria-labelledby="provenance-heading">
      <PanelHeading
        headingId="provenance-heading"
        eyebrow="RELEASE PROVENANCE"
        title="Reproducible report"
        description="Aggregate evidence from the frozen release snapshot."
        action={<span className={styles.qualificationTag}><i /> Native qualification passed</span>}
      />
      <dl className={styles.provenanceList}>
        <div><dt>Release</dt><dd>{report.releaseId}</dd></div>
        <div><dt>Target runtime</dt><dd>LinuxONE · {report.architecture} · CPU only</dd></div>
        <div><dt>Native qualification</dt><dd>{report.qualification.nativeTestsPassed} passed · fresh container · network disabled</dd></div>
        <div><dt>Model source</dt><dd><code>{shortHash(report.selectedSourceCommit)}</code></dd></div>
        <div><dt>Package source</dt><dd><code>{shortHash(report.packageSourceCommit)}</code></dd></div>
        <div><dt>Model pair ID</dt><dd><code>{shortHash(report.modelPairId)}</code></dd></div>
        <div><dt>Data manifest SHA-256</dt><dd><code>{shortHash(report.artifactHashes.dataManifest)}</code></dd></div>
      </dl>
      <div className={styles.provenanceLinks}>
        <a href={`${releaseBase}/releases/baseline-20261017/summary.json`} target="_blank" rel="noreferrer">Release summary ↗</a>
        <a href={`${releaseBase}/releases/baseline-20261017/qualification.json`} target="_blank" rel="noreferrer">Qualification record ↗</a>
      </div>
      <p className={styles.privateArtifactNote}>Model files and replay inputs remain in the private release kit; this page shows public aggregate results only.</p>
    </section>
  );
}
