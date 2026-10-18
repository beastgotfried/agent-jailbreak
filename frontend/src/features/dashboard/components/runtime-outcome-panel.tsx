import styles from "../dashboard.module.css";
import type { ReleaseReport } from "../types";
import { PanelHeading } from "./panel-heading";

export function RuntimeOutcomePanel({ runtime, episodes }: { runtime: ReleaseReport["runtime"]; episodes: number }) {
  return (
    <section className={`${styles.panel} ${styles.runtimePanel}`} aria-labelledby="runtime-outcome-heading">
      <PanelHeading eyebrow="SUPERVISED RESPONSE" headingId="runtime-outcome-heading" title="Recovery & containment" description="Reported outcomes in the paired replay." />
      <div className={styles.recoveryHeadline}>
        <strong>{runtime.restoredEpisodes}<span> / {episodes}</span></strong>
        <div><b>Episodes restored</b><small>Verified recovery in the selected protocol</small></div>
      </div>
      <div className={styles.runtimeFacts}>
        <div><span>Runtime confirmations</span><strong>{runtime.runtimeConfirmations}</strong></div>
        <div><span>Episodes contained</span><strong>{runtime.heldEpisodes}</strong></div>
        <div><span>Requests held</span><strong>{runtime.heldRequests.toLocaleString()}</strong></div>
      </div>
      <aside className={styles.holdNote}><strong>A hold is containment, not recovery.</strong><span>Held requests were not executed and reduce automatic coverage.</span></aside>
    </section>
  );
}
