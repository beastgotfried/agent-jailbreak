import styles from "../dashboard.module.css";
import type { DashboardSnapshot } from "../types";
import { Icon } from "./icon";
import { PanelHeading } from "./panel-heading";

const actionLabels: Record<NonNullable<DashboardSnapshot["recommendation"]>, string> = {
  recalibrate: "Recalibrate",
  quarantine: "Quarantine",
  escalate: "Escalate",
};

export function ActionPanel({ snapshot }: { snapshot: DashboardSnapshot | null }) {
  return (
    <section className={`${styles.panel} ${styles.actionPanel}`} aria-labelledby="action-heading">
      <PanelHeading headingId="action-heading" eyebrow="HUMAN REVIEW" title="Operator decision" />
      {snapshot?.recommendation ? (
        <div className={styles.recommendation}>
          <span className={styles.recommendationIcon}><Icon name="check" /></span>
          <div><span className={styles.eyebrow}>RECOMMENDED RESPONSE</span><strong>{actionLabels[snapshot.recommendation]}</strong><p>Review the run evidence before acting. The monitor detects change; it does not confirm harm.</p></div>
        </div>
      ) : (
        <div className={styles.actionEmpty}>
          <span className={styles.reviewIcon}><Icon name="clock" /></span>
          <strong>No action pending</strong>
          <p>A recommendation will appear after the monitor records an alarm.</p>
        </div>
      )}
      <div className={styles.actionFootnote}><span className={styles.lockDot} />Human approval required for repair</div>
    </section>
  );
}
