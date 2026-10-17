import styles from "../dashboard.module.css";
import type { EvidenceEvent } from "../types";
import { Icon } from "./icon";
import { PanelHeading } from "./panel-heading";

export function EvidencePanel({ events }: { events: EvidenceEvent[] }) {
  return (
    <section className={styles.panel} id="evidence" aria-labelledby="evidence-heading">
      <PanelHeading headingId="evidence-heading" eyebrow="AUDIT" title="Evidence trail" description="Run events and references from the hash-chained log." action={<span className={styles.chainStatus}><span /> Preview events</span>} />
      {events.length === 0 ? (
        <div className={styles.listEmpty}>
          <span className={styles.emptyIcon}><Icon name="file" /></span>
          <div><strong>No evidence events</strong><p>Events will appear when a replay log is connected.</p></div>
        </div>
      ) : (
        <ol className={styles.eventList}>
          {events.map((event) => <li key={event.id}><span className={styles.eventMarker} /><div><strong>{event.label}</strong><p>{event.detail}</p></div><time>{event.occurredAt}</time></li>)}
        </ol>
      )}
    </section>
  );
}
