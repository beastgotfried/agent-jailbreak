import styles from "../dashboard.module.css";
import type { ArmOutcome } from "../types";
import { PanelHeading } from "./panel-heading";

const measures: Array<{ key: keyof ArmOutcome; label: string; note: string }> = [
  { key: "harmfulAllowRate", label: "Harmful-request allow rate", note: "Lower is better" },
  { key: "operationalAccuracy", label: "Operational accuracy", note: "Source labels · synthetic study" },
  { key: "automaticCoverage", label: "Automatic decision coverage", note: "Held requests excluded" },
];

const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

export function PairedOutcomePanel({ baseline, supervised }: { baseline: ArmOutcome; supervised: ArmOutcome }) {
  return (
    <section className={`${styles.panel} ${styles.outcomePanel}`} aria-labelledby="paired-outcome-heading">
      <PanelHeading
        headingId="paired-outcome-heading"
        eyebrow="PAIRED REPLAY · AGGREGATE"
        title="Guardrail only vs supervised runtime"
        description="Same study inputs across both arms; release report results, not live traffic."
      />
      <div className={styles.outcomeLegend}>
        <span><i className={styles.baselineDot} /> Guardrail only</span>
        <span><i className={styles.supervisedDot} /> Watcher + supervised response</span>
      </div>
      <div className={styles.outcomeRows}>
        {measures.map(({ key, label, note }) => {
          const baselineValue = baseline[key];
          const supervisedValue = supervised[key];
          return (
            <article className={styles.outcomeRow} key={key}>
              <div className={styles.outcomeHeading}><strong>{label}</strong><span>{note}</span></div>
              <div className={styles.outcomeArm}>
                <span className={styles.armLabel}>A</span>
                <div className={styles.outcomeTrack}><i className={styles.baselineBar} style={{ width: percent(baselineValue) }} /></div>
                <strong>{percent(baselineValue)}</strong>
              </div>
              <div className={styles.outcomeArm}>
                <span className={styles.armLabel}>B</span>
                <div className={styles.outcomeTrack}><i className={styles.supervisedBar} style={{ width: percent(supervisedValue) }} /></div>
                <strong>{percent(supervisedValue)}</strong>
              </div>
            </article>
          );
        })}
      </div>
      <p className={styles.outcomeFootnote}>60 paired episodes · 30,720 requests · five reused seeds · 91 reviewed semantic groups</p>
    </section>
  );
}
