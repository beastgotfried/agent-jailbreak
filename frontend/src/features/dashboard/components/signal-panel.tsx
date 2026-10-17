import styles from "../dashboard.module.css";
import type { SignalPoint } from "../types";
import { Icon } from "./icon";
import { PanelHeading } from "./panel-heading";
import { SignalChart } from "./signal-chart";

export function SignalPanel({ points }: { points: SignalPoint[] }) {
  const thresholdCrossed = points.some((point) => point.ksStatistic >= point.threshold);

  return (
    <section className={`${styles.panel} ${styles.signalPanel}`} aria-labelledby="signal-heading">
      <PanelHeading
        headingId="signal-heading"
        eyebrow="DETECTION"
        title="Score-shift signal"
        description="KS statistic compared with the configured alarm threshold."
        action={(
          <div className={styles.signalHeaderActions}>
            <span className={styles.windowBadge}>Sliding window</span>
            <span className={thresholdCrossed ? styles.thresholdCrossed : styles.thresholdClear}>
              {thresholdCrossed ? "Threshold crossed" : "Within threshold"}
            </span>
          </div>
        )}
      />
      {points.length === 0 ? (
        <div className={styles.chartEmpty}>
          <div className={styles.chartGlyph}><Icon name="pulse" /></div>
          <strong>No telemetry yet</strong>
          <p>Load a real replay log to see the KS statistic, threshold, and shift point.</p>
        </div>
      ) : (
        <SignalChart points={points} />
      )}
      <div className={styles.chartLegend}>
        <span><i className={styles.legendBlue} /> KS statistic</span>
        <span><i className={styles.legendDashed} /> Alarm threshold</span>
        <span className={styles.legendNote}>No prompt text is shown</span>
      </div>
    </section>
  );
}
