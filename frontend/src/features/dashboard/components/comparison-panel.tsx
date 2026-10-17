import styles from "../dashboard.module.css";
import { Icon } from "./icon";
import { PanelHeading } from "./panel-heading";
import type { DashboardSnapshot } from "../types";

const metricRows = [
  ["Detection latency", "detectionLatency"],
  ["False alarms (negative control)", "falseAlarms"],
  ["Harmful pass-through", "harmfulPassThrough"],
  ["Holdout coverage", "holdoutCoverage"],
] as const;

export function ComparisonPanel({ comparison }: { comparison: DashboardSnapshot["comparison"] }) {
  return (
    <section className={styles.panel} id="runs" aria-labelledby="comparison-heading">
      <PanelHeading headingId="comparison-heading" eyebrow="PAIRED EVALUATION" title="Run A vs Run B" description="Same seed and stream, with and without the watcher." action={<span className={styles.compareTag}>Illustrative</span>} />
      <div className={styles.comparisonTable} role="table" aria-label="Run A and Run B comparison">
        <div className={styles.comparisonRow} role="row"><span role="columnheader">Metric</span><span role="columnheader">Run A</span><span role="columnheader">Run B</span></div>
        {metricRows.map(([metric, key]) => <div className={styles.comparisonRow} role="row" key={key}><span role="cell">{metric}</span><span role="cell">{comparison[key].runA}</span><span role="cell">{comparison[key].runB}</span></div>)}
      </div>
      <p className={styles.comparisonNote}><Icon name="file" />Illustrative values only · not measured results</p>
    </section>
  );
}
