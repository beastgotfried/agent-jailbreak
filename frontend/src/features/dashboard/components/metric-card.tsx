import styles from "../dashboard.module.css";
import { Icon, type IconName } from "./icon";

type MetricCardProps = { label: string; hint: string; icon: IconName; value?: string; tone?: "lime" | "orange" | "yellow" | "dark" | "plain" };

export function MetricCard({ label, hint, icon, value, tone = "dark" }: MetricCardProps) {
  return (
    <article className={`${styles.metricCard} ${styles[`metric${tone[0].toUpperCase()}${tone.slice(1)}`]}`}>
      <div className={styles.metricTop}><span>{label}</span><span className={styles.metricIcon}><Icon name={icon} /></span></div>
      <strong className={value ? styles.metricValue : styles.metricEmpty}>{value ?? "—"}</strong>
      <p>{hint}</p>
    </article>
  );
}
