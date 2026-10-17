import type { SignalPoint } from "../types";
import styles from "../dashboard.module.css";

const width = 720;
const height = 196;
const inset = { top: 14, right: 14, bottom: 28, left: 38 };

export function SignalChart({ points }: { points: SignalPoint[] }) {
  const latest = points[points.length - 1];
  const values = points.flatMap((point) => [point.ksStatistic, point.threshold]);
  const max = Math.max(0.25, ...values) * 1.12;
  const x = (index: number) => inset.left + (index / Math.max(points.length - 1, 1)) * (width - inset.left - inset.right);
  const y = (value: number) => inset.top + (1 - value / max) * (height - inset.top - inset.bottom);
  const signal = points.map((point, index) => `${index ? "L" : "M"}${x(index)},${y(point.ksStatistic)}`).join(" ");
  const fill = `${signal} L${x(points.length - 1)},${height - inset.bottom} L${x(0)},${height - inset.bottom} Z`;
  const threshold = points.map((point, index) => `${index ? "L" : "M"}${x(index)},${y(point.threshold)}`).join(" ");
  const firstCrossing = points.find((point) => point.ksStatistic >= point.threshold);
  const labels = [0, Math.floor((points.length - 1) / 3), Math.floor(((points.length - 1) * 2) / 3), points.length - 1];

  return (
    <div className={styles.chartWrap}>
      <svg className={styles.signalChart} role="img" aria-labelledby="signal-chart-title signal-chart-description" viewBox={`0 0 ${width} ${height}`}>
        <title id="signal-chart-title">KS statistic over a sliding window</title>
        <desc id="signal-chart-description">The orange line shows the KS statistic. The gray dashed line shows the alarm threshold.</desc>
        <defs><linearGradient id="signal-fill" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stopColor="#f06445" stopOpacity=".2" /><stop offset="100%" stopColor="#f06445" stopOpacity="0" /></linearGradient></defs>
        {[0, 0.25, 0.5, 0.75, 1].map((fraction) => {
          const lineY = inset.top + fraction * (height - inset.top - inset.bottom);
          return <g key={fraction}><line className={styles.gridLine} x1={inset.left} x2={width - inset.right} y1={lineY} y2={lineY} /><text className={styles.axisLabel} x={inset.left - 10} y={lineY + 3} textAnchor="end">{(max * (1 - fraction)).toFixed(1)}</text></g>;
        })}
        <path d={fill} fill="url(#signal-fill)" />
        <path className={styles.thresholdLine} d={threshold} />
        <path className={styles.signalLine} d={signal} />
        {points.map((point, index) => <circle key={point.step} className={styles.signalPoint} cx={x(index)} cy={y(point.ksStatistic)} r={index === points.length - 1 ? 4.5 : 2.5} />)}
        {labels.map((index, position) => <text key={`${index}-${position}`} className={styles.axisLabel} x={x(index)} y={height - 7} textAnchor={position === 0 ? "start" : position === labels.length - 1 ? "end" : "middle"}>W{points[index].step}</text>)}
      </svg>
      <p className={styles.chartSummary}>Latest: KS {latest.ksStatistic.toFixed(3)} · threshold {latest.threshold.toFixed(3)} · step {latest.step}{firstCrossing ? ` · first crossing at step ${firstCrossing.step}` : " · no threshold crossing"}</p>
    </div>
  );
}
