import styles from "../dashboard.module.css";
import type { ReleaseReport } from "../types";
import { PanelHeading } from "./panel-heading";

const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
const auc = (value: number) => value.toFixed(3);

export function GeneralizationPanel({ regression }: { regression: ReleaseReport["externalRegression"] }) {
  const models = [
    { name: "Guardrail", recall: regression.guardrailRecall, fpr: regression.guardrailFalsePositiveRate, auc: regression.guardrailRocAuc },
    { name: "Canary", recall: regression.canaryRecall, fpr: regression.canaryFalsePositiveRate, auc: regression.canaryRocAuc },
  ];

  return (
    <section className={`${styles.panel} ${styles.generalizationPanel}`} aria-labelledby="generalization-heading">
      <PanelHeading
        headingId="generalization-heading"
        eyebrow="GENERALIZATION CHECK"
        title="External regression"
        description={regression.source}
        action={<span className={styles.regressionTag}>Regression exposed</span>}
      />
      <div className={styles.regressionModels}>
        {models.map((model) => (
          <article className={styles.regressionModel} key={model.name}>
            <div className={styles.regressionModelTitle}><strong>{model.name}</strong><span>ROC-AUC {auc(model.auc)}</span></div>
            <div className={styles.regressionMetrics}>
              <div><span>Recall</span><strong>{percent(model.recall)}</strong></div>
              <div><span>False-positive rate</span><strong>{percent(model.fpr)}</strong></div>
            </div>
          </article>
        ))}
      </div>
      <p className={styles.regressionFootnote}>{regression.status}. Results do not support a production-safety claim.</p>
    </section>
  );
}
