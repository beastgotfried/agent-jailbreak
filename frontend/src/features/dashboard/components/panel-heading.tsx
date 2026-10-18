import type { ReactNode } from "react";
import styles from "../dashboard.module.css";

type PanelHeadingProps = { eyebrow?: string; title: string; headingId?: string; description?: string; action?: ReactNode };

export function PanelHeading({ eyebrow, title, headingId, description, action }: PanelHeadingProps) {
  return (
    <div className={styles.panelHeading}>
      <div>{eyebrow && <span className={styles.eyebrow}>{eyebrow}</span>}<h2 id={headingId}>{title}</h2>{description && <p>{description}</p>}</div>
      {action}
    </div>
  );
}
