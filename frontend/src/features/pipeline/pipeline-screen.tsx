import type { ReactNode } from "react";
import { WorkspaceHeader } from "@/features/workspace/workspace-header";
import styles from "./pipeline.module.css";

type NodeProps = {
  className: string;
  eyebrow: string;
  title: string;
  status: "green" | "yellow" | "red" | "purple";
  children: ReactNode;
  input?: string;
};

function FlowNode({ className, eyebrow, title, status, children, input }: NodeProps) {
  return (
    <article className={`${styles.node} ${className}`}>
      <header className={styles.nodeHeader}><span className={`${styles.port} ${styles[status]}`} /><h2>{title}</h2><span className={`${styles.port} ${styles[status]} ${styles.outputPort}`} /></header>
      <div className={styles.nodeBody}>
        <span className={styles.nodeEyebrow}>{eyebrow}</span>
        {children}
        {input && <div className={styles.nodeInput}>{input}<span>↵</span></div>}
      </div>
      <span className={`${styles.nodeLabel} ${styles[status]}`}>{status === "green" ? "active" : status === "yellow" ? "watch" : status === "red" ? "alarm" : "repair"}</span>
    </article>
  );
}

export function PipelineScreen() {
  return (
    <main className={styles.page}>
      <WorkspaceHeader activePage="pipeline" />

      <section className={styles.workspace} aria-label="Safety monitoring pipeline diagram">
        <div className={styles.canvasTopline}>
          <div><span className={styles.canvasEyebrow}>WORKFLOW CANVAS / 01</span><h1>Guardrail integrity flow</h1></div>
          <span className={styles.canvasState}><i />All systems represented</span>
        </div>

        <div className={styles.canvasScroll}>
          <div className={styles.canvas}>
            <svg className={styles.connections} viewBox="0 0 1480 720" preserveAspectRatio="none" aria-hidden="true">
              <path d="M322 277 C380 277 305 75 355 75" />
              <path d="M322 277 C390 277 304 381 355 381" />
              <path d="M643 75 C710 75 668 235 730 235" />
              <path d="M643 381 C710 381 675 235 730 235" />
              <path d="M1030 235 C1085 235 1046 71 1100 71" />
              <path d="M1030 235 C1085 235 1048 403 1100 403" />
              <circle cx="322" cy="277" r="3" /><circle cx="730" cy="235" r="3" />
              <circle cx="1030" cy="235" r="3" />
            </svg>

            <FlowNode className={styles.modelNode} eyebrow="PRIMARY GUARDRAIL" title="Model" status="yellow" input="Production safety filter">
              <div className={styles.outputList}><span>safe score <i className={styles.green} /></span><span>unsafe score <i className={styles.red} /></span><span>decision <i className={styles.yellow} /></span></div>
              <p>Scores each request against the current safety policy.</p>
            </FlowNode>

            <FlowNode className={styles.canaryNode} eyebrow="LOW COST · PARALLEL" title="Canary classifier" status="green" input="Shadow scoring enabled">
              <p>Independent signal runs beside the main guardrail.</p>
              <div className={styles.miniStat}><strong>0.91</strong><span>agreement score</span><i>↗</i></div>
            </FlowNode>

            <FlowNode className={styles.windowNode} eyebrow="STREAM PROCESSOR" title="Sliding window" status="green" input="Last 1,000 requests">
              <div className={styles.windowBars} aria-hidden="true"><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /></div>
              <p>Aligns guardrail and canary score distributions.</p>
            </FlowNode>

            <FlowNode className={styles.driftNode} eyebrow="DRIFT DETECTION" title="KS + martingale" status="yellow" input="KS 0.31 / limit 0.25">
              <div className={styles.driftReadout}><strong>Drift detected</strong><span>threshold crossed · step 842</span></div>
              <div className={styles.miniChart} aria-hidden="true"><svg viewBox="0 0 260 54" preserveAspectRatio="none"><path d="M0 40 C26 39 22 28 48 32 S88 45 110 25 S146 30 170 17 S203 11 220 16 S246 5 260 7" /><path className={styles.threshold} d="M0 31 H260" /></svg></div>
            </FlowNode>

            <FlowNode className={styles.recommendNode} eyebrow="OPERATOR REVIEW" title="Alarm & recommendation" status="red">
              <div className={styles.alarmCard}><span>RECOMMENDED ACTION</span><strong>Quarantine</strong><p>Pause the affected route pending review.</p></div>
              <div className={styles.reviewFooter}><span className={styles.avatarSmall}>A</span><span>Human approval required</span><i>↗</i></div>
            </FlowNode>

            <FlowNode className={styles.repairNode} eyebrow="CALIBRATION REPAIR" title="Conformal calibration" status="purple" input="Evidence logged · immutable">
              <div className={styles.repairSteps}><span><b>01</b> Recalibrate</span><span><b>02</b> Validate coverage</span><span><b>03</b> Write evidence</span></div>
              <p>Repair is proposed for operator approval.</p>
            </FlowNode>

            <div className={styles.canvasFootnote}><span>T 00:00s</span><span>KS 0.31</span><span>N 1,000</span><span>WINDOW 15m</span></div>
            <div className={styles.zoomControl} aria-hidden="true"><span>−</span><span>100%</span><span>+</span></div>
          </div>
        </div>
      </section>
    </main>
  );
}
