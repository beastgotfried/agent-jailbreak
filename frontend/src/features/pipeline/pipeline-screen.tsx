import type { ReactNode } from "react";
import { WorkspaceHeader } from "@/features/workspace/workspace-header";
import styles from "./pipeline.module.css";

type NodeProps = { className: string; eyebrow: string; title: string; status: "green" | "yellow" | "red" | "purple"; label: string; children: ReactNode; input?: string };

function FlowNode({ className, eyebrow, title, status, label, children, input }: NodeProps) {
  return (
    <article className={`${styles.node} ${className}`}>
      <header className={styles.nodeHeader}><span className={`${styles.port} ${styles[status]}`} /><h2>{title}</h2><span className={`${styles.port} ${styles[status]} ${styles.outputPort}`} /></header>
      <div className={styles.nodeBody}><span className={styles.nodeEyebrow}>{eyebrow}</span>{children}{input && <div className={styles.nodeInput}>{input}<span>↵</span></div>}</div>
      <span className={`${styles.nodeLabel} ${styles[status]}`}>{label}</span>
    </article>
  );
}

export function PipelineScreen() {
  return (
    <main className={styles.page}>
      <WorkspaceHeader activePage="pipeline" />
      <section className={styles.workspace} aria-label="Safety monitoring pipeline diagram">
        <div className={styles.canvasTopline}>
          <div><span className={styles.canvasEyebrow}>WORKFLOW CANVAS / RELEASE IMPLEMENTATION</span><h1>Guardrail integrity flow</h1></div>
          <span className={styles.canvasState}><i />Offline replay · read only</span>
        </div>
        <div className={styles.canvasScroll}>
          <div className={styles.canvas}>
            <svg className={styles.connections} viewBox="0 0 1480 720" preserveAspectRatio="none" aria-hidden="true">
              <path d="M322 277 C380 277 305 75 355 75" /><path d="M322 277 C390 277 304 381 355 381" />
              <path d="M643 75 C710 75 668 235 730 235" /><path d="M643 381 C710 381 675 235 730 235" />
              <path d="M1030 235 C1085 235 1046 71 1100 71" /><path d="M1030 235 C1085 235 1048 403 1100 403" />
              <circle cx="322" cy="277" r="3" /><circle cx="730" cy="235" r="3" /><circle cx="1030" cy="235" r="3" />
            </svg>

            <FlowNode className={styles.modelNode} eyebrow="FROZEN MODEL PAIR" title="Guardrail + canary" status="yellow" label="paired scoring" input="Selected release · TF-IDF pair">
              <div className={styles.outputList}><span>guardrail score <i className={styles.green} /></span><span>canary score <i className={styles.yellow} /></span><span>policy decision <i className={styles.red} /></span></div>
              <p>Both models score the same request; the frozen guardrail policy makes the safety decision.</p>
            </FlowNode>

            <FlowNode className={styles.canaryNode} eyebrow="FINITE HEALTH PANEL" title="Matched probes" status="green" label="8 fit probes" input="Fixed prompt identities">
              <p>Known inputs have recorded guardrail and canary scores.</p>
              <div className={styles.miniStat}><strong>1e−8</strong><span>maximum score tolerance</span></div>
            </FlowNode>

            <FlowNode className={styles.windowNode} eyebrow="RUNTIME SCHEDULE" title="Periodic check" status="green" label="every 16 events" input="Response due after 32 events">
              <div className={styles.windowBars} aria-hidden="true"><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /></div>
              <p>Checks run on accepted events. This panel is finite and cannot cover every input.</p>
            </FlowNode>

            <FlowNode className={styles.driftNode} eyebrow="INCIDENT CONDITION" title="Probe change or unavailable" status="yellow" label="opens incident" input="No traffic KS threshold authority">
              <div className={styles.driftReadout}><strong>Compare paired outputs</strong><span>Change above tolerance, or missing output</span></div>
              <div className={styles.miniChart} aria-hidden="true"><svg viewBox="0 0 260 54" preserveAspectRatio="none"><path d="M0 32 C26 32 31 21 54 25 S91 34 112 24 S151 31 171 19 S212 27 231 14 S248 12 260 9" /><path className={styles.threshold} d="M0 38 H260" /></svg></div>
            </FlowNode>

            <FlowNode className={styles.recommendNode} eyebrow="RECOVERY AUTHORITY" title="Verified recovery" status="green" label="explicit approval">
              <div className={styles.alarmCard}><span>CANDIDATE ACTIVATION</span><strong>Approve + verify</strong><p>A candidate generation is checked before it can become active.</p></div>
              <div className={styles.reviewFooter}><span>Human or actor approval required</span><i>↗</i></div>
            </FlowNode>

            <FlowNode className={styles.repairNode} eyebrow="CONTAINMENT AUTHORITY" title="Hold or unavailable" status="purple" label="separate approval" input="Hold release requires full verification">
              <div className={styles.repairSteps}><span><b>01</b> Approved containment holds requests</span><span><b>02</b> Without authority, unresolved requests are unavailable</span><span><b>03</b> A hold is not recovery</span></div>
            </FlowNode>

            <div className={styles.canvasFootnote}><span>CHECK EVERY 16 ACCEPTED EVENTS</span><span>8 FIXED PROBES</span><span>RESPONSE DELAY 32</span><span>OFFLINE RELEASE FLOW</span></div>
            <div className={styles.zoomControl} aria-hidden="true"><span>−</span><span>100%</span><span>+</span></div>
          </div>
        </div>
        <div className={styles.pipelineNote}><strong>Diagnostic only:</strong> the legacy KS traffic-profile monitor has no action authority, and its reference profile is unqualified for the external XSTest regression. This view describes repository code; it is not connected to a live service.</div>
      </section>
    </main>
  );
}
