import Link from "next/link";
import { Icon } from "@/features/dashboard/components/icon";
import styles from "./workspace-header.module.css";

export function WorkspaceHeader({ activePage }: { activePage: "dashboard" | "pipeline" }) {
  return (
    <header className={styles.header}>
      <Link className={styles.brand} href="/" aria-label="Veyra dashboard">
        <span className={styles.brandMark}><Icon name="shield" /></span>
        <span><strong>Veyra</strong><small>SAFETY MONITOR</small></span>
      </Link>
      <nav className={styles.navigation} aria-label="Workspace">
        <Link className={activePage === "dashboard" ? styles.active : ""} href="/" aria-current={activePage === "dashboard" ? "page" : undefined}>Dashboard</Link>
        <Link className={activePage === "pipeline" ? styles.active : ""} href="/pipeline" aria-current={activePage === "pipeline" ? "page" : undefined}>Pipeline</Link>
      </nav>
      <div className={styles.meta}>
        <span className={styles.platform}><i />s390x qualification</span>
        <span className={styles.mode}><i />read-only report</span>
      </div>
    </header>
  );
}
