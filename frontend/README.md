# Integrity Watch dashboard

A modular Next.js App Router frontend for the Safety Monitor Integrity Watch replay demo.

## Run locally

```sh
npm install
npm run dev
```

Open `http://localhost:3000`.

## Structure

```text
src/
  app/                         Next.js routes, document layout, global styles
  features/dashboard/           Dashboard feature and view model
    components/                 Small, focused dashboard components
    data/                       Illustrative dashboard fixture
    dashboard-screen.tsx        Screen composition
    dashboard.module.css        Feature styles and responsive layout
    types.ts                    UI-facing types for the future run-log adapter
  features/pipeline/            Workflow canvas for the monitoring stages
  features/workspace/           Shared dashboard and pipeline navigation
```

The score-shift panel draws its KS statistic and alarm threshold with native SVG, so chart rendering adds no package dependency. The dashboard uses `src/features/dashboard/data/simulated-snapshot.ts` to make the preview usable before a replay log is connected. Every value is illustrative and must not be presented as measured results. Replace this fixture with an adapter to real replay logs when the backend contract is ready. The UI excludes raw prompt text and displays aggregate signals, run status, and evidence references.

The `DashboardSnapshot` type is a UI view model, not the backend JSONL contract. Add an adapter after the team agrees on the run-log schema. Keep parsing and data mapping out of the visual components.

Before deploying on LinuxONE, verify that the Next.js toolchain works natively on s390x, as required by the repository constraints.
