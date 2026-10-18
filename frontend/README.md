# Integrity Watch frontend

A modular Next.js App Router frontend for the `safety-watch-tfidf-20261017` release report.

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
  features/dashboard/           Paired replay outcomes and release evidence
    components/                 Focused report panels and shared primitives
    data/release-baseline.ts    Pinned aggregate snapshot from release JSON
    dashboard-screen.tsx        Dashboard composition
    dashboard.module.css        Dashboard styles and responsive layout
    types.ts                    Release report view model
  features/pipeline/            Implemented runtime supervision flow and limits
  features/workspace/            Shared dashboard and pipeline navigation
```

The dashboard is a read-only rendering of the public aggregate report and qualification record on `release/safety-watch-baseline-20261017`. The snapshot intentionally excludes prompts and private model artifacts. Update `release-baseline.ts` only from a reviewed release report, and keep source references and qualification scope visible in the UI.

The pipeline page describes the selected release implementation. Its primary runtime check uses a finite health-probe panel; the legacy KS profile monitor is diagnostic only. The release does not include a production API or persistent live service, and the frontend has no backend connector. Labels and panels must not imply live traffic, production qualification, general-purpose drift guarantees, or autonomous repair.

Before deploying on LinuxONE, verify that the Next.js toolchain works natively on s390x, as required by the repository constraints.
