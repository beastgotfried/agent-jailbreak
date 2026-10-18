import type { ReleaseReport } from "../types";

/** Aggregate-only snapshot copied from the public release summary and qualification records. */
export const releaseBaseline: ReleaseReport = {
  mode: "published-release-aggregate",
  releaseId: "safety-watch-tfidf-20261017",
  releaseBranch: "release/safety-watch-baseline-20261017",
  releaseCommit: "8a86ce01767a02d613b66f38a1d20de312cf61e1",
  selectedSourceCommit: "d86d384e2ee7d0f07f2f2fc5bf2b6dec64e13cac",
  packageSourceCommit: "c5fad060420c8cb1c34f84eb4fd951b193a969c6",
  modelPairId: "adb5b660369f0a8b9024a3d725548aeca034d86eac72cbaa3e975a4a77fa2790",
  model: "TF-IDF + logistic regression (C=4)",
  canary: "Word TF-IDF + logistic regression",
  architecture: "s390x",
  studyId: "synthetic-100k-20261017-01",
  studyRows: 100_000,
  reviewedSemanticGroups: 91,
  pairedEpisodes: 60,
  pairedRequests: 30_720,
  seedCount: 5,
  baseline: {
    harmfulAllowRate: 0.2991161616161616,
    operationalAccuracy: 0.7564453125,
    automaticCoverage: 1,
  },
  supervised: {
    harmfulAllowRate: 0.03756313131313131,
    operationalAccuracy: 0.8169270833333333,
    automaticCoverage: 0.9451497395833334,
  },
  runtime: {
    runtimeConfirmations: 55,
    restoredEpisodes: 45,
    heldEpisodes: 5,
    heldRequests: 1_685,
    preFaultOrNoFaultConfirmations: 0,
  },
  externalRegression: {
    source: "ToxicChat 0124 · human-annotated test subset",
    guardrailRecall: 0.39779005524861877,
    guardrailFalsePositiveRate: 0.44480128462464874,
    guardrailRocAuc: 0.4957792805480947,
    canaryRecall: 0.44751381215469616,
    canaryFalsePositiveRate: 0.5050180650341228,
    canaryRocAuc: 0.46902107254624936,
    status: "Exposed external regression; not fresh confirmation",
  },
  qualification: {
    status: "passed",
    nativeTestsPassed: 292,
    freshContainer: true,
    networkDisabled: true,
  },
  artifactHashes: {
    dataManifest: "63db0f854b36ffe7d6f1e3053d70502ed644055509628cd302934db05e639597",
    studyFreeze: "0febddd1a30e14a607ebe2ca3ef1b0489be3c1ca00398f87a0cfcab841002a72",
  },
};
