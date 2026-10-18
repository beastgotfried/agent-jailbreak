export type ArmOutcome = {
  harmfulAllowRate: number;
  operationalAccuracy: number;
  automaticCoverage: number;
};

export type ReleaseReport = {
  mode: "published-release-aggregate";
  releaseId: string;
  releaseBranch: string;
  releaseCommit: string;
  selectedSourceCommit: string;
  packageSourceCommit: string;
  modelPairId: string;
  model: string;
  canary: string;
  architecture: string;
  studyId: string;
  studyRows: number;
  reviewedSemanticGroups: number;
  pairedEpisodes: number;
  pairedRequests: number;
  seedCount: number;
  baseline: ArmOutcome;
  supervised: ArmOutcome;
  runtime: {
    runtimeConfirmations: number;
    restoredEpisodes: number;
    heldEpisodes: number;
    heldRequests: number;
    preFaultOrNoFaultConfirmations: number;
  };
  externalRegression: {
    source: string;
    guardrailRecall: number;
    guardrailFalsePositiveRate: number;
    guardrailRocAuc: number;
    canaryRecall: number;
    canaryFalsePositiveRate: number;
    canaryRocAuc: number;
    status: string;
  };
  qualification: {
    status: "passed" | "not-qualified";
    nativeTestsPassed: number;
    freshContainer: boolean;
    networkDisabled: boolean;
  };
  artifactHashes: {
    dataManifest: string;
    studyFreeze: string;
  };
};
