// Every value here is from the frozen native studies. No invented numbers.
// Sources: synthetic-100k-20261017-01 benchmark-report, release-package-20261017-02
// qualification receipts, data-manifest, Dolly traffic profile report.

export const RELEASE_URL =
  'https://github.com/beastgotfried/agent-jailbreak/tree/release/safety-watch-baseline-20261017'

export const heroStats = [
  { value: '55/55', label: 'runtime incidents confirmed' },
  { value: '45/50', label: 'fault episodes restored' },
  { value: '0/10', label: 'false confirmations in controls' },
  { value: 's390x', label: 'native LinuxONE execution' },
]

export const streamResults = {
  A: {
    correct: 23238, score: 75.6445, harmfulAllowed: 4738, harmfulAllowRate: 29.9116,
    safeRefusals: 2744, safeRefusalRate: 18.4409, held: 0, coverage: 100,
  },
  B: {
    correct: 25096, score: 81.6927, harmfulAllowed: 595, harmfulAllowRate: 3.7563,
    safeRefusals: 3344, safeRefusalRate: 22.4731, held: 1685, coverage: 94.515,
  },
  total: 30720, harmfulTotal: 15840, safeTotal: 14880,
}

export const detection = {
  episodes: 60, faults: 50, confirmed: 55, restored: 45, held: 5,
  controlFalseConfirmations: 0, controlEpisodes: 10,
  confirmationDelayRequests: 15, responseDelayRequests: 32,
}

export const classifier = {
  rows: 20000, accuracy: 88.665, recall: 99.96, fpr: 22.63, rocAuc: 0.9917,
}

export const dataset = {
  total: 100000, train: 70000, validation: 10000, test: 20000,
  families: 100, groups: 91, overlap: 0,
}

export const platform = {
  arch: 's390x', byteOrder: 'big-endian', python: '3.12.15',
  fitSeconds: 39.82, benchmarkSeconds: 191.64, reproductionSeconds: 239.0,
  peakRssMiB: 923.84, testsPass: 292, testsSkip: 4,
  container: '2 CPU quota / 4 GiB memory', sklearn: '1.9.0',
}

export const scoreFlipExample = {
  index: 135, healthy: 0.7426, scaled: 0.1856, threshold: 0.2083,
}

export const dataSources = [
  { name: 'Synthetic study', role: 'Selected model training + benchmark', license: 'Internal', size: '100,000 rows' },
  { name: 'ToxicChat 0124', role: 'Original models + external regression', license: 'CC-BY-NC-4.0', size: '2,853 human test' },
  { name: 'XSTest', role: 'External regression', license: 'CC-BY-4.0', size: '300 rows' },
  { name: 'Dolly 15k', role: 'Traffic profile (no safety labels)', license: 'CC-BY-SA-3.0', size: '14,744 groups' },
  { name: 'WAInjectBench', role: 'Original traffic pools', license: 'Not verified', size: '3,698 rows' },
  { name: 'JBB + HarmBench + StrongREJECT', role: 'Original attack pools', license: 'MIT / not listed', size: '700+ rows' },
]
