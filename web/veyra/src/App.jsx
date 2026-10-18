import React, { useEffect } from 'react'
import {
  RELEASE_URL, heroStats, streamResults, detection, classifier,
  dataset, platform, scoreFlipExample, dataSources,
} from './data.js'

function useReveal() {
  useEffect(() => {
    const els = document.querySelectorAll('.reveal')
    const io = new IntersectionObserver(
      (entries) => entries.forEach((e) => e.isIntersecting && e.target.classList.add('is-in')),
      { threshold: 0.15 }
    )
    els.forEach((el) => io.observe(el))
    return () => io.disconnect()
  }, [])
}

function Mark() {
  return (
    <span className="brand-mark" aria-hidden="true">
      <svg viewBox="0 0 12 12" fill="none">
        <rect x="1" y="1" width="4" height="4" fill="#001141" />
        <rect x="7" y="1" width="4" height="4" fill="#001141" opacity="0.55" />
        <rect x="1" y="7" width="4" height="4" fill="#001141" opacity="0.55" />
        <rect x="7" y="7" width="4" height="4" fill="#001141" />
      </svg>
    </span>
  )
}

function Nav() {
  return (
    <nav className="nav" aria-label="Primary">
      <div className="nav-inner">
        <a className="brand" href="#top" aria-label="Veyra home"><Mark />Veyra</a>
        <div className="nav-links">
          <a href="#product">Product</a>
          <a href="#benchmarks">Benchmarks</a>
          <a href="#recovery">Recovery</a>
          <a href="#linuxone">LinuxONE</a>
          <a href="#evidence">Evidence</a>
          <a className="nav-cta" href={RELEASE_URL} target="_blank" rel="noreferrer">View release</a>
        </div>
      </div>
    </nav>
  )
}

function Hero() {
  return (
    <header className="hero" id="top">
      <div className="wrap">
        <div className="hero-grid">
          <div>
            <p className="eyebrow">IBM Z Datathon · Native LinuxONE · s390x</p>
            <h1>A guardrail<br />needs a <em>watcher</em>.</h1>
            <p className="hero-sub">
              Veyra checks guardrail runtime behavior and verifies an approved response before activation.
            </p>
            <div className="hero-ctas">
              <a className="btn btn-primary" href="#benchmarks">See the benchmark</a>
              <a className="btn btn-ghost" href={RELEASE_URL} target="_blank" rel="noreferrer">Release branch ↗</a>
            </div>
          </div>
          <figure className="hero-card reveal">
            <figcaption className="card-label">Recorded native replay — fault → confirm → verified recovery</figcaption>
            <img src="./visuals/runtime-trace.svg" width={809} height={253} fetchPriority="high" decoding="async" alt="Score error over 512 requests: guardrail-only stays elevated after a fault at request 128; Veyra returns to baseline after verified recovery at request 175." />
          </figure>
        </div>
        <div className="hero-stats reveal">
          {heroStats.map((s) => (
            <div className="hstat" key={s.label}>
              <div className="v">{s.value}</div>
              <div className="l">{s.label}</div>
            </div>
          ))}
        </div>
      </div>
    </header>
  )
}

function Problem() {
  const e = scoreFlipExample
  return (
    <section className="sec" id="problem">
      <div className="wrap">
        <h2 className="sec-title reveal">A valid score can hide a changed control.</h2>
        <p className="sec-sub reveal">
          The request is identical. The model is identical. One deployment fault quietly rescales the output — and the decision flips.
        </p>
        <div className="problem-grid">
          <div className="flip-card reveal">
            <p className="card-label">Recorded scale fault · seed 0 · request index {e.index}</p>
            <div className="flip-row">
              <div className="flip-score">
                <div className="lbl">Approved baseline</div>
                <div className="num">{e.healthy.toFixed(4)}</div>
              </div>
              <div className="flip-arrow" aria-hidden="true">→</div>
              <div className="flip-score bad">
                <div className="lbl">After ×0.25 runtime fault</div>
                <div className="num">{e.scaled.toFixed(4)}</div>
              </div>
            </div>
            <div className="threshold-line">
              <span>Decision threshold</span>
              <strong>{e.threshold}</strong>
            </div>
          </div>
          <ul className="problem-list reveal">
            <li>A lower score can mean a safer request — or a changed scoring path. One number cannot tell you which.</li>
            <li>A score alone does not confirm that the runtime still matches the approved baseline.</li>
            <li>Labels can arrive after the system makes affected decisions.</li>
          </ul>
        </div>
      </div>
    </section>
  )
}

function Product() {
  return (
    <section className="sec" id="product" style={{ background: 'var(--layer)' }}>
      <div className="wrap">
        <h2 className="sec-title reveal">In simple terms: a smoke detector for your smoke detector.</h2>
        <p className="sec-sub reveal">
          Your guardrail checks requests. Veyra checks whether the guardrail still behaves like the baseline you approve.
        </p>
        <div className="lanes reveal">
          <div className="lane">
            <span className="num">01</span>
            <h3>Decide</h3>
            <p>The frozen guardrail keeps making allow / refuse decisions. A canary model scores the same requests in parallel, purely as an observer.</p>
            <span className="tag">guardrail + canary</span>
          </div>
          <div className="lane">
            <span className="num">02</span>
            <h3>Verify</h3>
            <p>Every 16 requests, eight fixed probes check both models against the approved baseline. A change confirms an incident — within 15 requests in our tests.</p>
            <span className="tag">matched probes</span>
          </div>
          <div className="lane">
            <span className="num">03</span>
            <h3>Respond</h3>
            <p>A candidate fix runs on real verification probes before activation. If nothing verifies, the service holds explicitly. Containment is never dressed up as recovery.</p>
            <span className="tag">approval-gated</span>
          </div>
        </div>
      </div>
    </section>
  )
}

function Benchmarks() {
  const { A, B } = streamResults
  const rows = [
    ['Correct automatic decisions', `${A.correct.toLocaleString()} / 30,720`, `${B.correct.toLocaleString()} / 30,720`, true],
    ['Operational score', `${A.score}%`, `${B.score}%`, true],
    ['Harmful allows', `${A.harmfulAllowed.toLocaleString()} / 15,840`, `${B.harmfulAllowed.toLocaleString()} / 15,840`, true],
    ['Harmful allow rate', `${A.harmfulAllowRate}%`, `${B.harmfulAllowRate}%`, true],
    ['Safe refusal rate', `${A.safeRefusalRate}%`, `${B.safeRefusalRate}%`, false],
    ['Held requests (containment)', '0', `${B.held.toLocaleString()} (5.49%)`, false],
  ]
  return (
    <section className="sec" id="benchmarks">
      <div className="wrap">
        <div className="bench-head">
          <div>
            <p className="eyebrow">Measured on 30,720 paired requests</p>
            <h2 className="sec-title reveal">Benchmarks: same traffic, two arms.</h2>
            <p className="sec-sub reveal">
              The guardrail-only arm and the Veyra arm receive identical requests. Every request counts, including holds.
            </p>
          </div>
          <div className="bench-delta reveal">
            <div className="big">+6.05 pp</div>
            <div className="cap">correct automatic decisions<br />75.64% → 81.69%</div>
          </div>
        </div>
        <figure className="fig reveal">
          <img src="./visuals/stream-outcomes.svg" width={949} height={166} loading="lazy" decoding="async" alt="Three bar comparisons: correct automatic decisions 75.64 versus 81.69 percent; harmful allow rate 29.91 versus 3.76 percent; safe refusal rate 18.44 versus 22.47 percent." />
          <figcaption>fig. 01 — complete-stream outcomes · guardrail only vs Veyra · 60 paired episodes</figcaption>
        </figure>
        <div className="bench-wrap reveal" role="region" aria-label="Complete-stream benchmark comparison" tabIndex={0}>
          <table className="bench-table">
            <thead>
              <tr><th scope="col">Metric</th><th className="num" scope="col">Guardrail only</th><th className="num" scope="col">Veyra</th></tr>
            </thead>
            <tbody>
              {rows.map(([label, a, b, good]) => (
                <tr key={label}>
                  <td>{label}</td>
                  <td className="num">{a}</td>
                  <td className={`num ${good ? 'good' : ''}`}>{b}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <figure className="fig reveal">
          <img src="./visuals/response-outcomes.svg" width={882} height={139} loading="lazy" decoding="async" alt="Fifty fault episodes: 45 restored, 5 held; plus 10 no-fault control episodes with zero false confirmations." />
          <figcaption>fig. 02 — {detection.confirmed}/55 runtime incidents confirmed · 45 restored · 5 held · {detection.controlFalseConfirmations}/{detection.controlEpisodes} false confirmations in controls</figcaption>
        </figure>
      </div>
    </section>
  )
}

function Recovery() {
  return (
    <section className="sec" id="recovery" style={{ background: 'var(--layer)' }}>
      <div className="wrap">
        <h2 className="sec-title reveal">Recovery requires evidence, not optimism.</h2>
        <p className="sec-sub reveal">
          A candidate becomes active only after it executes on real verification probes and matches the approved baseline — twice.
        </p>
        <div className="flow reveal">
          <div className="flow-step"><span className="idx">128</span><h3>Fault begins</h3><p>A runtime change alters guardrail output mid-stream.</p></div>
          <div className="flow-step"><span className="idx">143</span><h3>Probes confirm</h3><p>Eight matched probes detect the change in 15 requests.</p></div>
          <div className="flow-step"><span className="idx">175</span><h3>Candidate executes</h3><p>The fix runs on 128 disjoint verification probes.</p></div>
          <div className="flow-step"><span className="idx">→</span><h3>Verified activation</h3><p>Both models match the baseline. The plan activates once, approval-bound.</p></div>
          <div className="flow-step hold"><span className="idx">else</span><h3>Explicit hold</h3><p>Nothing verified? Service holds. Never a silent pass.</p></div>
        </div>
        <div className="terminal reveal" aria-label="Native LinuxONE verification terminal">
          <div><span className="p">$</span> uname -m <span className="dim"># on IBM LinuxONE</span></div>
          <div className="ok">s390x</div>
          <div><span className="p">$</span> safety-watch inspect --models frozen/</div>
          <div className="dim">verified_model_files: 4 · model_pair: adb5b660… · architecture: s390x <span className="ok">✓</span></div>
          <div><span className="p">$</span> pytest -q</div>
          <div className="ok">292 passed, 4 skipped (optional pyarrow) — 14.10s</div>
        </div>
      </div>
    </section>
  )
}

function LinuxOne() {
  return (
    <section className="sec" id="linuxone">
      <div className="wrap">
        <p className="eyebrow">Native IBM LinuxONE</p>
        <h2 className="sec-title reveal">Built and verified where enterprise AI runs.</h2>
        <p className="sec-sub reveal">
          Training, inference, monitoring, recovery, and audit all execute natively on s390x — no x86 emulation, no external model API, no GPU.
        </p>
        <div className="z-grid">
          <div className="z-facts reveal">
            <div className="z-fact"><div className="v">{platform.fitSeconds}s</div><div className="k">classifier fit, native CPU</div></div>
            <div className="z-fact"><div className="v">{platform.peakRssMiB} MiB</div><div className="k">peak memory, full study</div></div>
            <div className="z-fact"><div className="v">{platform.testsPass}</div><div className="k">native tests passing</div></div>
            <div className="z-fact"><div className="v">{platform.reproductionSeconds}s</div><div className="k">offline-container reproduction</div></div>
          </div>
          <ul className="z-list reveal">
            <li><span className="tick">✓</span><div><strong>Big-endian s390x, end to end</strong><span>Data generation, model fit, inference, recovery, and audit run on the same native architecture.</span></div></li>
            <li><span className="tick">✓</span><div><strong>Reproduced with the network off</strong><span>A fresh container with networking disabled regenerated 100,000 rows and matched every score, decision, and metric.</span></div></li>
            <li><span className="tick">✓</span><div><strong>Local model inference</strong><span>The selected pipeline needs no external model API. Raw requests and model binaries stay outside public Git.</span></div></li>
            <li><span className="tick">✓</span><div><strong>Evidence you can re-run</strong><span>SHA-256 manifests bind every artifact. The audit recomputes all 30,720 events independently.</span></div></li>
          </ul>
        </div>
      </div>
    </section>
  )
}

function Evidence() {
  return (
    <section className="sec" id="evidence" style={{ background: 'var(--layer)' }}>
      <div className="wrap">
        <h2 className="sec-title reveal">We publish the denominator.</h2>
        <p className="sec-sub reveal">
          Every claim on this page traces to a hash-frozen artifact. Shipped capability and enterprise roadmap stay clearly separated.
        </p>
        <div className="status-cols reveal">
          <div className="status-col shipped">
            <h3>Shipped &amp; verified on LinuxONE</h3>
            <ul>
              <li>Frozen TF-IDF guardrail + canary pair</li>
              <li>Runtime probes with 15-request confirmation</li>
              <li>Verified affine correction and trusted reload</li>
              <li>Explicit containment with approved release</li>
              <li>100,000-row benchmark, independently audited</li>
              <li>Fresh offline-container reproduction</li>
            </ul>
          </div>
          <div className="status-col roadmap">
            <h3>Enterprise roadmap (PRD)</h3>
            <ul>
              <li>Web console and incident views</li>
              <li>Live collectors, Kafka, durable state</li>
              <li>Customer enforcement connectors</li>
              <li>Multi-tenant HA and capacity qualification</li>
              <li>Hardware attestation profiles</li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  )
}

function Datasets() {
  return (
    <section className="sec" id="data">
      <div className="wrap">
        <h2 className="sec-title reveal">Every dataset has a role, a split, and a license.</h2>
        <p className="sec-sub reveal">
          The selected models train on our controlled synthetic study. External sources are pinned, hashed, and used only for their declared purpose.
        </p>
        <figure className="fig reveal">
          <img src="./visuals/data-split.svg" width={882} height={125} loading="lazy" decoding="async" alt="100,000 synthetic rows split into 70,000 train, 10,000 validation, and 20,000 test, class-balanced." />
          <figcaption>fig. 03 — {dataset.total.toLocaleString()} rows · {dataset.families} template families · {dataset.groups} reviewed intent groups · zero cross-split overlap</figcaption>
        </figure>
        <div className="ds-grid reveal">
          {dataSources.map((d) => (
            <div className="ds" key={d.name}>
              <div className="name">{d.name}</div>
              <div className="role">{d.role}</div>
              <div className="meta"><span>{d.license}</span><span className="size">{d.size}</span></div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

function Footer() {
  return (
    <footer>
      <div className="wrap">
        <div className="foot-grid">
          <div>
            <a className="brand" href="#top" style={{ marginBottom: 18, display: 'inline-flex' }}><Mark />Veyra</a>
            <p className="foot-tag">Observe change. Verify recovery. Keep control.</p>
          </div>
          <div className="foot-links">
            <div>
              <h3>Project</h3>
              <a href="#product">Product</a>
              <a href="#benchmarks">Benchmarks</a>
              <a href="#linuxone">LinuxONE</a>
            </div>
            <div>
              <h3>Code</h3>
              <a href={RELEASE_URL} target="_blank" rel="noreferrer">Release branch ↗</a>
              <a href="https://github.com/beastgotfried/agent-jailbreak" target="_blank" rel="noreferrer">Repository ↗</a>
            </div>
          </div>
        </div>
        <div className="foot-base">
          <span>IBM Z Datathon · native s390x · CPU-only</span>
          <span>frozen release 8a86ce0 · synthetic benchmark evidence</span>
        </div>
      </div>
    </footer>
  )
}

export default function App() {
  useReveal()
  useEffect(() => {
    if (document.body.classList.contains('flat')) {
      document.querySelectorAll('.reveal').forEach((el) => el.classList.add('is-in'))
    }
  }, [])
  return (
    <>
      <a className="skip-link" href="#main-content">Skip to content</a>
      <Nav />
      <main id="main-content" tabIndex={-1}>
        <Hero />
        <Problem />
        <Product />
        <Benchmarks />
        <Recovery />
        <LinuxOne />
        <Evidence />
        <Datasets />
      </main>
      <Footer />
    </>
  )
}
