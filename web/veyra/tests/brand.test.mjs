import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync, readdirSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { resolve, dirname } from 'node:path'
import { streamResults, detection } from '../src/data.js'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const read = path => readFileSync(resolve(root, path), 'utf8')

test('browser title, descriptions, and visible React copy use Veyra', () => {
  assert.match(read('index.html'), /<title>Veyra<\/title>/)
  assert.match(read('src/App.jsx'), /aria-label="Veyra home"/)
  for (const path of ['index.html', 'src/App.jsx', 'src/styles.css', 'package.json']) {
    assert.doesNotMatch(read(path), /integrity[ -]watch/i)
  }
  assert.equal(JSON.parse(read('package.json')).name, 'veyra-web')
})

test('published chart assets carry no old product label', () => {
  for (const name of readdirSync(resolve(root, 'public/visuals'))) {
    if (name.endsWith('.svg')) assert.doesNotMatch(read(`public/visuals/${name}`), /integrity[ -]watch/i)
  }
})

test('brand change preserves the frozen benchmark counts', () => {
  assert.equal(streamResults.total, 30720)
  assert.equal(streamResults.A.correct, 23238)
  assert.equal(streamResults.B.correct, 25096)
  assert.equal(streamResults.B.harmfulAllowed, 595)
  assert.equal(streamResults.B.held, 1685)
  assert.equal(detection.confirmed, 55)
  assert.equal(detection.restored, 45)
  assert.equal(detection.held, 5)
})

test('smooth scroll retains its reduced-motion alternative', () => {
  const css = read('src/styles.css')
  assert.match(css, /scroll-behavior: smooth/)
  assert.match(css, /prefers-reduced-motion: reduce/)
  assert.match(css, /scroll-behavior: auto/)
})

test('static evidence supports keyboard access and stable image layout', () => {
  const app = read('src/App.jsx')
  const css = read('src/styles.css')
  assert.match(app, /className="skip-link" href="#main-content"/)
  assert.match(app, /<main id="main-content" tabIndex=\{-1\}/)
  assert.match(app, /aria-label="Complete-stream benchmark comparison" tabIndex=\{0\}/)
  assert.doesNotMatch(app, /<h[45]>/)
  assert.doesNotMatch(css, /transition:\s*all\b/)
  assert.match(css, /:focus-visible/)
  for (const image of app.matchAll(/<img\b[^>]+>/g)) {
    assert.match(image[0], /width=\{/)
    assert.match(image[0], /height=\{/)
    assert.match(image[0], /alt=/)
  }
})
