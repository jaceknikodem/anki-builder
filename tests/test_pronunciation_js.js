#!/usr/bin/env node
// Tests for the pronunciation card JS scoring logic.
// Run: node tests/test_pronunciation_js.js

// ── minimal DOM mock ──────────────────────────────────────────────────────────

class MockEl {
  constructor(className = '', text = '') {
    this.className = className;
    this._text     = text;
    this.children  = [];
    this._parent   = null;
  }
  get textContent() {
    return this.children.length ? this.children.map(c => c.textContent).join('') : this._text;
  }
  querySelectorAll(selector) {
    const classes = selector.split(',').map(s => s.trim().replace(/^\./, ''));
    const results = [];
    const walk = (el) => {
      if (classes.includes(el.className)) results.push(el);
      el.children.forEach(walk);
    };
    this.children.forEach(walk);
    return { forEach: (fn) => results.forEach(fn), length: results.length };
  }
  cloneNode(deep) {
    const clone = new MockEl(this.className, this._text);
    if (deep) {
      clone.children = this.children.map(c => {
        const cc = c.cloneNode(true);
        cc._parent = clone;
        return cc;
      });
    }
    return clone;
  }
  remove() {
    if (this._parent) {
      this._parent.children = this._parent.children.filter(c => c !== this);
    }
  }
}

function makeTypeans(...spans) {
  // spans: [{cls, text}, ...]
  const el = new MockEl('', '');
  for (const { cls, text } of spans) {
    const child = new MockEl(cls, text);
    child._parent = el;
    el.children.push(child);
  }
  return el;
}

// ── scoring functions (copied verbatim from _PRON_BACK_TMPL) ─────────────────

function lev(a, b) {
  var m = a.length, n = b.length, i, j;
  var dp = [];
  for (i = 0; i <= m; i++) { dp[i] = [i]; }
  for (j = 1; j <= n; j++) { dp[0][j] = j; }
  for (i = 1; i <= m; i++) {
    for (j = 1; j <= n; j++) {
      dp[i][j] = a[i-1] === b[j-1] ? dp[i-1][j-1]
        : 1 + Math.min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]);
    }
  }
  return dp[m][n];
}

function sim(a, b) {
  if (!a && !b) return 1;
  if (!a || !b) return 0;
  return 1 - lev(a, b) / Math.max(a.length, b.length);
}

function norm(s) {
  return s.trim().replace(/[ァ-ヶ]/g, function (c) {
    return String.fromCharCode(c.charCodeAt(0) - 0x60);
  });
}

function extractTyped(typeans) {
  if (!typeans) return '';
  var spans = typeans.querySelectorAll('.typeGood, .typeBad');
  if (spans.length > 0) {
    var t = ''; spans.forEach(function (el) { t += el.textContent; }); return t;
  }
  var clone = typeans.cloneNode(true);
  clone.querySelectorAll('.typeMissed').forEach(function (el) { el.remove(); });
  return clone.textContent.trim();
}

function score(userRaw, plainTarget, hiraganaTarget) {
  const userNorm = norm(userRaw);
  return Math.max(
    sim(userNorm, norm(plainTarget || '')),
    hiraganaTarget ? sim(userNorm, norm(hiraganaTarget)) : 0
  );
}

// ── test runner ───────────────────────────────────────────────────────────────

let passed = 0, failed = 0;

function assert(desc, actual, expected, tolerance = 0) {
  const ok = tolerance
    ? Math.abs(actual - expected) <= tolerance
    : actual === expected;
  if (ok) {
    console.log(`  ✓  ${desc}`);
    passed++;
  } else {
    console.error(`  ✗  ${desc}`);
    console.error(`       got: ${JSON.stringify(actual)}`);
    console.error(`  expected: ${JSON.stringify(expected)}`);
    failed++;
  }
}

// ── lev / sim ─────────────────────────────────────────────────────────────────

console.log('\nlev / sim');
assert('identical strings → distance 0',        lev('abc', 'abc'), 0);
assert('empty vs empty → distance 0',            lev('', ''),       0);
assert('one insertion',                          lev('ab', 'abc'),  1);
assert('one deletion',                           lev('abc', 'ab'),  1);
assert('one substitution',                       lev('abc', 'axc'), 1);
assert('sim: identical → 1',                     sim('abc', 'abc'), 1);
assert('sim: empty vs empty → 1',               sim('', ''),       1);
assert('sim: completely different → 0',          sim('abc', 'xyz'), 0);
assert('sim: one char off / 3',  sim('abc', 'axc'), 1 - 1/3, 0.001);

// ── norm ──────────────────────────────────────────────────────────────────────

console.log('\nnorm (katakana → hiragana + trim)');
assert('katakana ア→あ',  norm('ア'), 'あ');
assert('katakana ン→ん',  norm('ン'), 'ん');
assert('hiragana unchanged', norm('あいう'), 'あいう');
assert('kanji unchanged',    norm('食べる'), '食べる');
assert('mixed: カタカナ→かたかな', norm('カタカナ'), 'かたかな');
assert('trims whitespace',   norm('  あ  '), 'あ');
assert('STT often returns katakana for loanwords',
       norm('コーヒーが好きです'), 'こーひーが好きです');

// ── extractTyped ──────────────────────────────────────────────────────────────

console.log('\nextractTyped from #typeans');

// Standard Anki format: typeGood + typeBad + typeMissed
assert('all correct (typeGood only)',
  extractTyped(makeTypeans({cls:'typeGood', text:'食べます'})),
  '食べます');

assert('all wrong (typeBad only)',
  extractTyped(makeTypeans({cls:'typeBad', text:'たべます'})),
  'たべます');

assert('partial: good + bad',
  extractTyped(makeTypeans(
    {cls:'typeGood', text:'食べ'},
    {cls:'typeBad',  text:'る'},
    {cls:'typeMissed', text:'ます'},
  )),
  '食べる');

assert('user typed nothing → typeMissed only → empty string',
  extractTyped(makeTypeans({cls:'typeMissed', text:'食べます'})),
  '');

assert('fallback: no class spans, plain text',
  extractTyped(makeTypeans({cls:'', text:'食べます'})),
  '食べます');

assert('null element → empty string', extractTyped(null), '');

// ── end-to-end scoring ────────────────────────────────────────────────────────

console.log('\nend-to-end pronunciation scoring');

const SENTENCE_KANJI    = '毎朝コーヒーを飲みます';
const SENTENCE_HIRAGANA = 'まいあさこーひーをのみます';

// Perfect kanji match
assert('perfect kanji match → 100%',
  Math.round(score('毎朝コーヒーを飲みます', SENTENCE_KANJI, SENTENCE_HIRAGANA) * 100), 100);

// Perfect hiragana match (STT returns hiragana for kanji words)
assert('perfect hiragana → 100%',
  Math.round(score('まいあさこーひーをのみます', SENTENCE_KANJI, SENTENCE_HIRAGANA) * 100), 100);

// STT returns katakana for loanword, should still score well after norm
assert('katakana loanword normalises correctly',
  Math.round(score('まいあさコーヒーをのみます', SENTENCE_KANJI, SENTENCE_HIRAGANA) * 100), 100);

// One word wrong
const partial = score('毎朝コーヒーを食べます', SENTENCE_KANJI, SENTENCE_HIRAGANA);
assert('one wrong word → score between 60–90%', partial >= 0.60 && partial <= 0.90, true);

// Completely empty
assert('empty input → 0%',
  Math.round(score('', SENTENCE_KANJI, SENTENCE_HIRAGANA) * 100), 0);

// Completely wrong
assert('totally wrong input → low score',
  score('全然違います', SENTENCE_KANJI, SENTENCE_HIRAGANA) < 0.4, true);

// ── summary ───────────────────────────────────────────────────────────────────

console.log(`\n${passed + failed} tests: ${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
