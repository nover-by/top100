/* ============================================================
   Unbiased Top 100 — app.js
   ============================================================ */

const DATA_URL = "results.json";

// ── Danish alphabet letter → index 1..29 ──────────────────────────────────
const DANISH_ALPHA = [...'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'Æ', 'Ø', 'Å'];
const POS_TO_LETTER = Object.fromEntries(DANISH_ALPHA.map((l, i) => [i + 1, l]));

// ── Global state ─────────────────────────────────────────────────────────
let allSongs   = [];
let modelSummary = {};
let biasChart       = null;
let letterChart     = null;
let letterCountChart = null;

// ── Boot ──────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  setupTabs();
  setupControls();
  loadData();
});

// ── Tabs ──────────────────────────────────────────────────────────────────
function setupTabs() {
  const buttons = document.querySelectorAll('.tab-btn');
  const sections = {
    ranking: document.getElementById('tab-ranking'),
    bias:    document.getElementById('tab-bias'),
    about:   document.getElementById('tab-about'),
  };

  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab;
      buttons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      Object.values(sections).forEach(s => s.classList.add('hidden'));
      sections[tab].classList.remove('hidden');

      // Render charts lazily — only when bias tab is first opened
      if (tab === 'bias' && allSongs.length && !biasChart) {
        renderLetterCountChart();
        renderBiasChart();
        renderLetterChart();
        renderBetaSummary();
      }
    });
  });
}

// ── Controls ─────────────────────────────────────────────────────────────
function setupControls() {
  document.getElementById('search-input').addEventListener('input', renderTable);
  document.getElementById('sort-select').addEventListener('change', renderTable);
}

// ── Data loading ──────────────────────────────────────────────────────────
async function loadData() {
  try {
    const res = await fetch(DATA_URL);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    allSongs     = data.songs;
    modelSummary = data.summary;
    renderStats();
    renderTable();
  } catch (err) {
    document.getElementById('ranking-body').innerHTML =
      `<tr><td colspan="6" class="px-6 py-12 text-center text-red-400">
         Could not load data: ${err.message}<br>
         <span class="text-slate-400 text-xs">Run <code>python model.py</code> to generate results.json</span>
       </td></tr>`;
    console.error(err);
  }
}

// ── Stats bar ─────────────────────────────────────────────────────────────
function renderStats() {
  const bigMovers = allSongs.filter(s => Math.abs(s.rank_delta) >= 5);
  const { beta_mean, prob_negative } = modelSummary;
  const pct = prob_negative != null ? (prob_negative * 100).toFixed(0) + '%' : '—';

  document.getElementById('stats-bar').innerHTML = `
    <div class="bg-white border border-slate-200 shadow-sm rounded-xl px-6 py-3 text-center">
      <div class="text-2xl font-display font-bold text-slate-900">${allSongs.length}</div>
      <div class="text-xs text-slate-500 mt-0.5">Songs analysed</div>
    </div>
    <div class="bg-white border border-slate-200 shadow-sm rounded-xl px-6 py-3 text-center">
      <div class="text-2xl font-display font-bold ${prob_negative > 0.8 ? 'text-brand' : 'text-slate-700'}">${pct}</div>
      <div class="text-xs text-slate-500 mt-0.5">P(bias exists)</div>
    </div>
    <div class="bg-white border border-slate-200 shadow-sm rounded-xl px-6 py-3 text-center">
      <div class="text-2xl font-display font-bold text-amber-600">${bigMovers.length}</div>
      <div class="text-xs text-slate-500 mt-0.5">Big movers (≥5 places)</div>
    </div>
  `;
}

// ── Ranking table ─────────────────────────────────────────────────────────
function renderTable() {
  const query  = document.getElementById('search-input').value.trim().toLowerCase();
  const sortBy = document.getElementById('sort-select').value;

  let songs = [...allSongs];

  // Filter
  if (query) {
    songs = songs.filter(s =>
      s.song_title.toLowerCase().includes(query) ||
      s.artist.toLowerCase().includes(query)
    );
  }

  // Sort
  switch (sortBy) {
    case 'biased':    songs.sort((a, b) => a.rank_biased - b.rank_biased);    break;
    case 'delta_up':  songs.sort((a, b) => b.rank_delta  - a.rank_delta);     break;
    case 'delta_down':songs.sort((a, b) => a.rank_delta  - b.rank_delta);     break;
    default:          songs.sort((a, b) => a.rank_unbiased - b.rank_unbiased);break;
  }

  const tbody = document.getElementById('ranking-body');
  if (!songs.length) {
    tbody.innerHTML = `<tr><td colspan="6" class="px-6 py-10 text-center text-slate-500">No results</td></tr>`;
    return;
  }

  tbody.innerHTML = songs.map((s, idx) => {
    const delta = s.rank_delta;
    const absDelta = Math.abs(delta);
    const isBigUp   = delta >= 5;
    const isBigDown = delta <= -5;

    // Badge
    let badge;
    if (delta > 0) {
      badge = `<span class="badge-up inline-flex items-center gap-0.5 text-xs font-medium px-2 py-0.5 rounded-full">▲ +${delta}</span>`;
    } else if (delta < 0) {
      badge = `<span class="badge-down inline-flex items-center gap-0.5 text-xs font-medium px-2 py-0.5 rounded-full">▼ ${delta}</span>`;
    } else {
      badge = `<span class="badge-same inline-flex items-center text-xs font-medium px-2 py-0.5 rounded-full">— 0</span>`;
    }

    // Row highlight class
    const rowClass = isBigUp ? 'big-mover-up' : isBigDown ? 'big-mover-down' : '';

    // Bias effect pill (how many log-vote-units alphabet gave)
    const biasSign = s.bias_effect >= 0 ? '+' : '';
    const biasPill = `<span class="text-xs text-slate-500 font-mono">${biasSign}${s.bias_effect.toFixed(3)}</span>`;

    // Alternate row BG
    const bg = idx % 2 === 0 ? '' : 'bg-slate-50';

    return `<tr class="song-row ${rowClass} ${bg}">
      <td class="px-4 py-3 font-mono text-slate-900 font-semibold">${s.rank_unbiased}</td>
      <td class="px-4 py-3 font-mono text-slate-400">${s.rank_biased}</td>
      <td class="px-4 py-3">
        <div class="font-medium text-slate-800 leading-snug">${escHtml(s.song_title)}</div>
        <div class="text-xs text-slate-500 sm:hidden mt-0.5">${escHtml(s.artist)}</div>
      </td>
      <td class="px-4 py-3 text-slate-600 hidden sm:table-cell">${escHtml(s.artist)}</td>
      <td class="px-4 py-3 text-center hidden sm:table-cell">
        <span class="inline-block bg-slate-100 text-slate-600 text-xs font-mono rounded px-2 py-0.5">${escHtml(s.alpha_letter ?? '')} (${s.alpha_pos})</span>
      </td>
      <td class="px-4 py-3 text-center">${badge}</td>
    </tr>`;
  }).join('');
}

// ── Letter count histogram ───────────────────────────────────────────────
function renderLetterCountChart() {
  const ctx = document.getElementById('letter-count-chart').getContext('2d');

  // Count songs per initial letter, preserving alphabet order
  const counts = {};
  DANISH_ALPHA.forEach(l => { counts[l] = 0; });
  allSongs.forEach(s => {
    const letter = s.alpha_letter ?? POS_TO_LETTER[s.alpha_pos] ?? '?';
    if (counts[letter] !== undefined) counts[letter]++;
  });

  // Only include letters that appear at least once
  const labels = DANISH_ALPHA.filter(l => counts[l] > 0);
  const values = labels.map(l => counts[l]);

  // Colour bars by position: earlier letters get a warmer tint
  const maxPos = DANISH_ALPHA.length;
  const barColors = labels.map(l => {
    const pos = DANISH_ALPHA.indexOf(l) + 1;
    const t = 1 - (pos - 1) / (maxPos - 1);  // 1 at A, 0 at Å
    // Interpolate indigo → slate
    const r = Math.round(99  + t * (220 - 99));
    const g = Math.round(102 + t * (55  - 102));
    const b = Math.round(241 + t * (60  - 241));
    return `rgba(${r},${g},${b},0.75)`;
  });

  // Expected count if perfectly uniform (100 songs / letters-that-appear)
  const expected = 100 / labels.length;

  letterCountChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Songs in Top 100',
          data: values,
          backgroundColor: barColors,
          borderRadius: 4,
          order: 2,
        },
        {
          label: 'Expected (uniform)',
          data: labels.map(() => expected),
          type: 'line',
          borderColor: '#e63946',
          borderWidth: 1.5,
          borderDash: [4, 3],
          pointRadius: 0,
          fill: false,
          order: 1,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: '#64748b', font: { size: 11 } } },
        tooltip: {
          callbacks: {
            title: ctx => `Letter: ${ctx[0].label}`,
            label: ctx => ctx.datasetIndex === 0
              ? `${ctx.parsed.y} song${ctx.parsed.y !== 1 ? 's' : ''} in Top 100`
              : `Expected if uniform: ${expected.toFixed(1)}`,
          },
          backgroundColor: '#ffffff',
          titleColor: '#0f172a',
          bodyColor: '#64748b',
          borderColor: '#e2e8f0',
          borderWidth: 1,
        },
      },
      scales: {
        x: {
          title: { display: true, text: 'Artist initial letter', color: '#94a3b8' },
          ticks: { color: '#64748b' },
          grid: { display: false },
        },
        y: {
          title: { display: true, text: 'Number of songs', color: '#94a3b8' },
          ticks: { color: '#94a3b8', stepSize: 1 },
          grid: { color: 'rgba(226,232,240,0.8)' },
          beginAtZero: true,
        },
      },
    },
  });
}

// ── Bias scatter chart ────────────────────────────────────────────────────
function renderBiasChart() {
  const ctx = document.getElementById('bias-chart').getContext('2d');

  const points = allSongs.map(s => ({
    x: s.alpha_pos,
    y: s.rank_biased,   // lower = better
    label: s.song_title,
    artist: s.artist,
  }));

  // Simple linear regression (alpha_pos vs votes proxy = 101 - rank_biased)
  const xs = allSongs.map(s => s.alpha_pos);
  const ys = allSongs.map(s => 101 - s.rank_biased);
  const { slope, intercept } = linearRegression(xs, ys);

  const xMin = 1, xMax = 29;
  const regLine = [
    { x: xMin, y: slope * xMin + intercept },
    { x: xMax, y: slope * xMax + intercept },
  ];

  biasChart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Songs',
          data: allSongs.map(s => ({ x: s.alpha_pos, y: 101 - s.rank_biased })),
          backgroundColor: 'rgba(99,102,241,0.45)',
          pointRadius: 5,
          pointHoverRadius: 7,
        },
        {
          label: 'Regression line (β)',
          data: regLine,
          type: 'line',
          borderColor: '#e63946',
          borderWidth: 2,
          pointRadius: 0,
          fill: false,
          tension: 0,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: '#64748b', font: { size: 11 } } },
        tooltip: {
          callbacks: {
            label: ctx => {
              if (ctx.datasetIndex === 1) return null;
              const s = allSongs[ctx.dataIndex];
              return [`${s.song_title}`, `${s.artist}`, `Pos: ${s.alpha_pos}  Votes: ${101 - s.rank_biased}`];
            },
          },
          backgroundColor: '#ffffff',
          titleColor: '#0f172a',
          bodyColor: '#64748b',
          borderColor: '#e2e8f0',
          borderWidth: 1,
        },
      },
      scales: {
        x: {
          title: { display: true, text: 'Alphabetical position (1=A … 29=Å)', color: '#94a3b8' },
          ticks: { color: '#94a3b8', stepSize: 2,
            callback: v => POS_TO_LETTER[v] ?? v,
          },
          grid: { color: 'rgba(226,232,240,0.8)' },
        },
        y: {
          title: { display: true, text: 'Vote proxy (101 − rank)', color: '#94a3b8' },
          ticks: { color: '#94a3b8' },
          grid: { color: 'rgba(226,232,240,0.8)' },
        },
      },
    },
  });
}

// ── Letter-group bar chart ────────────────────────────────────────────────
function renderLetterChart() {
  const ctx = document.getElementById('letter-chart').getContext('2d');

  // Group by alpha_pos, compute mean rank_delta per group
  const groups = {};
  allSongs.forEach(s => {
    const letter = s.alpha_letter ?? POS_TO_LETTER[s.alpha_pos] ?? '?';
    if (!groups[letter]) groups[letter] = { deltas: [], pos: s.alpha_pos };
    groups[letter].deltas.push(s.rank_delta);
  });

  // Sort by alphabetical position
  const sorted = Object.entries(groups).sort((a, b) => a[1].pos - b[1].pos);
  const labels = sorted.map(([l]) => l);
  const values = sorted.map(([, g]) => {
    const mean = g.deltas.reduce((a, b) => a + b, 0) / g.deltas.length;
    return parseFloat(mean.toFixed(2));
  });

  const colors = values.map(v =>
    v > 0 ? 'rgba(74,222,128,0.7)' : 'rgba(248,113,113,0.7)'
  );

  letterChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Avg. rank change (positive = rose)',
        data: values,
        backgroundColor: colors,
        borderRadius: 3,
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => {
              const v = ctx.parsed.y;
              return `Avg. change: ${v > 0 ? '+' : ''}${v}`;
            },
          },
          backgroundColor: '#ffffff',
          bodyColor: '#64748b',
          borderColor: '#e2e8f0',
          borderWidth: 1,
        },
      },
      scales: {
        x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(226,232,240,0.6)' } },
        y: {
          title: { display: true, text: 'Avg. rank change', color: '#94a3b8' },
          ticks: { color: '#94a3b8' },
          grid: { color: 'rgba(226,232,240,0.8)' },
        },
      },
    },
  });
}

// ── Beta summary cards ─────────────────────────────────────────────────────
function renderBetaSummary() {
  const { beta_mean, beta_sd, beta_hdi_low, beta_hdi_high, beta_raw, prob_negative, spearman_rho, spearman_p } = modelSummary;

  // Confidence label
  const pct = (prob_negative * 100).toFixed(0);
  let evidenceLabel, evidenceClass;
  if (prob_negative > 0.95) {
    evidenceLabel = 'Strong evidence of bias';
    evidenceClass = 'text-red-600 bg-red-50 border-red-200';
  } else if (prob_negative > 0.80) {
    evidenceLabel = 'Moderate evidence of bias';
    evidenceClass = 'text-amber-700 bg-amber-50 border-amber-200';
  } else {
    evidenceLabel = 'Weak / inconclusive evidence';
    evidenceClass = 'text-slate-600 bg-slate-50 border-slate-200';
  }

  const hdiSpansZero = beta_hdi_low < 0 && beta_hdi_high > 0;

  document.getElementById('beta-summary').innerHTML = `
    <div class="bg-slate-50 rounded-xl p-4 border border-slate-200">
      <div class="text-xs text-slate-500 mb-1">β (standardised slope)</div>
      <div class="text-2xl font-display font-bold ${beta_mean < 0 ? 'text-brand' : 'text-green-600'}">${beta_mean?.toFixed(4)}</div>
      <div class="text-xs text-slate-400 mt-1">94% HDI [${beta_hdi_low?.toFixed(3)}, ${beta_hdi_high?.toFixed(3)}]${hdiSpansZero ? ' — <span class="text-amber-600 font-medium">spans zero</span>' : ''}</div>
    </div>
    <div class="bg-slate-50 rounded-xl p-4 border border-slate-200">
      <div class="text-xs text-slate-500 mb-1">P(β &lt; 0)</div>
      <div class="text-2xl font-display font-bold text-slate-800">${pct}%</div>
      <div class="text-xs text-slate-400 mt-1">Spearman ρ = ${spearman_rho?.toFixed(3)}, p = ${spearman_p?.toFixed(3)}</div>
    </div>
    <div class="rounded-xl p-4 border ${evidenceClass}">
      <div class="text-xs font-medium mb-1">Verdict</div>
      <div class="text-sm font-semibold mb-1">${evidenceLabel}</div>
      <div class="text-xs leading-snug opacity-80">
        ${hdiSpansZero
          ? `The 94% credible interval spans zero — we cannot rule out β = 0. The alphabetical effect is visible in <em>selection</em> (the histogram), but is too weak to correct confidently within the ranked top 100.`
          : `The credible interval excludes zero. The corrected ranking adjusts for the estimated alphabetical advantage of ${Math.abs(beta_raw * 100).toFixed(1)}% per letter.`
        }
      </div>
    </div>`;
}

// ── Utilities ─────────────────────────────────────────────────────────────
function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function linearRegression(xs, ys) {
  const n = xs.length;
  const xMean = xs.reduce((a, b) => a + b, 0) / n;
  const yMean = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    num += (xs[i] - xMean) * (ys[i] - yMean);
    den += (xs[i] - xMean) ** 2;
  }
  const slope = den ? num / den : 0;
  return { slope, intercept: yMean - slope * xMean };
}
