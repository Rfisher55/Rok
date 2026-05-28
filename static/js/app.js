/* ===== ROK — Stock Intelligence Dashboard ===== */

let currentData = null;

async function loadAnalysis() {
  try {
    const res = await fetch('/api/latest');
    const data = await res.json();
    if (!data || !data.id) {
      showEmptyState();
      return;
    }
    currentData = data;
    renderDashboard(data);
  } catch (e) {
    console.error('ROK load failed:', e);
    showEmptyState();
  }
}

async function loadStatus() {
  try {
    const res = await fetch('/api/status');
    const status = await res.json();
    const el = document.getElementById('footerStatus');
    if (el) {
      const key = status.anthropic_key_set ? '✓ AI Active' : '⚠ Demo Mode — Add API Key';
      const tw = status.twitter_enabled ? ' · Twitter ✓' : '';
      el.textContent = `${key}${tw} · ${status.total_analyses} analyses total`;
    }
  } catch (_) {}
}

async function triggerRefresh() {
  const btn = document.getElementById('refreshBtn');
  btn.classList.add('loading');
  showToast('ROK is collecting and analyzing markets... check back in ~90 seconds.');
  try {
    await fetch('/api/refresh', { method: 'POST' });
    setTimeout(async () => {
      await loadAnalysis();
      await loadStatus();
      btn.classList.remove('loading');
      showToast('✓ Analysis updated.');
    }, 90000);
  } catch (e) {
    btn.classList.remove('loading');
    showToast('Refresh failed — check that the server is running.');
  }
}

function renderDashboard(data) {
  setEl('analysisDate', `Last updated: ${data.analysis_date}`);

  // Sentiment badge
  const badge = document.getElementById('sentimentBadge');
  if (badge) {
    badge.textContent = data.market_sentiment || '—';
    badge.className = `sentiment-badge ${data.market_sentiment || ''}`;
  }

  // ROK message
  if (data.rok_message) {
    const bar = document.getElementById('rokMessageBar');
    const txt = document.getElementById('rokMessageText');
    if (bar && txt) { txt.textContent = data.rok_message; bar.style.display = 'flex'; }
  }

  // Week summary
  setEl('weekSummary', data.week_summary || '');

  // Stats
  const stats = data.source_stats || {};
  setEl('statReddit', fmt(stats.reddit));
  setEl('statNews', fmt(stats.news));
  setEl('statStocks', fmt(stats.stocks));
  setEl('statSec', fmt(stats.sec));
  setEl('statEarnings', fmt(stats.earnings_upcoming));
  setEl('statOptions', fmt(stats.unusual_options));

  // Market indices
  renderIndices(data.market_indices || {});

  // Fear/Greed gauge (CNN score 0-100 → our 1-10 scale)
  const fg = data.fear_greed || {};
  const fgScore = fg.score || 50;
  const ourScore = data.sentiment_score || Math.round(fgScore / 10);
  const gaugeFill = document.getElementById('gaugeFill');
  const gaugeScore = document.getElementById('gaugeScore');
  if (gaugeFill) {
    const pct = Math.min(Math.max((fgScore / 100) * 100, 2), 98);
    gaugeFill.style.left = `calc(${pct}% - 9px)`;
  }
  if (gaugeScore) {
    gaugeScore.textContent = fg.rating
      ? `${fgScore}/100 — ${fg.rating}`
      : `${ourScore}/10`;
  }

  // Social sentiment bars
  const agg = data.aggregate_sentiment || {};
  animateBar('bullBar', 'bullPct', agg.bullish_pct || 0);
  animateBar('bearBar', 'bearPct', agg.bearish_pct || 0);
  animateBar('neutBar', 'neutPct', agg.neutral_pct || 0);

  // Sector heat
  const heat = data.sector_heat || {};
  if (heat.hottest || heat.coldest) {
    const row = document.getElementById('sectorRow');
    if (row) row.style.display = 'grid';
    setEl('sectorHot', heat.hottest || '');
    setEl('sectorCold', heat.coldest || '');
  }

  // Trending tickers
  if (data.ticker_mentions && data.ticker_mentions.length) {
    const sec = document.getElementById('trendingSection');
    if (sec) sec.style.display = 'block';
    const tape = document.getElementById('tickerTape');
    if (tape) {
      tape.innerHTML = data.ticker_mentions.slice(0, 24).map(([t, c]) =>
        `<div class="ticker-chip">$${esc(t)}<span class="chip-count">${c > 0 ? c : ''}</span></div>`
      ).join('');
    }
  }

  // Buy/Sell/Watch
  const buys = data.buy_signals || [];
  const sells = data.sell_signals || [];
  const watches = data.watch_list || [];
  setEl('buyCount', `${buys.length} signal${buys.length !== 1 ? 's' : ''} identified`);
  setEl('sellCount', `${sells.length} signal${sells.length !== 1 ? 's' : ''} identified`);
  setEl('watchCount', `${watches.length} position${watches.length !== 1 ? 's' : ''} on radar`);
  renderBuySellCards('buyCards', buys, 'buy');
  renderBuySellCards('sellCards', sells, 'sell');
  renderWatchCards('watchCards', watches);

  // Short squeeze alerts
  const squeezes = data.short_squeeze_alerts || [];
  if (squeezes.length) {
    const sec = document.getElementById('squeezeSection');
    if (sec) sec.style.display = 'block';
    const grid = document.getElementById('squeezeCards');
    if (grid) {
      grid.innerHTML = squeezes.map((s, i) => `
        <div class="squeeze-card" style="animation-delay:${i*0.08}s">
          <div class="squeeze-ticker">$${esc(s.ticker || '')}</div>
          <div class="squeeze-float">Short Float: <strong>${esc(s.short_float || 'n/a')}</strong></div>
          <div class="squeeze-setup">${esc(s.setup || '')}</div>
        </div>`).join('');
    }
  }

  // Earnings plays
  const earnings = data.earnings_plays || [];
  if (earnings.length) {
    const sec = document.getElementById('earningsSection');
    if (sec) sec.style.display = 'block';
    const grid = document.getElementById('earningsCards');
    if (grid) {
      grid.innerHTML = earnings.map((e, i) => `
        <div class="earnings-card" style="animation-delay:${i*0.08}s">
          <div class="earnings-ticker">$${esc(e.ticker || '')}</div>
          <div class="earnings-date">${esc(e.earnings_date || '')}</div>
          <span class="earnings-direction ${esc(e.direction || '')}">${esc(e.direction || 'WATCH')}</span>
          <div class="earnings-play">${esc(e.play || '')}</div>
        </div>`).join('');
    }
  }

  // Notable trends
  const trends = data.notable_trends || [];
  if (trends.length) {
    const sec = document.getElementById('trendsSection');
    if (sec) sec.style.display = 'block';
    const list = document.getElementById('trendsList');
    if (list) {
      list.innerHTML = trends.map((t, i) =>
        `<div class="trend-item"><span class="trend-num">${String(i+1).padStart(2,'0')}</span>${esc(t)}</div>`
      ).join('');
    }
  }

  document.title = `ROK — ${data.market_sentiment || 'Markets'}`;
}

function renderIndices(indices) {
  const bar = document.getElementById('indicesBar');
  if (!bar || !Object.keys(indices).length) return;
  bar.innerHTML = Object.entries(indices).map(([name, d]) => {
    const chg = d.change_pct || 0;
    const cls = chg >= 0 ? 'up' : 'down';
    const sign = chg >= 0 ? '+' : '';
    return `<div class="idx-item">
      <span class="idx-name">${esc(name)}</span>
      <span class="idx-price">$${d.price}</span>
      <span class="idx-chg ${cls}">${sign}${chg.toFixed(2)}%</span>
    </div>`;
  }).join('');
}

function renderBuySellCards(containerId, signals, type) {
  const container = document.getElementById(containerId);
  if (!container) return;
  if (!signals.length) {
    container.innerHTML = `<div class="loading-card" style="opacity:0.5">No ${type} signals in this analysis</div>`;
    return;
  }
  container.innerHTML = signals.map((s, i) => {
    const strength = s.signal_strength || 0;
    const strengthLabel = strength >= 8 ? 'STRONG' : strength >= 6 ? 'MODERATE' : 'WEAK';

    let priceRow = '';
    if (type === 'buy' && (s.current_price || s.price_target)) {
      priceRow = `<div class="card-price-row">
        ${s.current_price ? `<span class="price-current">$${s.current_price.toFixed(2)}</span>` : ''}
        ${s.price_target ? `<span class="price-target"><span class="target-arrow">→</span> $${s.price_target.toFixed(2)}</span>` : ''}
        ${s.time_horizon ? `<span class="time-horizon">${esc(s.time_horizon)}</span>` : ''}
      </div>`;
    } else if (type === 'sell' && s.current_price) {
      const urgClass = (s.urgency || '').replace(' ', '-');
      priceRow = `<div class="card-price-row">
        <span class="price-current">$${s.current_price.toFixed(2)}</span>
        ${s.urgency ? `<span class="urgency-badge ${urgClass}">${esc(s.urgency)}</span>` : ''}
      </div>`;
    }

    const catalyst = s.catalyst
      ? `<div class="catalyst-banner">${esc(s.catalyst)}</div>` : '';

    const signals_chips = (s.data_signals || []).map(sig =>
      `<span class="sig-chip ${esc(sig)}">${esc(sig)}</span>`).join('');
    const dataSignals = signals_chips
      ? `<div class="data-signals">${signals_chips}</div>` : '';

    const reasons = (s.reasons || []).map(r => `<li>${esc(r)}</li>`).join('');
    const rokTake = s.rok_take ? `<div class="rok-take">${esc(s.rok_take)}</div>` : '';

    return `<div class="signal-card ${type}" style="animation-delay:${i*0.08}s">
      <div class="card-header">
        <div class="card-ticker-block">
          <div class="card-ticker">$${esc(s.ticker || '')}</div>
          <div class="card-company">${esc(s.company || '')}</div>
        </div>
        <div class="card-badges">
          <span class="strength-badge">${strengthLabel} · ${strength}/10</span>
          ${s.risk_level ? `<span class="risk-badge ${esc(s.risk_level)}">${esc(s.risk_level)}</span>` : ''}
        </div>
      </div>
      ${dataSignals}
      ${priceRow}
      ${catalyst}
      ${reasons ? `<ul class="card-reasons">${reasons}</ul>` : ''}
      ${rokTake}
    </div>`;
  }).join('');
}

function renderWatchCards(containerId, watches) {
  const container = document.getElementById(containerId);
  if (!container) return;
  if (!watches.length) {
    container.innerHTML = `<div class="loading-card" style="opacity:0.5">No watch list items</div>`;
    return;
  }
  container.innerHTML = watches.map((w, i) => `
    <div class="signal-card watch" style="animation-delay:${i*0.08}s">
      <div class="card-header">
        <div class="card-ticker-block">
          <div class="card-ticker">$${esc(w.ticker || '')}</div>
          <div class="card-company">${esc(w.company || '')}</div>
        </div>
        <div class="card-badges"><span class="strength-badge">WATCHING</span></div>
      </div>
      <div class="rok-take" style="margin-bottom:10px">${esc(w.why_watching || '')}</div>
      <div class="watch-trigger">
        <div class="watch-trigger-label">BUY TRIGGER</div>
        <div class="watch-trigger-text">${esc(w.trigger || '')}</div>
      </div>
      ${w.risk ? `<div class="watch-risk">${esc(w.risk)}</div>` : ''}
      ${w.potential ? `<div class="watch-potential">⬆ Potential: +${w.potential}%</div>` : ''}
    </div>`).join('');
}

function showEmptyState() {
  ['buyCards', 'sellCards', 'watchCards'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.innerHTML = `<div class="loading-card">ROK is running its first analysis — check back in ~2 minutes or hit REFRESH.</div>`;
  });
  setEl('weekSummary', 'Collecting market intelligence from Reddit, news feeds, SEC EDGAR, and Yahoo Finance...');
  setEl('analysisDate', 'Initializing ROK...');
}

function showToast(msg) {
  const toast = document.getElementById('toast');
  if (!toast) return;
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 6000);
}

function animateBar(barId, pctId, value) {
  setTimeout(() => {
    const bar = document.getElementById(barId);
    const pct = document.getElementById(pctId);
    if (bar) bar.style.width = `${value}%`;
    if (pct) pct.textContent = `${value}%`;
  }, 300);
}

function setEl(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function fmt(n) {
  if (n == null) return '—';
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n);
}

function esc(str) {
  return String(str || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// Poll for new data every 5 minutes
function startAutoRefresh() {
  setInterval(async () => {
    try {
      const res = await fetch('/api/status');
      const status = await res.json();
      if (!status || !currentData) return;
      const lastRun = status.last_run;
      if (lastRun && currentData.analysis_date && lastRun !== currentData._raw_last_run) {
        await loadAnalysis();
      }
    } catch (_) {}
  }, 5 * 60 * 1000);
}

document.addEventListener('DOMContentLoaded', () => {
  loadAnalysis();
  loadStatus();
  startAutoRefresh();
});
