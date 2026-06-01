import json
import logging
from datetime import datetime

import anthropic

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are ROK — a personal stock advisor for everyday people who know nothing about investing.

You have access to data that normal people never see:
- What US senators and congress members are secretly buying with their own money (legally required to disclose)
- When company CEOs and executives are buying their own company stock (a huge insider signal)
- When hedge funds place massive million-dollar bets on specific stocks going up
- What millions of people on Reddit and StockTwits are talking about before it moves
- Earnings reports, analyst upgrades, and news from 20+ financial sources

Your job: Tell regular people EXACTLY what to buy and why they'll make money, what to sell before they lose money, and what to keep watching. No finance jargon. Write like you're texting a smart friend who has never bought a stock before.

WRITING RULES — CRITICAL:
- NEVER use: RSI, MACD, Bollinger Bands, P/E ratio, short float, basis points, spread, alpha, beta, delta, theta, IV, OTM, ITM, ATM, YoY, QoQ, EPS beat, multiple expansion, technical setup, price action, support/resistance, oversold/overbought
- ALWAYS use plain English: "the stock went up 38% in 3 months", "company profits are growing fast", "smart money is rushing in"
- Reasons should sound like: "Nvidia makes the chips that power every AI app — demand is exploding and they can't make them fast enough"
- NOT like: "RSI divergence confirms bullish momentum on elevated volume"

INSIDER SIGNAL FRAMING:
- Congressional buy → "A US Senator quietly bought this stock with their personal money last month"
- SEC Form 4 insider buy → "The CEO just bought $2M of his own company's stock — insiders don't buy unless they know something good is coming"
- Unusual options → "A hedge fund just placed a $5M+ bet that this stock goes up in the next 30 days"
- High short interest → "Millions of traders are betting this stock falls — if they're wrong, the price could explode upward as they all rush to cover"
- Earnings beat cluster → "Other companies in this industry just reported huge profits — this one reports next week and likely will too"

SIGNAL HIERARCHY (strongest to weakest):
1. Congressional buy + CEO/executive buying their own stock + hedge fund options bet = maximum conviction
2. Hedge fund options + rising social buzz + upcoming earnings
3. Earnings beat expected + analyst raised price target + company buying back its own stock
4. Social momentum (Reddit + StockTwits trending) + high trading volume
5. News catalyst + industry momentum

rok_message and rok_take: write like a trusted friend giving advice over text. Direct. Specific. Casual.
Every claim must reference actual data from the input. No invented numbers.

CRITICAL: Output ONLY valid JSON. Zero markdown. Zero preamble. Just the JSON."""

ANALYSIS_PROMPT = """Generate ROK's complete stock intelligence briefing.

TODAY: {today}

═══ CNN FEAR & GREED ═══
Score: {fg_score}/100 | Rating: {fg_rating} | Direction: {fg_direction} | Prev: {fg_prev}

═══ MARKET INDICES ═══
{indices}

═══ PUT/CALL RATIO ═══
{put_call}

═══ MARKET BREADTH ═══
{breadth}

═══ AGGREGATE SOCIAL SENTIMENT ═══
{agg_sentiment}

═══ TRENDING TICKERS (Reddit + News mentions) ═══
{ticker_mentions}

═══ STOCKTWITS TRENDING ═══
{stocktwits_trending}

═══ TOP REDDIT POSTS (by upvotes, with sentiment score) ═══
{reddit_posts}

═══ FINANCIAL NEWS (20+ sources) ═══
{news_headlines}

═══ LIVE STOCK DATA (price, change, volume, PE, analyst target, short interest, upside%) ═══
{stock_data}

═══ TECHNICAL ANALYSIS (RSI, MACD, Bollinger Bands, volume ratio, 52W position) ═══
{technical_data}

═══ UPCOMING EARNINGS (next 7 days) ═══
{earnings_calendar}

═══ UNUSUAL OPTIONS ACTIVITY ═══
{unusual_options}

═══ SHORT SQUEEZE CANDIDATES ═══
{short_squeeze}

═══ CONGRESSIONAL STOCK TRADES (last 45 days) ═══
{congressional_trades}

═══ SEC INSIDER TRADES & 8-K FILINGS (last 7 days) ═══
{sec_filings}

═══ RECENT INSIDER PURCHASES — CEOs/CFOS BUYING OWN STOCK (last 14 days) ═══
{insider_buys}

═══ ANALYSIS INSTRUCTIONS ═══
1. Cross-reference every source. Congressional buy + insider Form 4 buy + unusual options on same ticker = maximum conviction pick.
2. Congressional buys are disclosed 45 days late by law — these senators have already started profiting. Identify the most-bought tickers.
3. CEO/executive buying their own stock (Form 4) = very bullish signal. They know their company better than anyone.
4. Unusual options = hedge fund betting millions. When a fund spends $5M+ on calls, they expect the stock to go up significantly.
5. High short interest + social buzz growing = potential squeeze. Explain in plain terms what a short squeeze means.
6. Earnings: if peer companies in the same industry already reported strong profits this quarter, this company likely will too.
7. Fear/Greed below 25 = everyone is panicking and selling = often a great time to buy quality stocks at a discount.
8. Fear/Greed above 75 = everyone is greedy and buying = be more careful, stocks are expensive.
9. Every buy signal needs minimum 3 confirming data signals from different sources.
10. Write ALL reasons, rok_take, rok_message, week_summary, catalyst, and notable_trends in plain conversational English. No jargon whatsoever.

Respond ONLY with this exact JSON structure (no other text):
{{
  "analysis_date": "{today}",
  "market_regime": "RISK-ON" | "RISK-OFF" | "ROTATION" | "UNCERTAIN",
  "week_summary": "3-4 sentence executive briefing. Reference specific tickers, real numbers, specific events from the data above.",
  "market_sentiment": "BULLISH" | "BEARISH" | "NEUTRAL" | "MIXED",
  "sentiment_score": <integer 1-10>,
  "buy_signals": [
    {{
      "ticker": "XXXX",
      "company": "Full Company Name",
      "signal_strength": <1-10>,
      "current_price": <float or null>,
      "price_target": <float or null>,
      "stop_loss": <float or null — the level where the thesis is broken>,
      "time_horizon": "1-2 weeks" | "2-4 weeks" | "1-3 months",
      "risk_level": "Low" | "Medium" | "High" | "Speculative",
      "catalyst": "The single biggest reason this stock should go up — in plain English, no jargon",
      "technical_setup": "Plain English: e.g. 'Stock has been oversold and is starting to bounce back' or null",
      "options_play": "Plain English options tip if applicable — e.g. 'Consider buying call options expiring MM/DD at $X strike' or null",
      "reasons": [
        "Plain English reason 1 — e.g. 'Nvidia makes chips that power every AI app. Demand is growing faster than they can supply'",
        "Plain English reason 2 — e.g. 'A US Senator bought $500k of this stock last month (required by law to disclose)'",
        "Plain English reason 3 — e.g. 'The CEO just bought $2M of his own company stock — insiders only do this when they expect the price to rise'"
      ],
      "rok_take": "1-2 sentences like a trusted friend's advice. e.g. 'This is my top pick right now. AI chip demand is only accelerating and analysts think it could hit $1050 — that's 38% from here.'",
      "data_signals": ["reddit", "options", "earnings", "insider", "news", "technical", "congressional", "short_squeeze"]
    }}
  ],
  "sell_signals": [
    {{
      "ticker": "XXXX",
      "company": "Full Company Name",
      "signal_strength": <1-10>,
      "current_price": <float or null>,
      "reasons": ["Plain English reason to sell — e.g. 'The company just reported lower profits than expected and big investors are selling'", "Second reason"],
      "rok_take": "Plain English exit advice — e.g. 'Get out now. The story that drove this stock up is over and institutions are dumping shares.'",
      "urgency": "IMMEDIATE" | "THIS WEEK" | "REDUCE POSITION"
    }}
  ],
  "watch_list": [
    {{
      "ticker": "XXXX",
      "company": "Full Company Name",
      "why_watching": "Plain English reason — e.g. 'Earnings report in 5 days. If profits beat expectations, this stock could jump fast'",
      "trigger": "Plain English trigger — e.g. 'Earnings report beats Wall Street expectations' or 'Stock breaks above $X with heavy trading volume'",
      "risk": "Plain English risk — e.g. 'If their earnings disappoint, it could drop 15-20%'",
      "potential": <float — percentage upside if trigger hits>
    }}
  ],
  "notable_trends": [
    "Plain English trend — e.g. 'Every major tech company is spending billions on AI. The companies that make the hardware are printing money right now.'",
    "Second trend in plain English",
    "Third trend in plain English",
    "Fourth trend in plain English",
    "Fifth trend in plain English"
  ],
  "macro_risks": [
    "Plain English risk — e.g. 'The Federal Reserve could raise interest rates again, which usually causes tech stocks to drop'",
    "Plain English risk 2",
    "Plain English risk 3"
  ],
  "sector_heat": {{
    "Technology": {{"change_pct": 1.2, "signal": "bullish", "note": "AI demand driving broad sector strength"}},
    "Healthcare": {{"change_pct": -0.3, "signal": "neutral", "note": "Mixed earnings results"}},
    "Energy": {{"change_pct": 0.8, "signal": "bullish", "note": "Oil prices rising on supply constraints"}}
  }},
  "sector_rotation": "1-2 sentences on where institutional money is flowing this week",
  "short_squeeze_alerts": [
    {{
      "ticker": "XXXX",
      "short_float": "XX%",
      "social_velocity": "rising" | "stable" | "dropping",
      "setup": "1 sentence on squeeze setup and what needs to happen"
    }}
  ],
  "earnings_plays": [
    {{
      "ticker": "XXXX",
      "earnings_date": "Date + timing",
      "eps_estimate": "$X.XX",
      "play": "Beat/Miss expected and why — 1 sentence",
      "direction": "CALL" | "PUT" | "STRADDLE"
    }}
  ],
  "congressional_plays": [
    {{
      "ticker": "XXXX",
      "buy_count": <int>,
      "sell_count": <int or 0>,
      "members_preview": "Names of members who bought",
      "latest_date": "Most recent trade date e.g. '2 weeks ago'",
      "why_notable": "Why this congressional buy pattern matters for the stock"
    }}
  ],
  "technical_breakouts": [
    {{
      "ticker": "XXXX",
      "setup_type": "OVERSOLD_BOUNCE" | "MACD_BULLISH_CROSS" | "BB_BREAKOUT" | "NEAR_52W_LOW",
      "description": "Specific TA signal description with numbers",
      "timeframe": "days to play out"
    }}
  ],
  "rok_message": "Write like texting a friend. Max 4 sentences. Name the single best buy right now and why. Reference held positions by ticker. Name the biggest risk to watch. e.g. 'My top pick right now is NVDA — AI chip demand is exploding. Your AAPL position is showing some weakness, I'd consider trimming if it breaks below $185. Biggest risk this week is the Fed meeting on Wednesday.'",
  "position_analysis": [
    {{
      "ticker": "XXXX",
      "action": "HOLD" | "TRIM" | "SELL" | "STRONG_HOLD" | "ADD",
      "confidence": <1-10>,
      "thesis": "1-2 sentences: why hold or exit — plain English, reference signals",
      "risk": "Biggest threat to this position in plain English",
      "target": <float — price target or exit level if applicable, or null>
    }}
  ],
  "morning_game_plan": {{
    "step1": "First action to take when market opens — specific ticker and what to do",
    "step2": "Second priority action — monitor or act on this setup",
    "step3": "Risk management reminder — what level or event would change the plan",
    "best_entry_window": "Best time window today based on market conditions and day type",
    "max_new_positions": <integer — how many new trades to take today given current conditions>
  }}
}}

Rules:
- 4-7 buy signals, 2-4 sell signals, 3-5 watch list items
- Every buy signal needs signal_strength >= 6 and minimum 3 data_signals
- congressional_plays: only include if congressional data shows notable clustering
- technical_breakouts: only pure TA plays not already in buy/sell signals, max 4
- macro_risks: always include exactly 3"""


def _fmt_technical(ta_data: dict) -> str:
    if not ta_data:
        return "  No technical data available"
    lines = []
    for ticker, d in ta_data.items():
        rsi_str = f"RSI:{d.get('rsi', 'n/a')}({d.get('rsi_signal', '')})"
        macd_str = f"MACD:{d.get('macd_signal_label', 'n/a')}"
        bb_str = f"BB:{d.get('bb_signal', 'n/a')}"
        vol_str = f"Vol:{d.get('volume_ratio', 'n/a')}x"
        w52_str = f"52W:{d.get('pct_from_52w_high', 'n/a')}%fromHigh"
        lines.append(f"  {ticker}: {rsi_str} | {macd_str} | {bb_str} | {vol_str} | {w52_str}")
    return "\n".join(lines) or "  No data"


def _build_exit_intel_str(lmc: dict) -> str:
    ei = lmc.get("exit_intelligence") or {}
    positions = ei.get("positions") or []
    if not positions:
        return "  No exit intelligence data yet (runs after first closed trades)"
    lines = []
    for p in positions[:8]:
        grade = p.get("grade", "NEUTRAL")
        score = p.get("score", 50)
        pnl   = p.get("pnl_pct", 0)
        stop_tight = " [stop tightened]" if p.get("stop_tightened") else ""
        lines.append(f"  {p.get('ticker','')}: score={score} grade={grade} P&L={pnl:+.1f}%{stop_tight}")
    if ei.get("exit_count", 0) > 0:
        lines.append(f"  *** {ei['exit_count']} EXIT SIGNAL(s) detected — brain sees weakness ***")
    return "\n".join(lines) or "  All positions showing continuation"


def _build_sector_wr_str(lmc: dict) -> str:
    sp = lmc.get("sector_performance") or {}
    if not sp:
        return "  No sector data yet"
    sorted_secs = sorted(sp.items(), key=lambda x: -(x[1].get("win_rate", 0)))
    lines = []
    for sec, v in sorted_secs[:6]:
        wr  = v.get("win_rate", 0)
        tot = v.get("total", 0)
        label = "🔥 HOT" if wr >= 65 else "❄ COLD" if wr < 40 else ""
        lines.append(f"  {sec}: {wr:.0f}% WR ({tot} trades) {label}")
    return "\n".join(lines) or "  No sector data"


def build_prompt(
    ticker_mentions=None, reddit_posts=None, news_articles=None,
    stock_data=None, sec_filings=None, fear_greed=None,
    earnings_calendar=None, unusual_options=None, short_squeeze_candidates=None,
    market_indices=None, aggregate_sentiment=None, live_market_context=None,
    stocktwits_trending=None, technical_data=None,
    congressional_buys=None, market_breadth=None, put_call_ratio=None,
    insider_buys=None, current_positions=None, scan_top=None, **kwargs,
) -> str:
    today = datetime.utcnow().strftime("%B %d, %Y")
    fg = fear_greed or {}
    agg = aggregate_sentiment or {}
    pcr = put_call_ratio or {}
    breadth = market_breadth or {}

    # Format each section
    indices_str = " | ".join(
        f"{k}: ${v['price']} ({v.get('change_pct', 0):+.1f}%)"
        for k, v in (market_indices or {}).items()
    ) or "No index data"

    pcr_str = (
        f"Total P/C: {pcr.get('total', 'n/a')} | Equity P/C: {pcr.get('equity', 'n/a')} | "
        f"Signal: {pcr.get('signal', 'UNKNOWN')}"
    )

    breadth_str = (
        f"Advancing sectors: {breadth.get('advancing_sectors', 'n/a')}/{(breadth.get('advancing_sectors', 0) + breadth.get('declining_sectors', 0))} | "
        f"Breadth: {breadth.get('breadth_pct', 'n/a')}% | "
        f"A/D ratio: {breadth.get('advance_decline_ratio', 'n/a')}"
    )

    agg_str = (
        f"Bullish: {agg.get('bullish_pct', 0)}% | Bearish: {agg.get('bearish_pct', 0)}% | "
        f"Neutral: {agg.get('neutral_pct', 0)}% | Mean: {agg.get('mean', 0):+.3f} | "
        f"Posts: {agg.get('total_posts', 0)}"
    )

    ticker_str = "\n".join(
        f"  ${t}: {c} mentions" for t, c in (ticker_mentions or [])[:30]
    ) or "  No data"

    st_str = "\n".join(
        f"  ${s['ticker']}: {s.get('name', '')} | Watchlists: {s.get('watchlist_count', 0)}"
        for s in (stocktwits_trending or [])[:15]
    ) or "  No StockTwits data"

    reddit_str = "\n".join(
        f"  [{p.get('upvotes', 0)} upvotes | {p.get('source', '')} | sent:{p.get('sentiment_score', 0):+.2f}] "
        f"{p.get('title', '')[:130]}"
        for p in sorted(reddit_posts or [], key=lambda x: x.get("upvotes", 0), reverse=True)[:30]
    ) or "  No data"

    news_str = "\n".join(
        f"  [{a.get('source', '')}] {a.get('title', '')[:130]}"
        for a in (news_articles or [])[:35]
    ) or "  No data"

    stock_str = "\n".join(
        f"  {s['ticker']}: ${s['price']} ({s.get('change_pct', 0):+.1f}%) | "
        f"Vol:{s.get('vol_ratio', 1.0):.1f}x avg | PE:{s.get('pe_ratio', 'n/a')} | "
        f"RSI:{s.get('rsi', 'n/a')} | "
        f"Target:${s.get('analyst_target', 'n/a')}(+{s.get('upside_to_target', 'n/a')}%) | "
        f"Short:{s.get('short_interest', 'n/a')} | Rec:{s.get('recommendation', 'n/a')} | "
        f"Social:{s.get('sentiment', {}).get('mean', 0):+.2f}({s.get('sentiment', {}).get('mention_count', 0)}x)"
        for s in (stock_data or []) if s
    ) or "  No data"

    tech_str = _fmt_technical(technical_data or {})

    earnings_str = "\n".join(
        f"  {e.get('ticker')} | {e.get('company', '')} | {e.get('date', '')} {e.get('timing', '')} | EPS est: {e.get('eps_estimate', 'n/a')}"
        for e in (earnings_calendar or [])[:20]
    ) or "  No earnings data"

    opts_str = "\n".join(
        f"  {o.get('ticker')}: {o.get('description', '')[:100]}"
        for o in (unusual_options or [])[:15]
    ) or "  No unusual options"

    squeeze_str = "\n".join(
        f"  {s.get('ticker')} — {s.get('company', '')} | Short: {s.get('short_float', 'n/a')}"
        for s in (short_squeeze_candidates or [])[:10]
    ) or "  No short squeeze data"

    congress_str = "\n".join(
        f"  ${c['ticker']}: {c['buy_count']} buys by {c['members_preview']} (latest: {c.get('latest_date', '')})"
        for c in (congressional_buys or [])[:15]
    ) or "  No recent congressional trades"

    sec_str = "\n".join(
        f"  [{f.get('form_type')}] {f.get('company_name', '')} — filed {f.get('filing_date', '')}"
        for f in (sec_filings or [])[:20]
    ) or "  No SEC filings"

    insider_str = "\n".join(
        f"  ${b.get('ticker')} ({b.get('company', '')}): {b.get('insider_name', '')} [{b.get('title', '')}] "
        f"bought {b.get('shares', 0):,} shares"
        + (f" = ${b.get('value_usd', 0):,}" if b.get('value_usd') else "")
        + f" on {b.get('date', '')}"
        for b in (insider_buys or [])[:15]
    ) or "  No recent insider purchases"

    # Current positions context — rich signal data for per-position AI guidance
    positions_str = ""
    pos_detail_lines = []
    if current_positions:
        for p in current_positions[:12]:
            tk   = p.get("ticker", "")
            pnl  = p.get("pnl_pct", 0) or 0
            cost = p.get("cost", 0) or 0
            cur  = p.get("price", 0) or 0
            val  = p.get("market_val", 0) or 0
            ls   = p.get("live_signals", {}) or {}
            rsi  = ls.get("rsi", "n/a")
            psar_bull = ls.get("psar_bull", True)
            st_bull   = ls.get("supertrend_bull", True)
            mfi       = ls.get("mfi", "n/a")
            roc5      = ls.get("roc5", 0) or 0
            adx       = ls.get("adx", 0) or 0
            vwap_pos  = ls.get("vwap_pos", 0) or 0
            mtf_tri   = ls.get("mtf_triple", False)
            accum     = ls.get("accum_score", 0) or 0
            news_acc  = ls.get("news_accelerating", False)
            gex_sqz   = ls.get("squeeze_potential", False)
            short_fl  = ls.get("short_float", 0) or 0
            # Score degradation
            sh = p.get("score_history", [])
            scores = [h.get("s") for h in sh if isinstance(h.get("s"), (int, float))]
            if len(scores) >= 3:
                score_trend = f"{scores[0]}→{scores[-1]} ({'↓DEG' if scores[0] - scores[-1] >= 15 else '→stable' if abs(scores[0]-scores[-1]) < 5 else '↑rising'})"
            else:
                score_trend = f"score={scores[-1] if scores else 'n/a'}"
            edaystr = f" ⚠EARNINGS IN {p.get('earnings_days')}d" if p.get("earnings_days") is not None and p.get("earnings_days") <= 10 else ""
            # Signal flags for AI
            flags = []
            if not psar_bull: flags.append("PSAR_FLIPPED_BEARISH")
            if not st_bull:   flags.append("SUPERTREND_BEARISH")
            if isinstance(rsi, (int, float)) and rsi > 72: flags.append(f"RSI_OVERBOUGHT({rsi:.0f})")
            if isinstance(rsi, (int, float)) and rsi < 35: flags.append(f"RSI_OVERSOLD({rsi:.0f})")
            if isinstance(mfi, (int, float)) and mfi > 80:  flags.append(f"MFI_OVERBOUGHT({mfi:.0f})")
            if mtf_tri:  flags.append("TRIPLE_TF_ALIGNED")
            if accum >= 8: flags.append(f"STRONG_ACCUMULATION({accum}/10)")
            if news_acc: flags.append("NEWS_ACCELERATING")
            if gex_sqz:  flags.append("GAMMA_SQUEEZE_SETUP")
            if short_fl > 0.15: flags.append(f"HIGH_SHORT_{round(short_fl*100)}pct")
            if roc5 > 5:  flags.append(f"MOMENTUM_STRONG(+{roc5:.1f}%)")
            if roc5 < -4: flags.append(f"MOMENTUM_WEAK({roc5:.1f}%)")
            cg = ls.get("consec_green", 0) or 0
            cr = ls.get("consec_red", 0) or 0
            if cg >= 3:   flags.append(f"GREEN_STREAK_{cg}d")
            if cr >= 3:   flags.append(f"RED_STREAK_{cr}d")
            if ls.get("hv_contracting"): flags.append("HV_COIL")
            if ls.get("hv_expanding"):   flags.append("HV_EXPANDING")
            fib_tgt = ls.get("fib_level_382") or 0
            if fib_tgt and cur and abs(cur - fib_tgt) / max(cur, 0.01) < 0.015:
                flags.append(f"AT_FIB38.2({fib_tgt:.2f})")
            w52 = ls.get("w52_range_pos") or 0
            if w52 >= 88:  flags.append(f"NEAR_52W_HIGH({w52:.0f}%)")
            if w52 <= 12:  flags.append(f"NEAR_52W_LOW({w52:.0f}%)")
            atm_iv = ls.get("atm_iv") or 0
            hv20   = ls.get("hv20") or 0
            if atm_iv > 0 and hv20 > 0:
                if atm_iv > hv20 * 1.35: flags.append(f"IV_ELEVATED({atm_iv:.0f}%vsHV{hv20:.0f}%)")
                elif atm_iv < hv20 * 0.8: flags.append(f"IV_CHEAP({atm_iv:.0f}%vsHV{hv20:.0f}%)")
            rs_sec = ls.get("rs_sector") or 0
            if rs_sec >= 10:   flags.append(f"SECTOR_LEADER(+{rs_sec:.0f}%vsETF)")
            elif rs_sec <= -8: flags.append(f"SECTOR_LAGGARD({rs_sec:.0f}%vsETF)")
            flag_str = " | ".join(flags) if flags else "no_flags"
            pos_detail_lines.append(
                f"  ${tk}: P&L {pnl:+.1f}% | cost ${cost:.2f} → ${cur:.2f} | val ${val:,.0f}"
                f" | signals: {score_trend} | {flag_str}{edaystr}"
            )
        positions_str = "\n".join(pos_detail_lines)
    else:
        positions_str = "  No open positions"

    # Rejected candidates (why bot passed on certain stocks this scan)
    rejected_str = ""
    _rejected = lmc.get("last_scan_rejected") or []
    if _rejected:
        rej_lines = []
        for r in _rejected[:6]:
            rej_lines.append(f"  ${r.get('ticker','')} score={r.get('score',0)} — SKIPPED: {r.get('reason','')}")
        rejected_str = "\n".join(rej_lines)

    # Recent Alpaca auto-executed trades
    recent_trades_str = ""
    _recent_trades = lmc.get("recent_alpaca_trades") or []
    if _recent_trades:
        tr_lines = []
        for t in _recent_trades[-8:]:
            act = t.get("action","")
            tk  = t.get("ticker","")
            px  = t.get("price") or 0
            pnl = t.get("pnl_pct")
            sc  = t.get("score") or "?"
            ts  = (t.get("time") or "")[:16]
            pnl_str = f" P&L {pnl:+.1f}%" if pnl is not None else ""
            tr_lines.append(f"  {act} ${tk} @${px:.2f} score={sc}{pnl_str} [{ts}]")
        recent_trades_str = "\n".join(tr_lines)

    # Last scan top candidates context
    scan_str = ""
    if scan_top:
        scan_lines = []
        for s in scan_top[:8]:
            tk = s.get("ticker", "")
            sc = s.get("score", 0)
            gr = s.get("grade", "")
            cat = s.get("catalyst", "")[:60]
            mtf = "3TF✓" if s.get("mtf_triple") else ""
            acc = f"ACC{s.get('accum_score',0)}" if (s.get("accum_score") or 0) >= 6 else ""
            sqz = "γSQZ" if s.get("squeeze_potential") else ""
            w52_s = s.get("w52_range_pos") or 0
            w52_tag = f"52W:{w52_s:.0f}%" if w52_s else ""
            hv_tag = "HV↓COIL" if s.get("hv_contracting") else ("HV↑EXP" if s.get("hv_expanding") else "")
            # New advanced pattern tags
            pp_tag  = "PP" if s.get("pocket_pivot") else ""
            htf_tag = f"HTF{s.get('htf_consec',0)}d" if s.get("htf") else ""
            tt_tag  = f"TT{s.get('trend_template',0)}" if (s.get("trend_template") or 0) >= 6 else ""
            e21_tag = "E21↩" if s.get("ema21_pullback") else ""
            fq_tag  = f"F+{s.get('fund_quality',0)}" if (s.get("fund_quality") or 0) >= 2 else ""
            rs_tag  = f"RS{s.get('rs_rating',0)}" if (s.get("rs_rating") or 0) >= 80 else ""
            avwap_tag = f"AVWAP+{(s.get('avwap_dist_pct') or 0):.0f}%" if s.get("above_avwap_52wl") else ""
            tags = " ".join(t for t in [mtf, acc, sqz, w52_tag, hv_tag, pp_tag, htf_tag, tt_tag, e21_tag, fq_tag, rs_tag, avwap_tag] if t)
            scan_lines.append(f"  ${tk} score={sc} grade={gr}{' ['+tags+']' if tags else ''} | {cat}")
        scan_str = "\n".join(scan_lines)
        # Append leaderboard summary
        _pp_list  = [s["ticker"] for s in scan_top if s.get("pocket_pivot")][:4]
        _htf_list = [s["ticker"] for s in scan_top if s.get("htf")][:4]
        _tt8_list = [s["ticker"] for s in scan_top if s.get("tt_full")][:4]
        if _pp_list:  scan_str += f"\n  Pocket Pivots: {', '.join(_pp_list)}"
        if _htf_list: scan_str += f"\n  High-Tight Flags: {', '.join(_htf_list)}"
        if _tt8_list: scan_str += f"\n  SEPA TT 8/8 (elite): {', '.join(_tt8_list)}"
    else:
        scan_str = "  No recent scan data"

    base = ANALYSIS_PROMPT.format(
        today=today,
        fg_score=fg.get("score", 50),
        fg_rating=fg.get("rating", "Neutral"),
        fg_direction=fg.get("direction", "flat"),
        fg_prev=fg.get("previous_score", 50),
        indices=indices_str,
        put_call=pcr_str,
        breadth=breadth_str,
        agg_sentiment=agg_str,
        ticker_mentions=ticker_str,
        stocktwits_trending=st_str,
        reddit_posts=reddit_str,
        news_headlines=news_str,
        stock_data=stock_str,
        technical_data=tech_str,
        earnings_calendar=earnings_str,
        unusual_options=opts_str,
        short_squeeze=squeeze_str,
        congressional_trades=congress_str,
        sec_filings=sec_str,
        insider_buys=insider_str,
    )
    # Build live market context string
    lmc = live_market_context or {}
    lmc_lines = []
    if lmc.get("day_type") and lmc["day_type"] != "unknown":
        dt = lmc["day_type"]
        eff = lmc.get("day_efficiency", 0.5)
        hint = lmc.get("strategy_hint", "neutral")
        lmc_lines.append(f"  Day type: {dt} (efficiency {round(eff*100)}%) → strategy: {hint}")
    if lmc.get("vts_regime"):
        lmc_lines.append(f"  VIX term structure: {lmc['vts_regime']} (contango=calm, backwardation=fear spike)")
    if lmc.get("timing_quality") is not None:
        tq_map = {3: "PRIME (power hour or morning sweet spot)", 2: "GOOD", 1: "CAUTION (lunch lull)", 0: "WAIT (first/last 10min)"}
        lmc_lines.append(f"  Market timing: {tq_map.get(lmc['timing_quality'], 'unknown')}")
    if lmc.get("win_rate") is not None:
        wr = round((lmc["win_rate"] or 0) * 100, 1)
        pf = lmc.get("profit_factor")
        lmc_lines.append(f"  Bot performance: {wr}% win rate{f' | profit factor {pf}' if pf else ''}")
    if lmc.get("drawdown_pct", 0) > 3:
        lmc_lines.append(f"  ⚠ Portfolio in drawdown: {lmc['drawdown_pct']:.1f}% — be selective with new entries")
    if lmc.get("portfolio_beta"):
        lmc_lines.append(f"  Portfolio beta: {lmc['portfolio_beta']:.2f}")
    if lmc.get("market_quality") is not None:
        mq = lmc["market_quality"]
        mq_label = "STRONG" if mq >= 65 else "OK" if mq >= 45 else "WEAK" if mq >= 30 else "POOR"
        lmc_lines.append(f"  Market quality: {mq}/100 ({mq_label})")
    if lmc.get("scan_breadth_pct") is not None:
        lmc_lines.append(f"  Scan breadth: {lmc['scan_breadth_pct']:.0f}% of scanned stocks advancing")
    pc = lmc.get("portfolio_concentration") or {}
    if pc.get("risk_level") == "HIGH":
        dom = pc.get("dominant_sector", "unknown")
        cnt = pc.get("max_sector_count", 0)
        lmc_lines.append(f"  ⚠ HIGH CONCENTRATION RISK: {cnt} positions in {dom} sector — diversification needed")
    elif pc.get("risk_level") == "MEDIUM":
        dom = pc.get("dominant_sector", "unknown")
        lmc_lines.append(f"  MEDIUM concentration: multiple {dom} positions — watch sector-wide risk")
    # New advanced pattern signals from live market context
    if lmc.get("tt8_stocks"):
        lmc_lines.append(f"  SEPA Trend Template 8/8 (elite quality): {', '.join(lmc['tt8_stocks'][:4])}")
    if lmc.get("pocket_pivots"):
        lmc_lines.append(f"  Pocket Pivot (O'Neil inst. buying): {', '.join(lmc['pocket_pivots'][:4])}")
    if lmc.get("htf_stocks"):
        htf_str = ", ".join(f"{e.get('ticker','')}({e.get('htf_consec',0)}d)" for e in lmc['htf_stocks'][:3])
        lmc_lines.append(f"  High-Tight Flags (Minervini): {htf_str}")
    if lmc.get("drawdown_halt"):
        lmc_lines.append("  ⛔ DRAWDOWN HALT: portfolio ≥5% below peak — NO NEW BUYS until recovery")
    # Sector ETF trends: highlight lagging sectors for exit context
    etf_trends = lmc.get("sector_etf_trends") or {}
    bearish_secs = [s for s, d in etf_trends.items() if d.get("chg5d", 0) < -3]
    if bearish_secs:
        lmc_lines.append(f"  Weakest sectors (5d decline): {', '.join(bearish_secs[:3])}")
    # Cross-asset: DXY and 10-year yield context
    dxy = lmc.get("dxy_level") or 0
    dxy_5d = lmc.get("dxy_5d") or 0
    tnx = lmc.get("tnx_level") or 0
    rate_env = lmc.get("rate_environment") or ""
    if dxy > 0:
        dxy_signal = "strengthening (bearish for multinationals & commodities)" if dxy_5d > 1.5 else \
                     "weakening (tailwind for global risk-on)" if dxy_5d < -1.5 else "stable"
        lmc_lines.append(f"  Dollar (DXY): {dxy:.1f} ({dxy_5d:+.2f}%/5d) — {dxy_signal}")
    if tnx > 0:
        tnx_signal = "high & rising (growth/tech headwind — compresses valuations)" if rate_env == "restrictive" else \
                     "elevated (watch rate-sensitive growth names)" if rate_env == "elevated" else \
                     "low (rate tailwind for growth stocks)" if rate_env == "accommodative" else "neutral"
        lmc_lines.append(f"  10-Year Yield (TNX): {tnx:.2f}% — {tnx_signal}")
    lmc_str = "\n".join(lmc_lines) if lmc_lines else "  No live context available"

    # Build weekend watchlist section
    wl = lmc.get("weekend_watchlist") or []
    wl_str = ""
    if wl:
        wl_lines = []
        for w in wl[:8]:
            tk = w.get("ticker", "")
            wc = w.get("week_chg", 0)
            sc = w.get("score", 0)
            cat = w.get("catalyst", "")[:60]
            wl_lines.append(f"  ${tk}: score={sc} week={wc:+.1f}% | {cat}")
        wl_str = "\n".join(wl_lines)
    # Build intraday performance context
    iw = lmc.get("intraday_wins", 0)
    il = lmc.get("intraday_losses", 0)
    lstreak = lmc.get("loss_streak", 0)
    drisk = lmc.get("daily_risk_mult", 1.0)
    intraday_lines = []
    if iw + il > 0:
        intraday_lines.append(f"  Today's trades: {iw}W / {il}L")
        if lstreak >= 2:
            intraday_lines.append(f"  ⚠ LOSS STREAK: {lstreak} consecutive losses — risk reduced to {round(drisk*100)}%")
        elif iw > 0 and il == 0:
            intraday_lines.append(f"  Clean day so far: {iw} wins, 0 losses")
    elif drisk < 1.0:
        intraday_lines.append(f"  Risk budget reduced to {round(drisk*100)}% (loss streak protection active)")
    intraday_str = "\n".join(intraday_lines) if intraday_lines else "  No trades taken today yet"

    # Build brain summary section
    conv = lmc.get("bot_conviction", 0)
    strat = lmc.get("strategy_mode", "")
    nA = lmc.get("neurons_active", 0)
    nT = lmc.get("neurons_total", 730)
    last_dec = lmc.get("last_decision", "")[:150]
    brain_str = f"  Conviction: {conv}/100 | Strategy: {strat} | Brain: {nA}/{nT} neurons active"
    if last_dec:
        brain_str += f"\n  Last decision: {last_dec}"

    # Build top neurons section (high-performing learned signal dimensions)
    top_neurons_lines = []
    for n in (lmc.get("top_neurons") or [])[:8]:
        label = n.get("neuron", n.get("key", "")).replace("_perf", "").replace("_", " ")
        wr = n.get("win_rate", 0)
        best = n.get("best_state", "")
        tot = n.get("total", 0)
        if tot >= 3:
            top_neurons_lines.append(f"  {label}: {wr:.0f}% WR when {best} ({tot} trades)")
    top_neurons_str = "\n".join(top_neurons_lines) if top_neurons_lines else "  Not enough trade history yet"

    # Build top signal synergies section
    syn_lines = []
    for s in (lmc.get("top_synapses") or [])[:6]:
        syn_lines.append(f"  {s.get('pair','').replace('+', ' + ')}: {s.get('wr',0):.0f}% WR ({s.get('n',0)} trades)")
    syn_str = "\n".join(syn_lines) if syn_lines else "  Not enough trade history yet"

    tri_lines = []
    for t in (lmc.get("top_triplets") or [])[:5]:
        tri_lines.append(f"  {t.get('combo','')[:50]}: {t.get('wr',0):.0f}% WR ({t.get('n',0)} trades)")
    tri_str = "\n".join(tri_lines) if tri_lines else "  Not enough trade history yet"

    # Build next entry conditions summary
    nec = lmc.get("next_entry_conditions") or {}
    nec_met = nec.get("met", 0)
    nec_total = nec.get("total", 7)
    nec_ready = nec.get("ready", False)
    nec_top = nec.get("top_candidate")
    nec_score = nec.get("top_score", 0)
    nec_str = f"  {nec_met}/{nec_total} conditions met | {'✅ READY TO TRADE' if nec_ready else '⏸ WAITING'}"
    if nec_top:
        nec_str += f" | Top pick: {nec_top} (score {nec_score})"

    # Build brain's top picks section
    top_picks_lines = []
    for pick in (nec.get("top_picks") or [])[:3]:
        tk  = pick.get("ticker", "")
        sc  = pick.get("score", 0)
        px  = pick.get("price", 0)
        chg = pick.get("chg_pct", 0)
        rvol = pick.get("vol_ratio", 1)
        cat  = "✓catalyst" if pick.get("catalyst") else ""
        trigs = ", ".join(pick.get("triggers", []))
        stop  = pick.get("stop_price", 0)
        rsi   = pick.get("rsi", 50)
        rs5   = pick.get("rs5", 0)
        top_picks_lines.append(
            f"  ${tk} score={sc} | ${px:.2f} ({chg:+.1f}%) | RVOL {rvol:.1f}x | "
            f"RSI {rsi:.0f} | RS5 {rs5:+.1f}% | stop ${stop:.2f} | "
            f"triggers: [{trigs}] {cat}"
        )
    top_picks_str = "\n".join(top_picks_lines) if top_picks_lines else "  No qualified picks yet"

    # Build portfolio correlation risk summary
    corr_lines = []
    if current_positions:
        # Group by sector and detect crypto correlation
        sector_groups = {}
        crypto_tickers = []
        CRYPTO_PROXY = {"COIN", "MSTR", "RIOT", "MARA", "CLSK", "HUT", "BTBT", "BITO", "GBTC"}
        for p in current_positions:
            sec = p.get("sector", "Unknown")
            tk = p.get("ticker", "")
            sector_groups.setdefault(sec, []).append(tk)
            if tk in CRYPTO_PROXY or "Crypto" in sec or "Bitcoin" in sec:
                crypto_tickers.append(tk)
        # High-concentration sectors
        for sec, tks in sorted(sector_groups.items(), key=lambda x: -len(x[1])):
            if len(tks) >= 2:
                corr_lines.append(f"  {sec}: {', '.join(tks)} ({len(tks)} positions)")
        if crypto_tickers:
            corr_lines.append(f"  ⚠ Crypto-correlated cluster: {', '.join(crypto_tickers)} — move together with BTC")
        total_pos = len(current_positions)
        max_cluster = max((len(v) for v in sector_groups.values()), default=0)
        if max_cluster / max(total_pos, 1) > 0.5:
            corr_lines.append(f"  ⚠ HIGH CONCENTRATION: {max_cluster}/{total_pos} positions in same sector — sector selloff would hit all")
    corr_str = "\n".join(corr_lines) if corr_lines else "  No concentration risk detected"

    # Build brain analytics section (optimal trading conditions from neuron learning)
    ba = lmc.get("brain_analytics") or {}
    ba_lines = []
    if ba.get("best_day_of_week"):
        bst = ba["best_day_of_week"]
        ba_lines.append(f"  Best day to enter: {bst.get('best_state','?')} → {bst.get('win_rate',0):.0f}% WR ({bst.get('samples',0)} trades)")
    if ba.get("best_session"):
        bst = ba["best_session"]
        ba_lines.append(f"  Best session: {bst.get('best_state','?')} → {bst.get('win_rate',0):.0f}% WR")
    if ba.get("best_entry_hour"):
        bh = ba["best_entry_hour"]
        ampm = "AM" if bh.get("hour_et", 0) < 12 else "PM"
        hr12 = bh["hour_et"] % 12 or 12
        ba_lines.append(f"  Best entry hour (ET): {hr12} {ampm} → {bh.get('win_rate',0):.0f}% WR ({bh.get('samples',0)} trades)")
    if ba.get("best_vix_regime"):
        bst = ba["best_vix_regime"]
        ba_lines.append(f"  Best VIX regime: {bst.get('best_state','?')} → {bst.get('win_rate',0):.0f}% WR")
    if ba.get("top_edge_neurons"):
        for n in ba["top_edge_neurons"][:2]:
            ba_lines.append(f"  Top edge: {n.get('neuron','').replace('_perf','').replace('_',' ')} | {n.get('best_state','')} → {n.get('win_rate',0):.0f}% WR ({n.get('samples',0)} trades)")
    if ba.get("best_hold_duration"):
        bh = ba["best_hold_duration"]
        ba_lines.append(f"  Best hold duration: {bh.get('state','?')} → {bh.get('win_rate',0):.0f}% WR, avg P&L {bh.get('avg_pnl',0):+.1f}% ({bh.get('samples',0)} trades)")
    if ba.get("best_catalyst_type"):
        bc = ba["best_catalyst_type"]
        ba_lines.append(f"  Best catalyst: {bc.get('state','?')} → {bc.get('win_rate',0):.0f}% WR ({bc.get('samples',0)} trades)")
    if ba.get("grade_performance"):
        gp = ba["grade_performance"]
        gp_parts = [f"{g}: {v.get('win_rate',0):.0f}%WR({v.get('samples',0)}t)" for g, v in sorted(gp.items(), key=lambda x: -x[1].get("win_rate",0))[:4]]
        if gp_parts:
            ba_lines.append(f"  Grade WRs: {' | '.join(gp_parts)}")
    if ba.get("weekly_wr_trend"):
        wt = ba["weekly_wr_trend"]
        arrow = "↑ IMPROVING" if wt.get("trend") == "improving" else ("↓ DECLINING" if wt.get("trend") == "declining" else "→ STABLE")
        ba_lines.append(f"  Win rate trend (4wk): {wt.get('first_wr',0):.0f}% → {wt.get('latest_wr',0):.0f}% [{arrow}]")
    ba_str = "\n".join(ba_lines) if ba_lines else "  Brain still accumulating trade data to identify optimal conditions"

    # Append live trading context so AI can give position-specific guidance
    return base + f"""

━━━ LIVE TRADING CONTEXT (PRIORITY — ROK BOT IS ACTIVELY HOLDING THESE) ━━━

LIVE MARKET CONDITIONS:
{lmc_str}

ROK BOT BRAIN STATUS:
{brain_str}

BRAIN'S LEARNED OPTIMAL TRADING CONDITIONS:
{ba_str}

ENTRY GATE STATUS:
{nec_str}

BRAIN'S TOP 3 PICKS RIGHT NOW (neural-scored, with entry triggers and stops):
{top_picks_str}

INTRADAY PERFORMANCE:
{intraday_str}

OPEN POSITIONS WITH LIVE TECHNICAL SIGNALS:
{positions_str}

Signal key: PSAR_FLIPPED_BEARISH=exit warning, SUPERTREND_BEARISH=trend turned down,
RSI_OVERBOUGHT=likely to pull back, TRIPLE_TF_ALIGNED=weekly+daily+hourly all bullish,
STRONG_ACCUMULATION=institutional buying, GAMMA_SQUEEZE_SETUP=short covering potential,
HIGH_SHORT=short squeeze fuel, NEWS_ACCELERATING=catalyst building, ↓DEG=score dropping

LAST SCAN TOP CANDIDATES (bot's freshest buy ideas):
{scan_str}
{f'''
RECENT ALPACA AUTO-EXECUTED TRADES (last 8 bot orders):
{recent_trades_str}
''' if recent_trades_str else ''}
{f'''
STOCKS BOT CONSIDERED BUT REJECTED THIS SCAN:
{rejected_str}
''' if rejected_str else ''}
{f'''
WEEKEND WATCHLIST — STOCKS BOT IS WATCHING FOR MONDAY:
{wl_str}
''' if wl_str else ''}
NEURAL EXIT INTELLIGENCE (brain's pattern-learned continuation scores per position):
{_build_exit_intel_str(lmc)}

SECTOR WIN RATES (brain's learned performance by sector):
{_build_sector_wr_str(lmc)}

TOP PERFORMING BRAIN NEURONS (highest win-rate signal dimensions — what the brain has learned works best):
{top_neurons_str}

TOP LEARNED SIGNAL PAIRS (synapse memory — when these 2 signals fire together, high win rate):
{syn_str}

TOP LEARNED 3-SIGNAL COMBOS (strongest neural triplets by historical win rate):
{tri_str}

PORTFOLIO CORRELATION RISK:
{corr_str}

CRITICAL INSTRUCTIONS:
1. For EACH open position, generate a position_analysis entry with: action (HOLD/TRIM/SELL/STRONG_HOLD/ADD),
   confidence (1-10), thesis (why hold or exit), risk (biggest threat), and target (price target or exit level)
2. Flag any position with PSAR_FLIPPED_BEARISH + SUPERTREND_BEARISH + negative P&L as urgent sell
3. Flag any position with TRIPLE_TF_ALIGNED + STRONG_ACCUMULATION as strong hold — consider adding
4. If Neural Exit Intelligence shows EXIT_SIGNAL for a position, strongly consider recommending exit
5. If a position's sector has <40% win rate in Sector Win Rates, add extra caution
6. Adjust advice for day type: on TREND day favor momentum plays; on RANGE day favor mean-reversion exits
7. Mention specific held tickers by name in rok_message — users see this as their pocket guide
8. Include top scan candidates and weekend watchlist as new buy ideas if they have strong signals
9. The user checks this dashboard as their pocket trading guide — make rok_message actionable and specific
10. If top learned signal pairs show high-WR patterns matching current positions, reference this in thesis
11. CORRELATION WARNING: If portfolio has 3+ positions in crypto/same sector, warn about concentrated risk in rok_message
12. When market is closed (weekend/after-hours), focus on Monday preparation — what to watch, what to set stops on
13. ADD signal: if a position is up 5%+ with strong technical signals and still has room to target, suggest adding shares
14. INTRADAY RISK: if loss_streak >= 2 in Intraday Performance, emphasize capital preservation in rok_message — suggest being selective and sizing down
15. If the brain's top neurons show a specific state with high win rate (e.g., "morning_session: 78% WR"), reference this timing context for entries
16. morning_game_plan: generate a concrete 3-step action plan. step1/step2 must name specific tickers. best_entry_window = "9:30-10:30 AM ET" for trend days, "10:30-11:30 AM ET" for range/choppy. max_new_positions: 1-2 if loss_streak or bear regime; 3-4 if strong_bull and clean day
17. Brain's Top 3 Picks: if any of these match buy signals in other data sources, elevate their signal_strength by +1 in buy_signals and mention the brain's score in reasons
18. Rejected Candidates: review STOCKS BOT CONSIDERED BUT REJECTED — if any were rejected only for minor reasons (sector full, thin vol) but have strong signals, mention them as alternative watches in rok_message
19. Recent Bot Trades: if RECENT ALPACA AUTO-EXECUTED TRADES shows recent buys, confirm they align with current signals; if a recently bought stock is now showing exit signals, flag it urgently in rok_message
"""


def run_analysis(
    api_key: str,
    model: str,
    ticker_mentions=None, reddit_posts=None, news_articles=None,
    stock_data=None, sec_filings=None, fear_greed=None,
    earnings_calendar=None, unusual_options=None,
    short_squeeze_candidates=None, market_indices=None,
    aggregate_sentiment=None,
    stocktwits_trending=None, technical_data=None,
    congressional_buys=None, market_breadth=None, put_call_ratio=None,
    insider_buys=None, current_positions=None, scan_top=None,
    live_market_context=None,
) -> dict:
    if not api_key:
        logger.warning("No API key — demo mode")
        return _demo_analysis()

    client = anthropic.Anthropic(api_key=api_key)
    prompt = build_prompt(
        ticker_mentions=ticker_mentions, reddit_posts=reddit_posts,
        news_articles=news_articles, stock_data=stock_data, sec_filings=sec_filings,
        fear_greed=fear_greed, earnings_calendar=earnings_calendar,
        unusual_options=unusual_options, short_squeeze_candidates=short_squeeze_candidates,
        market_indices=market_indices, aggregate_sentiment=aggregate_sentiment,
        stocktwits_trending=stocktwits_trending, technical_data=technical_data,
        congressional_buys=congressional_buys, market_breadth=market_breadth,
        put_call_ratio=put_call_ratio, insider_buys=insider_buys,
        current_positions=current_positions, scan_top=scan_top,
        live_market_context=live_market_context,
    )

    try:
        msg = client.messages.create(
            model=model,
            max_tokens=8096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)
        # Ensure new fields have defaults if Claude omitted them
        result.setdefault("market_regime", "UNCERTAIN")
        result.setdefault("macro_risks", [])
        result.setdefault("congressional_plays", [])
        result.setdefault("technical_breakouts", [])
        result.setdefault("sector_rotation", "")
        result.setdefault("position_analysis", [])
        result.setdefault("morning_game_plan", {})
        for pa in result.get("position_analysis", []):
            pa.setdefault("confidence", 5)
            pa.setdefault("target", None)
        for buy in result.get("buy_signals", []):
            buy.setdefault("stop_loss", None)
            buy.setdefault("options_play", None)
            buy.setdefault("technical_setup", None)
        for sq in result.get("short_squeeze_alerts", []):
            sq.setdefault("social_velocity", "stable")
        for ep in result.get("earnings_plays", []):
            ep.setdefault("eps_estimate", "n/a")
        return result
    except json.JSONDecodeError as e:
        logger.error(f"Claude JSON parse error: {e}")
        return None
    except Exception as e:
        logger.error(f"Claude API error: {e}")
        return None


def _demo_analysis() -> dict:
    today = datetime.utcnow().strftime("%B %d, %Y")
    return {
        "analysis_date": today,
        "market_regime": "UNCERTAIN",
        "week_summary": (
            "DEMO MODE — Add ANTHROPIC_API_KEY to activate live AI analysis. "
            "All 8 data scrapers are running live: Reddit (15 subs), 20+ news feeds, "
            "StockTwits, Congressional trades, SEC EDGAR, Technical Analysis, "
            "Yahoo Finance, and CNN Fear/Greed. ROK just needs the AI brain."
        ),
        "market_sentiment": "MIXED",
        "sentiment_score": 5,
        "buy_signals": [
            {
                "ticker": "NVDA",
                "company": "NVIDIA Corporation",
                "signal_strength": 9,
                "current_price": None,
                "price_target": None,
                "stop_loss": None,
                "time_horizon": "2-4 weeks",
                "risk_level": "Medium",
                "catalyst": "AI infrastructure demand continues to outpace supply",
                "technical_setup": None,
                "options_play": None,
                "reasons": [
                    "Top mentioned ticker across Reddit and StockTwits",
                    "Unusual call options activity in near-term expiries",
                    "Data center revenue guidance raised multiple quarters running",
                ],
                "rok_take": "NVDA is the infrastructure backbone of the AI revolution. Every dollar spent on AI runs through their chips.",
                "data_signals": ["reddit", "options", "news"],
            }
        ],
        "sell_signals": [
            {
                "ticker": "SETUP",
                "company": "Add your Anthropic API Key",
                "signal_strength": 10,
                "current_price": None,
                "reasons": [
                    "Go to console.anthropic.com to get your key",
                    "Add ANTHROPIC_API_KEY to GitHub Secrets",
                    "Trigger the workflow manually from GitHub Actions tab",
                ],
                "rok_take": "Takes 2 minutes. Then ROK goes fully live with real AI picks.",
                "urgency": "IMMEDIATE",
            }
        ],
        "watch_list": [
            {
                "ticker": "TSLA",
                "company": "Tesla Inc.",
                "why_watching": "Massive social volume with divided institutional/retail sentiment",
                "trigger": "Break above key resistance with high volume confirmation",
                "risk": "CEO news risk is permanent wildcard",
                "potential": 35.0,
            }
        ],
        "notable_trends": [
            "AI infrastructure spending at record levels — NVDA, AMD, AVGO all benefiting",
            "Retail investors rotating from meme stocks into AI infrastructure names",
            "Options volume surging — smart money positioning ahead of key events",
        ],
        "macro_risks": [
            "Fed rate policy uncertainty — any hawkish surprise crushes growth multiples",
            "Geopolitical escalation disrupting semiconductor supply chains",
            "Consumer spending slowdown hitting retail and discretionary sectors",
        ],
        "sector_heat": {
            "hottest": "Technology/AI — GPU and cloud infrastructure printing record revenue",
            "coldest": "Regional Banking — commercial real estate exposure and rate pressures",
        },
        "sector_rotation": "Institutional money flowing from defensives into growth/AI as inflation cools.",
        "short_squeeze_alerts": [],
        "earnings_plays": [],
        "congressional_plays": [],
        "technical_breakouts": [],
        "rok_message": (
            "All systems running. I'm pulling Reddit, news, StockTwits, congressional trades, "
            "technical analysis, SEC filings, options flow — everything. "
            "Add the Anthropic key and I tell you exactly what to buy, sell, and avoid."
        ),
    }
