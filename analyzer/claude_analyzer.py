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
  ]
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
            flag_str = " | ".join(flags) if flags else "no_flags"
            pos_detail_lines.append(
                f"  ${tk}: P&L {pnl:+.1f}% | cost ${cost:.2f} → ${cur:.2f} | val ${val:,.0f}"
                f" | signals: {score_trend} | {flag_str}{edaystr}"
            )
        positions_str = "\n".join(pos_detail_lines)
    else:
        positions_str = "  No open positions"

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
            tags = " ".join(t for t in [mtf, acc, sqz] if t)
            scan_lines.append(f"  ${tk} score={sc} grade={gr}{' ['+tags+']' if tags else ''} | {cat}")
        scan_str = "\n".join(scan_lines)
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
    lmc_str = "\n".join(lmc_lines) if lmc_lines else "  No live context available"

    # Append live trading context so AI can give position-specific guidance
    return base + f"""

━━━ LIVE TRADING CONTEXT (PRIORITY — ROK BOT IS ACTIVELY HOLDING THESE) ━━━

LIVE MARKET CONDITIONS:
{lmc_str}

OPEN POSITIONS WITH LIVE TECHNICAL SIGNALS:
{positions_str}

Signal key: PSAR_FLIPPED_BEARISH=exit warning, SUPERTREND_BEARISH=trend turned down,
RSI_OVERBOUGHT=likely to pull back, TRIPLE_TF_ALIGNED=weekly+daily+hourly all bullish,
STRONG_ACCUMULATION=institutional buying, GAMMA_SQUEEZE_SETUP=short covering potential,
HIGH_SHORT=short squeeze fuel, NEWS_ACCELERATING=catalyst building, ↓DEG=score dropping

LAST SCAN TOP CANDIDATES (bot's freshest buy ideas):
{scan_str}

CRITICAL INSTRUCTIONS:
1. For EACH open position, generate a position_analysis entry with: action (HOLD/TRIM/SELL/STRONG_HOLD),
   confidence (1-10), thesis (why hold or exit), risk (biggest threat), and target (price target or exit level)
2. Flag any position with PSAR_FLIPPED_BEARISH + SUPERTREND_BEARISH + negative P&L as urgent sell
3. Flag any position with TRIPLE_TF_ALIGNED + STRONG_ACCUMULATION as strong hold
4. Adjust advice for day type: on TREND day favor momentum plays; on RANGE day favor mean-reversion exits
5. Mention specific held tickers by name in rok_message — users see this as their pocket guide
6. Include top scan candidates as new buy ideas if they have strong signals
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
