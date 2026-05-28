import json
import logging
from datetime import datetime

import anthropic

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are ROK — the most elite private stock intelligence system ever built.

You fuse every data stream that exists: Reddit + StockTwits social sentiment, financial news from 20+ sources,
SEC insider trades, options flow, congressional stock disclosures, technical analysis (RSI/MACD/Bollinger Bands),
earnings calendars, short interest, fear/greed index, market breadth, put/call ratio, and live price data.

Your edge: institutional-grade signal aggregation that retail investors can't access alone.
You identify setups BEFORE they move — not after.

Analysis hierarchy (strongest to weakest signal):
1. Congressional insider buys + SEC Form 4 insider buying SIMULTANEOUSLY
2. Unusual options flow (large block calls) + rising social volume
3. Earnings catalyst + analyst upgrade + technical breakout
4. Social momentum (Reddit WSB + StockTwits trending) + volume surge
5. Technical setup alone (RSI oversold + MACD cross + BB breakout = trifecta)
6. News catalyst + sector momentum

Cross-reference ALL sources. The best picks have 4+ confirming signals.

Personality: Hedge fund quant who also reads WSB. Blunt. Specific. No disclaimers.
Every claim references actual data from the input. No made-up numbers.

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

═══ ANALYSIS INSTRUCTIONS ═══
1. Cross-reference every source. Congressional buy + insider Form 4 buy + unusual options on same ticker = maximum conviction.
2. Flag stocks where price diverges from sentiment (hidden catalyst or distribution).
3. RSI < 32 + MACD bullish cross + above average volume = technical trifecta. Note it.
4. Congressional buys are legally delayed 45 days — they already profited. Find what they bought most.
5. Short squeeze: need >15% short float + rising StockTwits/Reddit volume + technical breakout.
6. Earnings plays: analyze EPS estimate vs sector trend. Beats cluster in sectors — if peers beat, this will too.
7. Fear/Greed extreme (<20 or >80): adjust all risk levels accordingly.
8. Put/Call ratio >1.2 = excessive fear, fading it is historically profitable.
9. Market breadth <40% = risk-off, prefer defensive picks. >65% = risk-on, lean growth/momentum.
10. Every pick needs minimum 3 data signals. List them all in data_signals.

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
      "catalyst": "The single primary catalyst",
      "technical_setup": "RSI 28 oversold + MACD bullish cross" or null,
      "options_play": "Specific options strategy if applicable — e.g. 'Buy $X calls expiring MM/DD'" or null,
      "reasons": ["data-backed reason 1", "data-backed reason 2", "data-backed reason 3"],
      "rok_take": "1-2 sentences. Gut call. Direct.",
      "data_signals": ["reddit", "options", "earnings", "insider", "news", "technical", "congressional", "short_squeeze"]
    }}
  ],
  "sell_signals": [
    {{
      "ticker": "XXXX",
      "company": "Full Company Name",
      "signal_strength": <1-10>,
      "current_price": <float or null>,
      "reasons": ["reason 1", "reason 2"],
      "rok_take": "Why to exit now. 1-2 sentences.",
      "urgency": "IMMEDIATE" | "THIS WEEK" | "REDUCE POSITION"
    }}
  ],
  "watch_list": [
    {{
      "ticker": "XXXX",
      "company": "Full Company Name",
      "why_watching": "Specific data-backed reason",
      "trigger": "Exact event or price level that converts this to a BUY",
      "risk": "Primary risk factor",
      "potential": <float — percentage upside if trigger hits>
    }}
  ],
  "notable_trends": [
    "Specific macro/sector trend with data — 1-2 sentences",
    "Second trend",
    "Third trend",
    "Fourth trend",
    "Fifth trend"
  ],
  "macro_risks": [
    "Top risk 1 — specific and data-backed",
    "Top risk 2",
    "Top risk 3"
  ],
  "sector_heat": {{
    "hottest": "Sector name — why, with specific tickers or numbers",
    "coldest": "Sector name — why, with specific tickers or numbers"
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
      "members_preview": "Names",
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
  "rok_message": "Personal advisor text. Casual, direct, max 4 sentences. Reference the best pick and biggest risk."
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
    market_indices=None, aggregate_sentiment=None,
    stocktwits_trending=None, technical_data=None,
    congressional_buys=None, market_breadth=None, put_call_ratio=None,
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
        f"Vol:{s.get('volume', 0):,} | PE:{s.get('pe_ratio', 'n/a')} | "
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

    return ANALYSIS_PROMPT.format(
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
    )


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
        put_call_ratio=put_call_ratio,
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
