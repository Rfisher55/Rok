import json
import logging
from datetime import datetime

import anthropic

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are ROK — a private, elite stock market intelligence advisor.

You have access to an unprecedented fusion of data: Reddit/social media sentiment, financial news,
SEC insider trade filings, options flow, earnings catalysts, short interest, fear/greed index,
and real-time price data.

Your edge: you see the full picture that retail investors don't. You know which tickers are building
momentum before they pop, which have exhausted their moves, and which catalysts are coming.

Personality: You're like the smartest guy at a hedge fund who also reads WSB. Confident, direct,
no disclaimers, no hedging your language. You call it like you see it. Short sentences. Maximum impact.

CRITICAL: Output ONLY valid JSON. No markdown. No preamble. No explanation. Just the JSON object."""

ANALYSIS_PROMPT = """Generate this week's ROK stock intelligence briefing.

TODAY: {today}

=== CNN FEAR & GREED INDEX ===
Score: {fear_greed_score}/100 | Rating: {fear_greed_rating} | Direction: {fear_greed_direction}

=== MARKET INDICES ===
{market_indices}

=== AGGREGATE SOCIAL SENTIMENT ===
{agg_sentiment}

=== TRENDING TICKERS (mention count across Reddit + News) ===
{ticker_mentions}

=== TOP REDDIT POSTS (sorted by upvotes) ===
{reddit_posts}

=== FINANCIAL NEWS HEADLINES ===
{news_headlines}

=== LIVE STOCK DATA (price, change, volume, PE, analyst ratings, social sentiment) ===
{stock_data}

=== UPCOMING EARNINGS (next 7 days) ===
{earnings_calendar}

=== UNUSUAL OPTIONS ACTIVITY ===
{unusual_options}

=== SHORT SQUEEZE CANDIDATES (high short interest) ===
{short_squeeze}

=== SEC INSIDER TRADES & 8-K FILINGS (last 7 days) ===
{sec_filings}

ANALYSIS INSTRUCTIONS:
- Cross-reference ALL data sources. A stock with Reddit hype + unusual options + upcoming earnings
  and insider buying = strongest possible signal.
- Flag stocks where sentiment diverges from price action (hidden gems or trap setups).
- Short squeeze setups need: high short interest + rising retail sentiment.
- Earnings plays need: strong historical beat rate OR analyst estimate seems too low.
- Weight signals: Options flow > Insider buying > Earnings catalyst > Reddit/social > News sentiment.
- Be specific: use company names, real price levels, real reasons from the data.
- If the fear/greed is extreme (>80 or <20), factor that into risk assessment.

Respond ONLY with this exact JSON (no other text):
{{
  "analysis_date": "{today}",
  "week_summary": "2-3 sentence executive overview. Be direct. Use real data points from above.",
  "market_sentiment": "BULLISH" | "BEARISH" | "NEUTRAL" | "MIXED",
  "sentiment_score": <integer 1-10, maps fear/greed to ROK scale>,
  "buy_signals": [
    {{
      "ticker": "XXXX",
      "company": "Full Company Name",
      "signal_strength": <1-10>,
      "current_price": <float or null>,
      "price_target": <float or null>,
      "time_horizon": "1-2 weeks" | "2-4 weeks" | "1-3 months",
      "risk_level": "Low" | "Medium" | "High" | "Speculative",
      "catalyst": "Primary catalyst driving this pick",
      "reasons": ["specific reason from data 1", "specific reason from data 2", "specific reason 3"],
      "rok_take": "Your gut call on this. 1-2 sentences max. Direct.",
      "data_signals": ["reddit", "options", "earnings", "insider", "news", "short_squeeze"]
    }}
  ],
  "sell_signals": [
    {{
      "ticker": "XXXX",
      "company": "Full Company Name",
      "signal_strength": <1-10>,
      "current_price": <float or null>,
      "reasons": ["specific reason 1", "specific reason 2"],
      "rok_take": "Why you should get out now.",
      "urgency": "IMMEDIATE" | "THIS WEEK" | "REDUCE POSITION"
    }}
  ],
  "watch_list": [
    {{
      "ticker": "XXXX",
      "company": "Full Company Name",
      "why_watching": "Specific reason based on data",
      "trigger": "Exact event or price that converts this to a BUY",
      "risk": "Primary risk factor",
      "potential": <float percentage upside if trigger hits>
    }}
  ],
  "notable_trends": [
    "Specific trend with data backing",
    "Second trend",
    "Third trend"
  ],
  "sector_heat": {{
    "hottest": "Sector name — specific reason from the data",
    "coldest": "Sector name — specific reason from the data"
  }},
  "short_squeeze_alerts": [
    {{
      "ticker": "XXXX",
      "short_float": "XX%",
      "setup": "1 sentence on the squeeze setup"
    }}
  ],
  "earnings_plays": [
    {{
      "ticker": "XXXX",
      "earnings_date": "Date",
      "play": "Beat expected / Miss expected / Volatile — trade description",
      "direction": "CALL" | "PUT" | "STRADDLE"
    }}
  ],
  "rok_message": "Personal advisor message. Like a text from your trader friend. Casual, direct, specific. Max 3 sentences."
}}

Include 3-6 buy signals, 2-4 sell signals, 3-5 watch list items.
Fill short_squeeze_alerts and earnings_plays only if the data strongly supports them (can be empty arrays).
Every pick must have at least 2 data signals behind it."""


def build_analysis_prompt(
    ticker_mentions, reddit_posts, news_articles, stock_data, sec_filings,
    fear_greed=None, earnings_calendar=None, unusual_options=None,
    short_squeeze_candidates=None, market_indices=None, aggregate_sentiment=None,
) -> str:
    today = datetime.utcnow().strftime("%B %d, %Y")
    fg = fear_greed or {}
    agg = aggregate_sentiment or {}

    ticker_str = "\n".join(
        f"  ${t}: {c} mentions" for t, c in (ticker_mentions or [])[:25]
    ) or "  No data"

    reddit_str = "\n".join(
        f"  [{p.get('upvotes', 0)} upvotes | sentiment:{p.get('sentiment_score', 0):+.2f}] "
        f"{p.get('source', '')}: {p.get('title', '')[:120]}"
        for p in sorted(reddit_posts or [], key=lambda x: x.get("upvotes", 0), reverse=True)[:25]
    ) or "  No data"

    news_str = "\n".join(
        f"  [{a.get('source', '')}] {a.get('title', '')[:120]}"
        for a in (news_articles or [])[:25]
    ) or "  No data"

    stock_str = "\n".join(
        f"  {s.get('ticker')}: ${s.get('price')} ({s.get('change_pct', 0):+.1f}%) | "
        f"Vol:{s.get('volume', 0):,} | PE:{s.get('pe_ratio', 'n/a')} | "
        f"Analyst:{s.get('recommendation', 'n/a')} | Target:${s.get('analyst_target', 'n/a')} | "
        f"Social:{s.get('sentiment', {}).get('mean', 0):+.2f}({s.get('sentiment', {}).get('mention_count', 0)} mentions)"
        for s in (stock_data or []) if s
    ) or "  No data"

    earnings_str = "\n".join(
        f"  {e.get('ticker')} — {e.get('company')} | {e.get('date')} {e.get('timing')} | EPS est: {e.get('eps_estimate', 'n/a')}"
        for e in (earnings_calendar or [])[:15]
    ) or "  No upcoming earnings data"

    opts_str = "\n".join(
        f"  {o.get('ticker')}: {o.get('description', '')[:100]}"
        for o in (unusual_options or [])[:15]
    ) or "  No unusual options data"

    squeeze_str = "\n".join(
        f"  {s.get('ticker')} — {s.get('company')} | Short float: {s.get('short_float', 'n/a')}"
        for s in (short_squeeze_candidates or [])[:10]
    ) or "  No short squeeze data"

    sec_str = "\n".join(
        f"  [{f.get('form_type')}] {f.get('company_name')} — filed {f.get('filing_date')}"
        for f in (sec_filings or [])[:15]
    ) or "  No recent SEC filings"

    indices = market_indices or {}
    indices_str = " | ".join(
        f"{k}: ${v.get('price')} ({v.get('change_pct', 0):+.1f}%)"
        for k, v in indices.items()
    ) or "  No index data"

    agg_str = (
        f"  Bullish: {agg.get('bullish_pct', 0)}% | Bearish: {agg.get('bearish_pct', 0)}% | "
        f"Neutral: {agg.get('neutral_pct', 0)}% | Mean: {agg.get('mean', 0):+.3f} | "
        f"Posts analyzed: {agg.get('total_posts', 0)}"
    )

    return ANALYSIS_PROMPT.format(
        today=today,
        fear_greed_score=fg.get("score", 50),
        fear_greed_rating=fg.get("rating", "Neutral"),
        fear_greed_direction=fg.get("direction", "neutral"),
        market_indices=indices_str,
        agg_sentiment=agg_str,
        ticker_mentions=ticker_str,
        reddit_posts=reddit_str,
        news_headlines=news_str,
        stock_data=stock_str,
        earnings_calendar=earnings_str,
        unusual_options=opts_str,
        short_squeeze=squeeze_str,
        sec_filings=sec_str,
    )


def run_analysis(
    api_key: str,
    model: str,
    ticker_mentions=None,
    reddit_posts=None,
    news_articles=None,
    stock_data=None,
    sec_filings=None,
    fear_greed=None,
    earnings_calendar=None,
    unusual_options=None,
    short_squeeze_candidates=None,
    market_indices=None,
    aggregate_sentiment=None,
) -> dict | None:
    if not api_key:
        logger.warning("No Anthropic API key — returning demo analysis")
        return _demo_analysis()

    client = anthropic.Anthropic(api_key=api_key)
    prompt = build_analysis_prompt(
        ticker_mentions, reddit_posts, news_articles, stock_data, sec_filings,
        fear_greed, earnings_calendar, unusual_options,
        short_squeeze_candidates, market_indices, aggregate_sentiment,
    )

    try:
        message = client.messages.create(
            model=model,
            max_tokens=8096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"Claude returned invalid JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Claude API error: {e}")
        return None


def _demo_analysis() -> dict:
    today = datetime.utcnow().strftime("%B %d, %Y")
    return {
        "analysis_date": today,
        "week_summary": (
            "DEMO MODE — Add your ANTHROPIC_API_KEY to .env to activate live AI analysis. "
            "All data scrapers are running: Reddit, news feeds, SEC EDGAR, Yahoo Finance, "
            "fear/greed index, earnings calendar, and options flow are all live. "
            "ROK just needs the AI key to generate real picks."
        ),
        "market_sentiment": "MIXED",
        "sentiment_score": 5,
        "buy_signals": [
            {
                "ticker": "NVDA",
                "company": "NVIDIA Corporation",
                "signal_strength": 9,
                "current_price": 875.00,
                "price_target": 1050.00,
                "time_horizon": "2-4 weeks",
                "risk_level": "Medium",
                "catalyst": "AI infrastructure demand continues to outpace supply",
                "reasons": [
                    "Dominating Reddit/WSB conversation for 3+ weeks straight",
                    "Unusual call options activity spiking in near-term expiries",
                    "Data center revenue guidance raised — analysts still underestimating",
                ],
                "rok_take": "NVDA is the pick axe in the AI gold rush. Everyone needs their chips. This is not hype — it's infrastructure.",
                "data_signals": ["reddit", "options", "earnings"],
            },
        ],
        "sell_signals": [
            {
                "ticker": "SETUP",
                "company": "Configure ROK — Add API Key",
                "signal_strength": 10,
                "current_price": None,
                "reasons": [
                    "Step 1: Create .env file in the ROK folder",
                    "Step 2: Add ANTHROPIC_API_KEY=your-key-here",
                    "Step 3: Restart with python app.py",
                ],
                "rok_take": "Go to console.anthropic.com — takes 2 minutes to get your key. Then ROK goes live.",
                "urgency": "IMMEDIATE",
            }
        ],
        "watch_list": [
            {
                "ticker": "TSLA",
                "company": "Tesla Inc.",
                "why_watching": "Massive social volume with deeply divided sentiment — institutional accumulation conflicting with retail fear.",
                "trigger": "Break above $250 with volume confirmation — signals momentum re-entry",
                "risk": "Elon news risk is a permanent wildcard here",
                "potential": 35.0,
            },
            {
                "ticker": "PLTR",
                "company": "Palantir Technologies",
                "why_watching": "Government AI contract pipeline + commercial growth story gaining traction on Reddit.",
                "trigger": "Hold above $20 through earnings — then it's a legitimate swing",
                "risk": "Valuation stretched at current multiple",
                "potential": 28.0,
            },
        ],
        "notable_trends": [
            "AI/semiconductor stocks dominating ALL social media — not just WSB, but mainstream finance Twitter",
            "Retail investors rotating from meme stocks into AI infrastructure plays",
            "Options volume surging across the board — volatility priced higher than it should be",
            "Short interest spiking in EV names — potential squeeze setups building",
        ],
        "sector_heat": {
            "hottest": "Technology / AI — GPU and cloud names printing money on every earnings call",
            "coldest": "Regional Banking — rate uncertainty and commercial real estate exposure keeping investors away",
        },
        "short_squeeze_alerts": [],
        "earnings_plays": [],
        "rok_message": (
            "Everything is running. I've got Reddit, news, SEC filings, options flow, fear/greed — "
            "all of it. The only thing missing is the AI brain. Drop your Anthropic key in .env and I come alive. "
            "Get it at console.anthropic.com."
        ),
    }
