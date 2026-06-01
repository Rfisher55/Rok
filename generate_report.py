"""
ROK — Market intelligence pipeline for GitHub Pages.
Runs via GitHub Actions every 15 minutes.
Writes docs/intel_report.json (read by the trading dashboard via JS fetch).
Does NOT overwrite docs/index.html — the trading dashboard owns that file.
"""
import json
import logging
import sys
from datetime import datetime, timezone, date
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


class _Encoder(json.JSONEncoder):
    """Handle datetime/date objects that scrapers sometimes return."""
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


def _safe(fn, *args, default=None, label=""):
    try:
        result = fn(*args)
        logger.info(f"{label or fn.__name__}: ok ({_size(result)})")
        return result
    except Exception as e:
        logger.warning(f"{label or fn.__name__} failed: {e}")
        return default() if callable(default) else default


def _size(v):
    if isinstance(v, (list, dict)):
        return len(v)
    return "ok"


def _sanitize(obj):
    """Recursively convert any datetime objects to ISO strings so JSON serialization never fails."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(i) for i in obj]
    return obj


def run():
    try:
        _run()
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        # Always write ALL three output files so git add never fails on missing paths.
        docs_dir = Path(__file__).parent / "docs"
        docs_dir.mkdir(exist_ok=True)
        now = datetime.now(timezone.utc).isoformat()
        fallback = {
            "generated_at": now,
            "error": str(e),
            "market_sentiment": "UNKNOWN",
            "buy_signals": [],
            "sell_signals": [],
            "watch_list": [],
            "notable_trends": [],
            "rok_message": "Intelligence update unavailable — will retry shortly.",
        }
        (docs_dir / "intel_report.json").write_text(
            json.dumps(fallback, cls=_Encoder, indent=2), encoding="utf-8"
        )
        # Write history.json stub if it doesn't exist yet
        history_path = docs_dir / "history.json"
        if not history_path.exists():
            history_path.write_text(json.dumps({"runs": []}, indent=2), encoding="utf-8")
        # Write prices.json stub if it doesn't exist yet
        prices_path = docs_dir / "prices.json"
        if not prices_path.exists():
            prices_path.write_text(json.dumps({}), encoding="utf-8")
        logger.info("Wrote fallback output files")


def _build_weekly_bot_report(docs_dir):
    """Build a weekly performance summary from trades.json and equity.json."""
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    week_ago = now - timedelta(days=7)

    trades_path = docs_dir / "trades.json"
    equity_path = docs_dir / "equity.json"

    if not trades_path.exists():
        return None

    try:
        td = json.loads(trades_path.read_text())
    except Exception:
        return None

    all_trades = td.get("trades", [])
    lp = td.get("bot_learned_params", {})
    neurons_total = td.get("neurons_total", 630)
    neurons_active = td.get("neurons_active", 0)

    # Filter to this week's closed trades (SELL / COVER actions with pnl)
    week_trades = []
    for t in all_trades:
        if t.get("action") not in ("SELL", "SELL_HALF", "COVER"):
            continue
        ts = t.get("timestamp") or t.get("time") or ""
        try:
            trade_dt = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None
        except Exception:
            trade_dt = None
        if trade_dt and trade_dt >= week_ago:
            week_trades.append(t)

    wins = [t for t in week_trades if (t.get("pnl") or 0) > 0]
    losses = [t for t in week_trades if (t.get("pnl") or 0) <= 0]
    total_pnl = round(sum(t.get("pnl", 0) or 0 for t in week_trades), 2)
    win_rate = round(len(wins) / len(week_trades) * 100, 1) if week_trades else 0
    avg_pnl = round(total_pnl / len(week_trades), 2) if week_trades else 0

    # Best and worst trades this week
    sorted_by_pnl = sorted(week_trades, key=lambda t: t.get("pnl", 0) or 0, reverse=True)
    best_trades = [{"ticker": t.get("ticker"), "pnl": t.get("pnl")} for t in sorted_by_pnl[:3]]
    worst_trades = [{"ticker": t.get("ticker"), "pnl": t.get("pnl")} for t in sorted_by_pnl[-3:] if (t.get("pnl") or 0) < 0]

    # Top active neurons (those with learned data)
    top_neurons = []
    neuron_map = {
        "vix_entry_perf": "N103 VIX Bracket",
        "entry_session_perf": "N104 Session Quality",
        "breadth_entry_perf": "N105 Market Breadth",
        "trend_template_tier_perf": "N108 Trend Template",
        "rvol_entry_tier_perf": "N109 RVOL Tier",
        "spy_vwap_entry_perf": "N111 SPY VWAP",
        "signal_density_perf": "N112 Signal Density",
        "ai_sentiment_tier_perf": "N113 AI Sentiment",
        "hold_duration_perf": "N114 Hold Duration",
        "mktcap_tier_perf": "N115 Market Cap",
        "vts_perf": "N117 VIX Term Structure",
        "macro_hold_perf": "N119 Macro Events",
        "pcr_entry_perf": "N120 Options PCR",
        "si_squeeze_perf": "N121 Short Squeeze",
        "dist_200ema_perf": "N122 200 EMA Dist",
        "sector_etf_strength_perf": "N123 Sector ETF",
        "spy_alignment_perf": "N124 SPY Alignment",
        "news_velocity_perf_v1": "N125 News Velocity",
        "gap_entry_perf": "N126 Gap Entry",
        "rs_tier_entry_perf": "N127 RS Rating",
        "entry_score_tier_perf": "N128 Score Tier",
        "exit_trigger_perf_v1": "N129 Exit Trigger",
        "stock_stability_perf": "N130 Stability",
        "sector_type_perf": "N147 Sector Type",
        "cap_style_perf": "N148 Cap Style",
        "futures_signal_perf": "N149 Futures Signal",
        "exit_hour_perf": "N150 Exit Hour",
        "entry_dow_perf": "N151 Day of Week",
        "vix_trend_perf": "N152 VIX Trend",
        "crowd_tier_perf": "N153 Port Crowding",
        "sector_50d_trend_perf": "N154 Sector 50d",
        "short_int_perf": "N155 Short Interest",
        "regime_duration_perf_v1": "N156 Regime Duration",
        "orb_quality_perf": "N157 ORB Quality",
        "catalyst_type_perf_v1": "N158 Catalyst Type",
        "spy_52wh_zone_perf": "N159 SPY 52w Zone",
        "breakout_age_perf": "N160 Breakout Age",
        "dollar_vol_perf": "N161 Dollar Volume",
        "streak_state_perf": "N162 Bot Streak",
        "score_regime_align_perf": "N163 Score-Regime",
        "sector_mom_accel_perf": "N164 Sector Accel",
        "market_correl_perf": "N165 Mkt Correlation",
        "estimate_revision_perf": "N166 Est Revisions",
        "news_sent_mom_perf": "N167 Sentiment Mom",
        "tech_confluence_perf": "N168 Tech Confluence",
        "breadth_direction_perf": "N169 Breadth Dir",
        "risk_rotation_perf": "N170 Risk Rotation",
        "hold_time_perf": "N171 Hold Time",
        "exit_trigger_perf": "N172 Exit Trigger",
        "pos_size_tier_perf": "N173 Position Size",
        "consec_loss_perf": "N174 Consec Losses",
        "open_gap_perf": "N175 Open Gap",
        "opex_week_perf": "N176 OpEx Week",
        "sector_rs_phase_perf": "N177 Sector RS",
        "trade_cadence_perf": "N178 Trade Cadence",
        "spy_intraday_perf": "N179 SPY Intraday",
        "entry_score_decile_perf": "N180 Score Decile",
        "atr_pct_perf": "N181 ATR Percent",
        "spy_200d_position_perf": "N182 SPY 200d Pos",
        "volume_surge_state_perf": "N183 RVOL Surge",
        "float_size_perf": "N184 Float Size",
        "momentum_quality_perf": "N185 Mom Quality",
        "sector_news_flow_perf": "N186 Sector News",
        "morning_star_time_perf": "N187 Morning Star",
        "support_quality_perf": "N188 Support Qual",
        "relative_perf_1w_perf": "N189 Rel Perf 1W",
        "pre_market_action_perf": "N190 Pre-Market",
        "fed_week_perf": "N191 Fed Week",
        "earnings_season_perf": "N192 Earnings Season",
        "spy_rsi_zone_perf": "N193 SPY RSI Zone",
        "volume_vs_avg30_perf": "N194 Vol vs Avg30",
        "stock_beta_tier_perf": "N195 Beta Tier",
        "price_vs_vwap_perf": "N196 Price vs VWAP",
        "seasonal_month_perf": "N197 Seasonal Month",
        "market_breadth_level_perf": "N198 Breadth Level",
        "spy_gap_vs_stock_perf": "N199 Gap Diverge",
        "institutional_quality_perf": "N200 Inst Quality",
        "macro_shock_perf": "N201 Macro Shock",
        "earnings_surprise_direction_perf": "N202 EPS Surprise",
        "trend_age_days_perf": "N203 Trend Age",
        "dist_52wk_high_perf": "N204 Dist 52wk Hi",
        "adv_decline_ratio_perf": "N205 A/D Ratio",
        "option_implied_move_perf": "N206 Impl Move",
        "relative_volume_early_perf": "N207 Early RVOL",
        "bond_yield_direction_perf": "N208 Bond Yields",
        "social_sentiment_score_perf": "N209 Social Sent",
        "put_call_ratio_perf_v1": "N210 Put/Call Ratio",
        "market_cap_regime_perf": "N211 Cap Regime",
        "gold_signal_perf": "N212 Gold Signal",
        "sector_concentration_perf": "N213 Sector Conc",
        "entry_premium_count_perf": "N214 Premium Count",
        "daily_drawdown_state_perf": "N215 Daily DD State",
        "adv_decline_line_perf": "N216 A/D Line",
        "yield_curve_perf": "N217 Yield Curve",
        "cross_asset_momentum_perf": "N218 Cross Asset",
        "technical_pattern_strength_perf": "N219 Pattern Str",
        "position_duration_target_perf": "N220 Duration Target",
        "dollar_index_perf": "N221 Dollar Index",
        "sector_etf_momentum_perf": "N222 Sector ETF Mom",
        "position_count_at_entry_perf": "N223 Pos Count",
        "spy_5d_trend_perf": "N224 SPY 5d Trend",
        "vix_regime_perf": "N225 VIX Regime",
        "entry_hour_bucket_perf": "N226 Entry Hour",
        "consecutive_wins_perf": "N227 Consec Wins",
        "market_open_momentum_perf": "N228 Open Momentum",
        "spy_vs_vix_diverge_perf": "N229 SPY/VIX Div",
        "ticker_prior_day_gap_perf": "N230 Prior Day Gap",
        "spy_rsi_5d_change_perf": "N231 SPY RSI 5d",
        "market_internals_score_perf": "N232 Internals Score",
        "position_age_at_exit_perf": "N233 Hold Duration",
        "ticker_rs_rating_tier_perf": "N234 RS Rating Tier",
        "spy_50d_vs_200d_perf": "N235 SPY 50/200d",
        "sector_rotation_strength_perf": "N236 Sector Rotation",
        "news_catalyst_urgency_perf": "N237 Catalyst Urgency",
        "earnings_proximity_perf": "N238 Earnings Prox",
        "stop_distance_pct_perf": "N239 Stop Distance",
        "premarket_gap_direction_perf": "N240 PreMkt Gap Dir",
        "vol_contraction_entry_perf": "N241 Vol Contraction",
        "prior_week_trend_perf": "N242 Prior Wk Trend",
        "market_phase_perf": "N243 Market Phase",
        "intraday_reversal_perf_v1": "N244 Intraday Reversal",
        "sector_etf_vs_spy_perf": "N245 Sector vs SPY",
        "adv_decline_ratio_today_perf": "N246 A/D Ratio Today",
        "entry_near_high_low_perf": "N247 Entry Near Hi/Lo",
        "catalyst_sector_match_perf": "N248 Catalyst Sector",
        "recent_buy_count_perf": "N249 Recent Buy Count",
        "macro_stress_index_perf": "N250 Macro Stress",
        "overnight_gap_follow_perf": "N251 Overnight Gap",
        "spy_breadth_thrust_perf": "N252 Breadth Thrust",
        "tick_extreme_perf": "N253 TICK Extreme",
        "sector_leader_lag_perf": "N254 Sector Leader/Lag",
        "put_call_ratio_perf_v2": "N255 Put/Call Ratio",
        "options_expiry_week_perf": "N256 OPEX Week",
        "momentum_divergence_perf": "N257 Mom Divergence",
        "gap_fill_tendency_perf_v1": "N258 Gap Fill",
        "earnings_season_phase_perf": "N259 Earnings Season",
        "liquidity_score_perf": "N260 Liquidity Score",
        "spy_close_vs_open_perf": "N261 SPY Close vs Open",
        "atr_regime_perf": "N262 ATR Regime",
        "consecutive_spy_up_perf": "N263 Consec SPY Up",
        "vwap_position_perf": "N264 VWAP Position",
        "weekly_rs_trend_perf": "N265 Weekly RS Trend",
        "pre_market_volume_perf": "N266 Pre-Market Vol",
        "market_cap_tier_perf": "N267 Market Cap Tier",
        "trend_acceleration_perf": "N268 Trend Acceleration",
        "sector_breadth_perf_v1": "N269 Sector Breadth",
        "time_since_last_trade_perf_v1": "N270 Trade Pace",
        "market_internals_trend_perf": "N271 Internals Trend",
        "news_volume_perf": "N272 News Volume",
        "spy_distance_from_52w_high_perf": "N273 SPY 52w Hi Dist",
        "position_concentration_perf": "N274 Position Conc",
        "regime_duration_perf": "N275 Regime Duration",
        "crypto_correlation_perf": "N276 Crypto Correlation",
        "intraday_trend_persistence_perf": "N277 Intraday Persist",
        "entry_quality_score_perf": "N278 Entry Quality",
        "sector_momentum_rank_perf": "N279 Sector Rank",
        "fed_meeting_week_perf": "N280 Fed Week",
        "orb_15min_perf":                 "N281 ORB Breakout",
        "spy_rsi_overbought_perf": "N282 SPY RSI OB/OS",
        "ticker_earnings_beat_streak_perf": "N283 Earnings Streak",
        "holding_cost_vs_cash_perf": "N284 Cash vs Invested",
        "spy_volume_vs_avg_perf": "N285 SPY Volume",
        "technical_score_bucket_perf": "N286 Score Bucket",
        "day_of_week_perf": "N287 Day of Week",
        "market_hours_quadrant_perf": "N288 Market Quadrant",
        "position_pnl_before_entry_perf": "N289 Port P&L State",
        "ticker_beta_bucket_perf": "N290 Beta Bucket",
        "relative_volume_quality_perf": "N291 Rel Volume",
        "price_above_200ma_perf": "N292 Price vs 200MA",
        "spy_trend_strength_perf": "N293 SPY Trend",
        "rsi_at_entry_perf": "N294 RSI at Entry",
        "gap_overnight_direction_perf": "N295 Gap Direction",
        "vix_level_perf": "N296 VIX Level",
        "ticker_momentum_perf": "N297 Ticker Momentum",
        "atr_as_pct_price_perf": "N298 ATR % Price",
        "consecutive_win_streak_perf": "N299 Win Streak",
        "open_position_count_perf": "N300 Position Count",
        "spread_vs_atr_perf": "N301 Spread vs ATR",
        "price_vs_open_perf": "N302 Price vs Open",
        "sector_vs_spy_today_perf": "N303 Sector vs SPY",
        "portfolio_drawdown_perf": "N304 Port Drawdown",
        "buy_score_vs_threshold_perf": "N305 Score vs Threshold",
        "time_of_day_bucket_perf": "N306 Time of Day",
        "spy_vs_qqq_divergence_perf": "N307 SPY vs QQQ",
        "entry_after_halt_perf": "N308 Post-Halt Entry",
        "macro_day_risk_perf": "N309 Macro Day Risk",
        "regime_quality_combined_perf": "N310 Regime Quality",
        "entry_rank_in_session_perf": "N311 Session Entry Rank",
        "vwap_distance_pct_perf": "N312 VWAP Distance",
        "atr_multiple_gain_potential_perf": "N313 ATR Reward Risk",
        "mfi_zone_N314_perf": "N314 MFI Zone",
        "recent_market_breadth_perf": "N315 Market Breadth",
        "price_gap_size_perf": "N316 Gap Size",
        "sector_strength_score_perf": "N317 Sector Strength",
        "earnings_distance_perf": "N318 Earnings Distance",
        "portfolio_win_rate_trend_perf": "N319 Win Rate Trend",
        "position_size_bucket_perf": "N320 Position Size",
        "chg_ytd_bucket_perf": "N321 YTD Return",
        "market_leader_flag_perf": "N322 Market Leader",
        "macd_cross_state_perf": "N323 MACD Cross",
        "bb_position_perf": "N324 BB Position",
        "consecutive_green_days_perf": "N325 Green Streak",
        "sma50_slope_perf": "N326 SMA50 Slope",
        "entry_at_support_perf": "N327 Support Bounce",
        "psar_bull_entry_perf": "N328 PSAR Entry",
        "adx_trend_strength_perf": "N329 ADX Strength",
        "volume_trend_3d_perf_v1": "N330 Volume Trend",
        "sector_rotation_signal_perf": "N331 Sector Rotation",
        "spy_above_200ma_perf": "N332 SPY vs 200MA",
        "fear_greed_bucket_perf": "N333 Fear/Greed",
        "short_float_bucket_perf": "N334 Short Float",
        "iv_rank_bucket_perf": "N335 IV Rank",
        "catalyst_type_perf": "N336 Catalyst Type",
        "trend_age_bucket_perf": "N337 Trend Age",
        "index_divergence_perf": "N338 Index Divergence",
        "opening_gap_follow_perf": "N339 Gap Follow",
        "earnings_revision_perf": "N340 EPS Revision",
        "pm_gap_v1_perf":         "N341 Pre-Market Gap",
        "regime_transition_perf": "N342 Regime Transition",
        "ticker_age_bucket_perf": "N343 Ticker Age",
        "spy_options_oi_perf": "N344 SPY Options Flow",
        "breakout_confirmation_perf": "N345 Breakout Confirm",
        "portfolio_heat_perf": "N346 Portfolio Heat",
        "earnings_momentum_perf": "N347 Earnings Momentum",
        "sector_breadth_perf": "N348 Sector Breadth",
        "volatility_contraction_perf": "N349 Vol Contraction",
        "time_since_last_trade_perf": "N350 Time Since Trade",
        "float_size_bucket_perf": "N351 Float Size",
        "news_velocity_perf": "N352 News Velocity",
        "relative_pe_perf": "N353 Relative PE",
        "intraday_reversal_perf": "N354 Intraday Reversal",
        "market_breadth_score_perf": "N355 Market Breadth Score",
        "multi_timeframe_trend_perf": "N356 Multi-TF Trend",
        "smart_money_indicator_perf": "N357 Smart Money",
        "entry_price_vs_vwap_perf": "N358 Entry vs VWAP",
        "catalyst_recency_perf": "N359 Catalyst Recency",
        "sector_momentum_quality_perf": "N360 Sector Momentum",
        "liquidity_tier_perf": "N361 Liquidity Tier",
        "trend_reversal_signal_perf": "N362 Trend Reversal",
        "gap_size_bucket_perf": "N363 Gap Size",
        "sector_etf_vs_spy_today_perf": "N364 Sector vs SPY",
        "price_acceleration_perf": "N365 Price Acceleration",
        "opening_strength_perf": "N366 Opening Strength",
        "vwap_reclaim_perf": "N367 VWAP Reclaim",
        "institutional_size_entry_perf": "N368 Institutional Size",
        "earnings_drift_perf": "N369 Earnings Drift",
        "regime_momentum_sync_perf": "N370 Regime Sync",
        "pre_market_vs_prior_close_perf": "N371 Pre-Mkt vs Close",
        "daily_atr_move_perf": "N372 Daily ATR Move",
        "sector_weekly_rank_perf": "N373 Sector Weekly Rank",
        "short_squeeze_potential_perf": "N374 Short Squeeze Potential",
        "entry_vs_52w_high_perf": "N375 Entry vs 52w High",
        "spy_morning_action_perf": "N376 SPY Morning Action",
        "position_overlap_perf": "N377 Position Overlap",
        "news_impact_direction_perf": "N378 News Impact Dir",
        "rsi_vs_sector_rsi_perf": "N379 RSI vs Sector",
        "pre_entry_rvol_quality_perf": "N380 Pre-Entry RVOL",
        "trend_quality_score_perf": "N381 Trend Quality Score",
        "option_flow_imbalance_perf": "N382 Option Flow Imbalance",
        "sector_leadership_quality_perf": "N383 Sector Leadership Quality",
        "entry_candle_quality_perf": "N384 Entry Candle Quality",
        "macro_backdrop_perf": "N385 Macro Backdrop",
        "price_vs_ma20_perf": "N386 Price vs MA20",
        "breakout_volume_quality_perf": "N387 Breakout Volume Quality",
        "regime_spy_alignment_perf": "N388 Regime SPY Alignment",
        "entry_time_quality_perf": "N389 Entry Time Quality",
        "position_risk_reward_entry_perf": "N390 Position R/R Entry",
        "intraday_high_quality_perf": "N391 Intraday High Quality",
        "sector_etf_gap_perf": "N392 Sector ETF Gap",
        "spy_open_vs_close_perf": "N393 SPY Open vs Close",
        "prior_day_range_perf": "N394 Prior Day Range",
        "entry_vs_sector_beta_perf": "N395 Entry vs Sector Beta",
        "market_internals_quality_perf": "N396 Market Internals Quality",
        "ema_stack_quality_perf": "N397 EMA Stack Quality",
        "vol_expansion_at_entry_perf": "N398 Vol Expansion at Entry",
        "news_age_quality_perf": "N399 News Age Quality",
        "technical_score_quality_perf": "N400 Technical Score Quality",
        "relative_strength_vs_market_perf": "N401 Relative Strength vs Market",
        "sector_rotation_phase_perf": "N402 Sector Rotation Phase",
        "morning_momentum_quality_perf": "N403 Morning Momentum Quality",
        "volume_profile_entry_perf": "N404 Volume Profile Entry",
        "institutional_flow_quality_perf": "N405 Institutional Flow Quality",
        "gap_quality_context_perf": "N406 Gap Quality Context",
        "support_confluence_quality_perf": "N407 Support Confluence Quality",
        "catalyst_quality_tier_perf": "N408 Catalyst Quality Tier",
        "pre_market_volume_quality_perf": "N409 Pre-Market Volume Quality",
        "conviction_score_tier_perf": "N410 Conviction Score Tier",
        "sector_concentration_risk_perf": "N411 Sector Concentration Risk",
        "entry_rsi_context_perf": "N412 Entry RSI Context",
        "spy_5d_momentum_perf": "N413 SPY 5-Day Momentum",
        "short_float_quality_perf": "N414 Short Float Quality",
        "sector_news_momentum_perf": "N415 Sector News Momentum",
        "weekly_close_quality_perf": "N416 Weekly Close Quality",
        "entry_spread_quality_perf": "N417 Entry Spread Quality",
        "pre_breakout_compression_perf": "N418 Pre-Breakout Compression",
        "sector_vs_spx_week_perf": "N419 Sector vs SPX Week",
        "position_size_quality_perf": "N420 Position Size Quality",
        "atr_expansion_entry_perf": "N421 ATR Expansion Entry",
        "relative_volume_surge_perf": "N422 Relative Volume Surge",
        "price_vs_vwap_distance_perf": "N423 Price vs VWAP Distance",
        "market_regime_duration_perf": "N424 Market Regime Duration",
        "entry_day_of_month_perf": "N425 Entry Day of Month",
        "sector_breadth_quality_perf": "N426 Sector Breadth Quality",
        "consecutive_green_entry_perf": "N427 Consecutive Green Entry",
        "bollinger_position_entry_perf": "N428 Bollinger Position Entry",
        "earnings_window_entry_perf": "N429 Earnings Window Entry",
        "liquidity_dollar_volume_perf": "N430 Liquidity Dollar Volume",
        "overnight_gap_entry_perf": "N431 Overnight Gap Entry",
        "iv_percentile_entry_perf": "N432 IV Percentile Entry",
        "price_momentum_5d_perf": "N433 Price Momentum 5D",
        "news_sentiment_score_perf_n434": "N434 News Sentiment Score",
        "short_interest_ratio_perf_n435": "N435 Short Interest Ratio",
        "analyst_revision_trend_perf": "N436 Analyst Revision Trend",
        "beta_regime_fit_perf": "N437 Beta Regime Fit",
        "days_since_earnings_perf": "N438 Days Since Earnings",
        "put_call_ratio_perf": "N439 Put/Call Ratio",
        "trend_strength_adx_perf": "N440 Trend Strength ADX",
        "market_cap_tier_entry_perf": "N441 Market Cap Tier",
        "sector_momentum_entry_perf": "N442 Sector Momentum",
        "spread_quality_entry_perf": "N443 Spread Quality",
        "price_above_open_perf": "N444 Price Above Open",
        "week_vs_sector_perf_entry": "N445 Week vs Sector",
        "earnings_surprise_hist_perf": "N446 Earnings Surprise History",
        "fund_ownership_change_perf": "N447 Fund Ownership Change",
        "price_range_position_perf": "N448 Price Range Position",
        "momentum_quality_score_perf": "N449 Momentum Quality Score",
        "sector_relative_strength_perf_n450": "N450 Sector Relative Strength",
        "gap_fill_tendency_perf":         "N451 Gap Fill Tendency",
        "volume_trend_3d_perf":           "N452 Volume Trend 3D",
        "institutional_activity_perf":    "N453 Institutional Activity",
        "squeeze_setup_perf":             "N454 TTM Squeeze Setup",
        "price_discovery_perf":           "N455 Price Discovery",
        "catalyst_freshness_perf":        "N456 Catalyst Freshness",
        "options_unusualness_perf":       "N457 Options Unusual Activity",
        "relative_volume_entry_perf":     "N458 Relative Volume Entry",
        "vwap_position_entry_perf":       "N459 VWAP Position Entry",
        "daily_range_quality_perf":       "N460 Daily Range Quality",
        "opening_drive_quality_perf":     "N461 Opening Drive Quality",
        "price_vs_ema50_entry_perf":      "N462 Price vs EMA50 Entry",
        "consec_green_days_perf":         "N463 Consecutive Green Days",
        "weekly_trend_quality_perf":      "N464 Weekly Trend Quality",
        "dist_from_52w_high_perf":        "N465 Distance from 52W High",
        "spy_vs_sector_entry_perf":       "N466 SPY vs Sector Entry",
        "atr_vol_expansion_perf":         "N467 ATR/Vol Expansion Entry",
        "multi_day_breakout_perf":        "N468 Multi-Day Breakout",
        "inside_bar_resolution_perf":     "N469 Inside Bar Resolution",
        "earnings_drift_days_perf":       "N470 Earnings Drift Days",
        "premarket_gap_size_perf":        "N471 Pre-Market Gap Size",
        "stock_pcr_entry_perf":           "N472 Stock Put/Call Ratio Entry",
        "monthly_momentum_perf":          "N473 Monthly Price Momentum",
        "breakout_52w_entry_perf":        "N474 52-Week Breakout Entry",
        "hist_vol_level_perf":            "N475 Historical Volatility Level",
        "avwap_dist_entry_perf":          "N476 AVWAP Distance at Entry",
        "w52_range_position_perf":        "N477 52-Week Range Position",
        "rs_line_new_high_perf":          "N478 RS Line New High",
        "rs_line_trending_perf":          "N479 RS Line Trend Direction",
        "sector_alpha_entry_perf":        "N480 Sector Alpha at Entry",
        "obv_slope_entry_perf":           "N481 OBV Slope at Entry",
        "macd_slope_entry_perf":          "N482 MACD Slope at Entry",
        "price_vs_200ema_entry_perf":     "N483 Price vs 200EMA Entry",
        "higher_lows_entry_perf":         "N484 Higher Lows Pattern",
        "adx_strength_entry_perf":        "N485 ADX Trend Strength",
        "lr_trend_quality_entry_perf":    "N486 Linear Regression Trend Quality",
        "stoch_position_entry_perf":      "N487 Stochastic Position Entry",
        "pe_ratio_tier_entry_perf":       "N488 P/E Ratio Tier",
        "analyst_upgrade_entry_perf":     "N489 Analyst Upgrade at Entry",
        "short_squeeze_entry_perf":       "N490 Short Squeeze Potential",
        "cci_level_entry_perf":           "N491 CCI Level at Entry",
        "williams_r_entry_perf":          "N492 Williams %R Entry",
        "pivot_point_entry_perf":         "N493 Pivot Point Position",
        "mfi_level_entry_perf":           "N494 Money Flow Index Level",
        "roc_momentum_entry_perf":        "N495 Rate of Change Momentum",
        "keltner_position_entry_perf":    "N496 Keltner Channel Position",
        "donchian_breakout_entry_perf":   "N497 Donchian Breakout Entry",
        "cmf_entry_perf":                 "N498 Chaikin Money Flow",
        "dmi_cross_entry_perf":           "N499 DMI Crossover Signal",
        "ichimoku_cloud_entry_perf":      "N500 Ichimoku Cloud Position",
        "elder_ray_entry_perf":           "N501 Elder Ray Bull/Bear Power",
        "aroon_signal_entry_perf":        "N502 Aroon Signal Entry",
        "chande_momentum_entry_perf":     "N503 Chande Momentum Oscillator",
        "dpo_signal_entry_perf":          "N504 DPO Signal Entry",
        "price_vs_vwap_deviation_perf":   "N505 Price vs VWAP Deviation",
        "accumulation_dist_perf":         "N506 Accumulation/Distribution Trend",
        "chandelier_exit_entry_perf":     "N507 Chandelier Exit Position",
        "ppo_signal_entry_perf":          "N508 PPO Signal Entry",
        "coppock_curve_entry_perf":       "N509 Coppock Curve Entry",
        "hull_ma_entry_perf":             "N510 Hull Moving Average Signal",
        "market_internals_entry_perf":    "N511 Market Internals Entry",
        "tape_speed_entry_perf":          "N512 Tape Speed Entry",
        "options_skew_entry_perf":        "N513 Options Skew Entry",
        "dark_pool_entry_perf":           "N514 Dark Pool Activity",
        "inst_own_v1_perf":                  "N515 Institutional Ownership",
        "float_rotation_entry_perf":      "N516 Float Rotation Entry",
        "news_sentiment_v2_perf":         "N517 News Sentiment Score v2",
        "social_momentum_entry_perf":     "N518 Social Momentum Entry",
        "earnings_surprise_entry_perf":   "N519 Earnings Surprise Entry",
        "guidance_revision_entry_perf":   "N520 Guidance Revision Entry",
        "liquidity_score_entry_perf":     "N521 Liquidity Score Entry",
        "sector_rotation_entry_perf":     "N522 Sector Rotation Entry",
        "macro_regime_entry_perf":        "N523 Macro Regime Entry",
        "correlation_spy_entry_perf":     "N524 SPY Correlation Entry",
        "beta_bucket_entry_perf":         "N525 Beta Bucket Entry",
        "price_vs_52w_entry_perf":        "N526 Price vs 52W Entry",
        "gap_fill_status_entry_perf":     "N527 Gap Fill Status Entry",
        "order_flow_imbalance_perf":      "N528 Order Flow Imbalance",
        "regime_volatility_entry_perf":   "N529 Regime Volatility Entry",
        "breadth_thrust_entry_perf":      "N530 Breadth Thrust Entry",
        "catalyst_freshness_entry_perf":  "N531 Catalyst Freshness Entry",
        "entry_timing_session_perf":      "N532 Entry Timing Session",
        "institutional_flow_entry_perf":  "N533 Institutional Flow Entry",
        "vwap_relationship_entry_perf":   "N534 VWAP Relationship Entry",
        "momentum_persistence_entry_perf":"N535 Momentum Persistence Entry",
        "put_call_skew_entry_perf":       "N536 Put/Call Skew Entry",
        "spy_momentum_entry_perf":        "N537 SPY Momentum Entry",
        "earnings_proximity_entry_perf":  "N538 Earnings Proximity Entry",
        "price_acceleration_entry_perf":  "N539 Price Acceleration Entry",
        "risk_reward_at_entry_perf":      "N540 Risk/Reward at Entry",
        "short_int_ratio_v2_perf":        "N541 Short Interest Ratio v2",
        "float_rotation_perf":            "N542 Float Rotation",
        "tick_trend_entry_perf":          "N543 NYSE Tick Trend Entry",
        "market_phase_entry_perf":        "N544 Market Phase Entry",
        "sector_rs_v2_perf":              "N545 Sector Relative Strength v2",
        "news_sentiment_v3_perf":         "N546 News Sentiment Score v3",
        "relative_volume_spike_perf":     "N547 Relative Volume Spike",
        "atr_expansion_perf":             "N548 ATR Expansion",
        "close_vs_range_perf":            "N549 Close vs Range",
        "consecutive_up_days_perf":       "N550 Consecutive Up Days",
        "pre_market_gap_perf":            "N551 Pre-Market Gap",
        "opening_range_breakout_perf":    "N552 Opening Range Breakout",
        "institutional_ownership_perf":   "N553 Institutional Ownership",
        "analyst_consensus_perf":         "N554 Analyst Consensus",
        "earnings_growth_rate_perf":      "N555 Earnings Growth Rate",
        "revenue_growth_rate_perf":       "N556 Revenue Growth Rate",
        "profit_margin_perf":             "N557 Profit Margin",
        "debt_to_equity_perf":            "N558 Debt-to-Equity Ratio",
        "buyback_activity_perf":          "N559 Buyback Activity",
        "dividend_yield_entry_perf":      "N560 Dividend Yield Entry",
        "price_acceleration_1d_perf":     "N561 Price Acceleration 1d",
        "market_breadth_entry_perf":      "N562 Market Breadth Entry",
        "spy_intraday_trend_perf":        "N563 SPY Intraday Trend",
        "volume_vs_50d_avg_perf":         "N564 Volume vs 50d Avg",
        "gap_and_go_quality_perf":        "N565 Gap and Go Quality",
        "sector_etf_5d_momentum_perf":    "N566 Sector ETF 5d Momentum",
        "congressional_buy_signal_perf":  "N567 Congressional Buy Signal",
        "insider_purchase_signal_perf":   "N568 Insider Purchase Signal",
        "social_buzz_velocity_perf":      "N569 Social Buzz Velocity",
        "earnings_beat_streak_perf":      "N570 Earnings Beat Streak",
        "options_iv_rank_perf":           "N571 Options IV Rank",
        "call_volume_surge_perf":         "N572 Call Volume Surge",
        "dark_pool_flow_perf":            "N573 Dark Pool Flow",
        "smart_money_flow_perf":          "N574 Smart Money Flow",
        "trending_sector_rotation_perf":  "N575 Trending Sector Rotation",
        "squeeze_momentum_perf":          "N576 Squeeze Momentum",
        "put_wall_proximity_perf":        "N577 Put Wall Proximity",
        "weekly_options_expiry_perf":     "N578 Weekly Options Expiry",
        "gamma_exposure_perf":            "N579 Gamma Exposure",
        "market_regime_vix_perf":         "N580 Market Regime VIX",
        "trend_following_score_perf":     "N581 Trend Following Score",
        "mean_reversion_setup_perf":      "N582 Mean Reversion Setup",
        "breakout_false_signal_perf":     "N583 Breakout Authenticity",
        "intraday_momentum_shift_perf":   "N584 Intraday Momentum Shift",
        "sector_news_catalyst_perf":      "N585 Sector News Catalyst",
        "analyst_upgrade_momentum_perf":  "N586 Analyst Upgrade Momentum",
        "earnings_estimate_revision_perf":"N587 EPS Estimate Revision",
        "technical_pattern_quality_perf": "N588 Technical Pattern Quality",
        "market_open_strength_perf":      "N589 Market Open Strength",
        "price_range_percentile_perf":    "N590 Price Range Percentile",
        "entry_session_quality_perf":     "N591 Entry Session Quality",
        "vwap_deviation_entry_perf":      "N592 VWAP Deviation Entry",
        "prior_day_close_relation_perf":  "N593 Prior Day Close Relation",
        "intraday_high_proximity_perf":   "N594 Intraday High Proximity",
        "float_size_perf":                "N595 Float Size",
        "short_squeeze_velocity_perf":    "N596 Short Squeeze Velocity",
        "tape_reading_entry_perf":        "N597 Tape Reading Entry",
        "options_open_interest_perf":     "N598 Options Open Interest",
        "weekly_trend_alignment_perf":    "N599 Weekly Trend Alignment",
        "momentum_score_bucket_perf":     "N600 Momentum Score Bucket",
        "bid_ask_spread_perf":            "N601 Bid-Ask Spread",
        "news_recency_perf":              "N602 News Recency",
        "sector_rotation_perf":           "N603 Sector Rotation",
        "pre_market_volume_perf":         "N604 Pre-Market Volume",
        "breakout_volume_confirm_perf":   "N605 Breakout Volume Confirm",
        "trend_day_type_perf":            "N606 Trend Day Type",
        "entry_candle_pattern_perf":      "N607 Entry Candle Pattern",
        "resistance_proximity_perf":      "N608 Resistance Proximity",
        "market_breadth_perf":            "N609 Market Breadth",
        "time_of_day_score_perf":         "N610 Time of Day Score",
        "consolidation_length_perf":      "N611 Consolidation Length",
        "catalyst_type_detail_perf":      "N612 Catalyst Type Detail",
        "price_action_quality_perf":      "N613 Price Action Quality",
        "market_cap_tier_perf":           "N614 Market Cap Tier",
        "options_flow_signal_perf":       "N615 Options Flow Signal",
        "institutional_filing_perf":      "N616 Institutional Filing",
        "relative_strength_vs_spy_perf":  "N617 Relative Strength vs SPY",
        "gap_fill_proximity_perf":        "N618 Gap Fill Proximity",
        "multi_timeframe_rsi_perf":       "N619 Multi-Timeframe RSI",
        "earnings_quality_perf":          "N620 Earnings Quality",
        "momentum_divergence_perf":        "N621 Momentum Divergence",
        "atr_expansion_at_entry_perf":     "N622 ATR Expansion at Entry",
        "opening_range_position_perf":     "N624 Opening Range Position",
        "volume_price_trend_perf":         "N626 Volume Price Trend",
        "gap_to_close_performance_perf":   "N627 Gap to Close Performance",
        "fear_greed_index_perf":           "N628 Fear Greed Index",
        "breakout_retest_perf":            "N629 Breakout Retest",
        "trend_line_proximity_perf":       "N630 Trend Line Proximity",
        "high_low_range_perf":             "N631 High-Low Range",
        "price_momentum_5d_perf":          "N632 Price Momentum 5D",
        "catalyst_sector_alignment_perf":  "N633 Catalyst Sector Alignment",
        "volume_consistency_perf":         "N634 Volume Consistency",
        "rsi_trend_alignment_perf":        "N635 RSI Trend Alignment",
        "order_block_proximity_perf":      "N636 Order Block Proximity",
        "liquidity_score_perf":            "N637 Liquidity Score",
        "news_catalyst_sentiment_perf":    "N638 News Catalyst Sentiment",
        "technical_score_trend_perf":      "N639 Technical Score Trend",
        "ema_stack_quality_perf":          "N640 EMA Stack Quality",
        "pivot_point_proximity_perf":      "N641 Pivot Point Proximity",
        "float_rotation_perf":             "N642 Float Rotation",
        "short_interest_perf":             "N643 Short Interest",
        "insider_activity_perf":           "N644 Insider Activity",
        "earnings_season_phase_perf":      "N645 Earnings Season Phase",
        "put_call_ratio_perf":             "N646 Put/Call Ratio",
        "vwap_distance_perf":              "N647 VWAP Distance",
        "day_of_week_perf":                "N648 Day of Week",
        "market_breadth_perf":             "N649 Market Breadth",
        "premarket_volume_perf":           "N650 Pre-Market Volume",
        "consecutive_up_days_perf":        "N651 Consecutive Up Days",
        "market_cap_momentum_perf":        "N652 Market Cap Momentum",
        "institutional_ownership_perf":    "N653 Institutional Ownership",
        "analyst_revision_perf":           "N654 Analyst Revision",
        "relative_volume_trend_perf":      "N655 Relative Volume Trend",
        "price_vs_sma200_perf":            "N656 Price vs SMA200",
        "earnings_revision_direction_perf":"N657 Earnings Revision Direction",
        "premarket_gap_size_perf":         "N658 Pre-Market Gap Size",
        "social_sentiment_velocity_perf":  "N659 Social Sentiment Velocity",
        "option_implied_move_perf":        "N660 Option Implied Move",
        "trend_age_perf":                  "N661 Trend Age",
        "reversal_candle_perf":            "N662 Reversal Candle",
        "weekly_momentum_perf":            "N663 Weekly Momentum",
        "relative_strength_rank_perf":     "N664 RS Rank",
        "catalyst_freshness_perf":         "N665 Catalyst Freshness",
        "institutional_accumulation_perf": "N666 Institutional Accumulation",
        "breakout_volume_confirmation_perf":"N667 Breakout Volume",
        "adx_trend_strength_perf":         "N668 ADX Trend Strength",
        "risk_reward_ratio_perf":          "N669 Risk/Reward Ratio",
        "sector_etf_momentum_perf":        "N670 Sector ETF Momentum",
    }
    for key, label in neuron_map.items():
        data = lp.get(key, [])
        if not isinstance(data, list) or not data:
            continue
        best = max(data, key=lambda x: x.get("win_rate", 50), default=None)
        if best:
            top_neurons.append({
                "neuron": label,
                "best_state": best.get("state", "?"),
                "win_rate": best.get("win_rate", 50),
                "samples": best.get("total", 0),
            })
    top_neurons.sort(key=lambda x: x["win_rate"], reverse=True)

    # Equity curve this week (from equity.json)
    equity_week = []
    try:
        if equity_path.exists():
            eq = json.loads(equity_path.read_text())
            snapshots = eq.get("snapshots", [])
            for snap in snapshots:
                snap_dt_str = snap.get("date") or snap.get("timestamp", "")
                try:
                    snap_dt = datetime.fromisoformat(snap_dt_str.replace("Z", "+00:00")) if snap_dt_str else None
                except Exception:
                    snap_dt = None
                if snap_dt and snap_dt >= week_ago:
                    equity_week.append({
                        "date": snap.get("date") or snap_dt_str[:10],
                        "equity": snap.get("equity"),
                        "spy_benchmark": snap.get("spy_benchmark"),
                    })
    except Exception:
        pass

    # Learn log highlights
    learn_log = lp.get("learn_log", [])[-10:]

    # ── Brain Analytics: synthesize neuron performance across all dimensions ──
    # Find best states for key timing/regime neurons (for the "when to trade" insight)
    def _best_state(key, min_samples=3):
        data = lp.get(key, [])
        if not isinstance(data, list):
            return None
        qualified = [x for x in data if isinstance(x, dict) and x.get("total", 0) >= min_samples]
        if not qualified:
            return None
        return max(qualified, key=lambda x: x.get("win_rate", 50))

    brain_analytics = {}
    # Best day of week to trade
    dow_best = _best_state("day_of_week_perf")
    if dow_best:
        brain_analytics["best_day_of_week"] = {
            "state": dow_best.get("state", ""),
            "win_rate": dow_best.get("win_rate", 0),
            "samples": dow_best.get("total", 0),
        }
    # Best session (morning/midday/afternoon)
    sess_best = _best_state("entry_session_quality_perf")
    if sess_best:
        brain_analytics["best_session"] = {
            "state": sess_best.get("state", ""),
            "win_rate": sess_best.get("win_rate", 0),
            "samples": sess_best.get("total", 0),
        }
    # Best momentum state
    mom_best = _best_state("momentum_score_bucket_perf")
    if mom_best:
        brain_analytics["best_momentum_state"] = {
            "state": mom_best.get("state", ""),
            "win_rate": mom_best.get("win_rate", 0),
            "samples": mom_best.get("total", 0),
        }
    # Best volume state
    vol_best = _best_state("volume_surge_perf", min_samples=3)
    if not vol_best:
        vol_best = _best_state("volume_consistency_perf", min_samples=3)
    if vol_best:
        brain_analytics["best_volume_state"] = {
            "state": vol_best.get("state", ""),
            "win_rate": vol_best.get("win_rate", 0),
            "samples": vol_best.get("total", 0),
        }
    # Best VIX regime
    vix_best = _best_state("market_regime_vix_perf")
    if vix_best:
        brain_analytics["best_vix_regime"] = {
            "state": vix_best.get("state", ""),
            "win_rate": vix_best.get("win_rate", 0),
            "samples": vix_best.get("total", 0),
        }
    # Top 3 neurons with highest learned edge (best_wr > 65%, min 5 samples)
    top_edge_neurons = sorted(
        [n for n in top_neurons if n.get("win_rate", 0) >= 65 and n.get("samples", 0) >= 5],
        key=lambda x: -x["win_rate"]
    )[:3]
    if top_edge_neurons:
        brain_analytics["top_edge_neurons"] = top_edge_neurons

    # Win rate by hour-of-day (from trade log, based on entry time)
    hour_wr = {}
    for t in all_trades:
        if t.get("action") not in ("SELL", "COVER"):
            continue
        entry_time_str = t.get("entry_time") or t.get("time") or ""
        pnl_t = t.get("pnl_pct")
        if not entry_time_str or pnl_t is None:
            continue
        try:
            from zoneinfo import ZoneInfo
            et = datetime.fromisoformat(entry_time_str.replace("Z", "+00:00"))
            hr = et.astimezone(ZoneInfo("America/New_York")).hour
            hour_wr.setdefault(hr, {"wins": 0, "total": 0})
            hour_wr[hr]["total"] += 1
            if pnl_t > 0:
                hour_wr[hr]["wins"] += 1
        except Exception:
            pass
    if hour_wr:
        best_hour = max(hour_wr.items(), key=lambda x: x[1]["wins"] / max(x[1]["total"], 1) if x[1]["total"] >= 2 else 0)
        brain_analytics["best_entry_hour"] = {
            "hour_et": best_hour[0],
            "win_rate": round(best_hour[1]["wins"] / best_hour[1]["total"] * 100),
            "samples": best_hour[1]["total"],
        }
        brain_analytics["hourly_win_rates"] = [
            {"hour": h, "win_rate": round(v["wins"] / v["total"] * 100), "samples": v["total"]}
            for h, v in sorted(hour_wr.items()) if v["total"] >= 2
        ]

    # ── Strategy Insights: performance attribution from all learned neurons ──
    # Identify top "alpha" neuron states (win rate vs base rate)
    strategy_insights = {"top_alpha": [], "top_drag": [], "key_combos": []}
    try:
        base_wr = win_rate  # use weekly win rate as baseline
        alpha_states = []
        for key, label in neuron_map.items():
            data = lp.get(key, [])
            if not isinstance(data, list):
                continue
            for state_data in data:
                if not isinstance(state_data, dict):
                    continue
                tot = state_data.get("total", 0)
                wr  = state_data.get("win_rate", 50)
                if tot < 4:
                    continue
                alpha = wr - base_wr
                alpha_states.append({
                    "neuron": label,
                    "state": state_data.get("state", ""),
                    "win_rate": wr,
                    "alpha": round(alpha, 1),
                    "samples": tot,
                    "avg_pnl": state_data.get("avg_pnl", 0),
                })
        alpha_states.sort(key=lambda x: (-x["alpha"], -x["win_rate"]))
        strategy_insights["top_alpha"] = alpha_states[:6]
        strategy_insights["top_drag"] = sorted(
            [s for s in alpha_states if s["alpha"] < -10],
            key=lambda x: x["alpha"]
        )[:4]
        # Top signal combinations (already computed in signal_synergy)
        syn_data = td.get("signal_synergy", {}) if 'td' in dir() else {}
        if syn_data:
            strategy_insights["key_combos"] = sorted(
                [{"pair": k, "win_rate": v.get("win_rate", 0), "samples": v.get("total", 0), "avg_pnl": v.get("avg_pnl", 0)}
                 for k, v in syn_data.items() if v.get("total", 0) >= 3 and v.get("win_rate", 0) >= 65],
                key=lambda x: (-x["win_rate"], -x["samples"])
            )[:6]
    except Exception:
        pass

    return {
        "generated_at": now.isoformat(),
        "period": "Last 7 days",
        "week_start": week_ago.strftime("%Y-%m-%d"),
        "week_end": now.strftime("%Y-%m-%d"),
        "trades_total": len(week_trades),
        "trades_wins": len(wins),
        "trades_losses": len(losses),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_pnl_per_trade": avg_pnl,
        "best_trades": best_trades,
        "worst_trades": worst_trades,
        "neurons_active": neurons_active,
        "neurons_total": neurons_total,
        "top_neurons": top_neurons[:10],
        "equity_curve": equity_week,
        "strategy_insights": strategy_insights,
        "learn_log": learn_log,
        "strategy_mode": td.get("strategy_mode", "SELECTIVE"),
        "recovery_mode": td.get("recovery_mode", False),
        "effective_min_score": td.get("effective_min_score"),
        "cross_asset_risk_off": td.get("cross_asset_risk_off", False),
        "portfolio_value": td.get("portfolio_value"),
        "drawdown_pct": td.get("drawdown_pct"),
        "profit_factor": td.get("profit_factor"),
        "brain_analytics": brain_analytics,
    }


def _run():
    from config import Config
    from scrapers import reddit_scraper, news_scraper, yahoo_finance, sec_scraper
    from scrapers import market_data, twitter_scraper
    from scrapers import stocktwits_scraper, technical_analysis, congressional_trades
    from analyzer import ticker_extractor, claude_analyzer
    from analyzer import sentiment as sentiment_mod

    logger.info("=" * 60)
    logger.info("ROK INTELLIGENCE PIPELINE START")
    logger.info("=" * 60)

    # ── Social scraping ──────────────────────────────────────────
    reddit_posts = _safe(
        reddit_scraper.scrape_all,
        Config.REDDIT_SUBREDDITS, Config.REDDIT_MAX_POSTS,
        default=list, label="Reddit",
    )
    news_articles = _safe(
        news_scraper.scrape_all,
        Config.NEWS_FEEDS, Config.NEWS_MAX_ITEMS,
        default=list, label="News",
    )
    twitter_posts = []
    if Config.TWITTER_ENABLED:
        twitter_posts = _safe(
            twitter_scraper.scrape_tweets,
            Config.TWITTER_BEARER_TOKEN,
            default=list, label="Twitter",
        )

    all_posts = reddit_posts + news_articles + twitter_posts

    # ── Market data ───────────────────────────────────────────────
    fear_greed    = _safe(market_data.get_fear_greed_index, default=dict, label="FearGreed")
    earnings_cal  = _safe(market_data.get_earnings_calendar, 7, default=list, label="Earnings")
    unusual_opts  = _safe(market_data.get_unusual_options_activity, default=list, label="Options")
    most_active   = _safe(market_data.get_most_active_stocks, default=list, label="MostActive")
    short_squeeze = _safe(market_data.get_short_squeeze_candidates, default=list, label="ShortSqueeze")
    market_indices= _safe(market_data.get_market_indices, default=dict, label="Indices")
    trending_yahoo= _safe(market_data.get_trending_on_yahoo, default=list, label="YahooTrending")
    put_call_ratio= _safe(market_data.get_put_call_ratio, default=dict, label="PutCall")
    market_breadth= _safe(market_data.get_market_breadth, default=dict, label="Breadth")

    # ── New data sources ──────────────────────────────────────────
    stocktwits_data = _safe(stocktwits_scraper.get_trending, default=list, label="StockTwits")
    congress_buys   = _safe(
        congressional_trades.get_congress_buys,
        Config.CONGRESS_DAYS_BACK,
        default=list, label="Congress",
    )

    # ── Sentiment + ticker extraction ─────────────────────────────
    all_posts = sentiment_mod.score_posts(all_posts)
    agg_sentiment = sentiment_mod.aggregate_sentiment(all_posts)
    top_tickers = ticker_extractor.top_tickers(all_posts, n=40)

    extra = set()
    for s in trending_yahoo[:15] + most_active[:15]:
        t = (s.get("ticker") or "").strip().upper()
        if t and t.isalpha() and len(t) <= 5:
            extra.add(t)
    for s in stocktwits_data[:20]:
        t = (s.get("ticker") or "").strip().upper()
        if t and t.isalpha() and len(t) <= 5:
            extra.add(t)
    for c in congress_buys[:10]:
        extra.add(c["ticker"])

    seen = {t for t, _ in top_tickers}
    seed = _safe(yahoo_finance.get_trending_tickers, default=list, label="YahooTickers")
    ticker_list = list(dict.fromkeys(
        [t for t, _ in top_tickers]
        + [t for t in extra if t not in seen]
        + [t for t in (seed or []) if t not in seen and t not in extra]
    ))[:60]

    ticker_sentiment = sentiment_mod.per_ticker_sentiment(all_posts, ticker_list[:30])

    # ── Stock data ────────────────────────────────────────────────
    stock_data = []
    for ticker in ticker_list[:60]:
        data = _safe(yahoo_finance.get_stock_data, ticker, default=lambda: None, label=f"Stock:{ticker}")
        if data:
            data["sentiment"] = ticker_sentiment.get(ticker, {})
            stock_data.append(data)
    logger.info(f"Stock data: {len(stock_data)} tickers")

    # ── Technical analysis ────────────────────────────────────────
    ta_tickers = ticker_list[:Config.TA_MAX_TICKERS]
    ta_data = _safe(
        technical_analysis.analyze_multiple,
        ta_tickers, Config.TA_MAX_TICKERS,
        default=dict, label="TechnicalAnalysis",
    )
    ta_setups = technical_analysis.find_setups(ta_data) if ta_data else []

    # ── SEC filings ───────────────────────────────────────────────
    sec_filings = _safe(
        lambda: sec_scraper.get_recent_insider_trades(7) + sec_scraper.get_recent_8k_filings(7),
        default=list, label="SEC",
    )
    insider_buys = _safe(sec_scraper.get_insider_buys, 14, default=list, label="InsiderBuys")

    # ── Load history ──────────────────────────────────────────────
    docs_dir = Path(__file__).parent / "docs"
    docs_dir.mkdir(exist_ok=True)
    history_path = docs_dir / "history.json"
    history = {"runs": []}
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text())
        except Exception:
            pass

    # ── Load live trading data (positions + scan top + market context) ─
    trades_path = docs_dir / "trades.json"
    current_positions = []
    last_scan_top = []
    live_market_context = {}
    try:
        if trades_path.exists():
            td = json.loads(trades_path.read_text())
            current_positions = td.get("positions", [])
            last_scan_top     = td.get("last_scan_top", [])
            # Extract live market context for richer AI prompt
            # Extract RS leaders and EMA21 setups from last scan
            _rs_leaders = sorted(
                [e for e in last_scan_top if (e.get("rs_rating") or 50) >= 80],
                key=lambda e: -(e.get("rs_rating") or 50)
            )[:6]
            _ema21_setups = [e["ticker"] for e in last_scan_top if e.get("ema21_pullback")][:5]
            _pocket_pivots = [e["ticker"] for e in last_scan_top if e.get("pocket_pivot")][:5]
            _htf_stocks = sorted(
                [e for e in last_scan_top if e.get("htf")],
                key=lambda e: -(e.get("htf_consec") or 0)
            )[:4]
            _tt8_stocks = [e["ticker"] for e in last_scan_top if e.get("tt_full")][:5]
            _tt_leaders = sorted(
                [e for e in last_scan_top if (e.get("trend_template") or 0) >= 6],
                key=lambda e: -(e.get("trend_template") or 0)
            )[:6]
            live_market_context = {
                "market_open":     td.get("market_open"),
                "timing_quality":  td.get("timing_quality"),
                "day_type":        td.get("day_type"),
                "day_efficiency":  td.get("day_efficiency"),
                "strategy_hint":   td.get("strategy_hint"),
                "vts_regime":      td.get("regime", {}).get("vts_regime"),
                "dxy_level":       td.get("regime", {}).get("dxy_level"),
                "dxy_5d":          td.get("regime", {}).get("dxy_5d"),
                "tnx_level":       td.get("regime", {}).get("tnx_level"),
                "tnx_5d":          td.get("regime", {}).get("tnx_5d"),
                "rate_environment": td.get("regime", {}).get("rate_environment"),
                "effective_min_score": td.get("effective_min_score"),
                "win_rate":        td.get("win_rate"),
                "drawdown_pct":    td.get("drawdown_pct"),
                "drawdown_halt":   td.get("drawdown_halt", False),
                "regime_max_pos":  td.get("regime_max_pos", 12),
                "profit_factor":   td.get("profit_factor"),
                "portfolio_beta":  td.get("portfolio_beta"),
                "portfolio_heat":  td.get("portfolio_heat"),
                "market_quality":  td.get("market_quality"),
                "scan_breadth_pct": td.get("scan_breadth_pct"),
                "portfolio_concentration": td.get("portfolio_concentration", {}),
                "sector_etf_trends": td.get("sector_etf_trends", {}),
                "rs_leaders":      [{"ticker": e["ticker"], "rs_rating": e.get("rs_rating"), "score": e.get("score")} for e in _rs_leaders],
                "ema21_setups":    _ema21_setups,
                "pocket_pivots":   _pocket_pivots,
                "htf_stocks":      [{"ticker": e["ticker"], "htf_consec": e.get("htf_consec", 0)} for e in _htf_stocks],
                "tt8_stocks":      _tt8_stocks,
                "tt_leaders":      [{"ticker": e["ticker"], "trend_template": e.get("trend_template", 0)} for e in _tt_leaders],
                "weekend_watchlist": td.get("weekend_watchlist", [])[:10],
                "bot_conviction":   td.get("bot_conviction", 0),
                "strategy_mode":    td.get("strategy_mode", ""),
                "neurons_active":   td.get("neurons_active", 0),
                "neurons_total":    td.get("neurons_total", 630),
                "intraday_wins":    td.get("intraday_wins", 0),
                "intraday_losses":  td.get("intraday_losses", 0),
                "loss_streak":      td.get("loss_streak", 0),
                "daily_risk_mult":  td.get("daily_risk_mult", 1.0),
                "last_decision":    td.get("last_decision", ""),
                "next_run_utc":     td.get("next_run_utc", ""),
                "bot_brain_summary": td.get("bot_brain_summary", ""),
                "regime_name":      td.get("regime", {}).get("regime", "neutral"),
                "vix":              td.get("regime", {}).get("vix", 0),
                "market_open_plan":        td.get("market_open_plan", {}),
                "next_market_open":         td.get("next_market_open", ""),
                "position_news":            td.get("position_news", {}),
                "portfolio_attribution":    td.get("portfolio_attribution", {}),
                "portfolio_stress_test":    td.get("portfolio_stress_test", {}),
                "morning_brief":            td.get("morning_brief", {}),
                "exit_intelligence":        td.get("exit_intelligence", {}),
                "sector_performance":       {k: {"win_rate": v.get("win_rate",0), "total": v.get("total",0), "avg_pnl": v.get("avg_pnl",0)} for k, v in td.get("sector_performance", {}).items() if v.get("total",0) >= 2},
                "weekend_watchlist_scored": [w for w in td.get("weekend_watchlist", []) if w.get("score")],
                # Synaptic intelligence: top learned signal pairs and triplets
                "top_synapses": sorted(
                    [{"pair": k, "wr": v.get("win_rate",0), "avg_pnl": v.get("avg_pnl",0), "n": v.get("total",0)}
                     for k, v in td.get("signal_synergy", {}).items() if v.get("total",0) >= 2 and v.get("win_rate",0) >= 55],
                    key=lambda x: (-x["wr"], -x["n"])
                )[:12],
                "top_triplets": sorted(
                    [{"combo": k, "wr": v.get("win_rate",0), "avg_pnl": v.get("avg_pnl",0), "n": v.get("total",0)}
                     for k, v in td.get("signal_triplets", {}).items() if v.get("total",0) >= 2 and v.get("win_rate",0) >= 60],
                    key=lambda x: (-x["wr"], -x["n"])
                )[:8],
                # Next entry conditions and regime state
                "next_entry_conditions": td.get("next_entry_conditions", {}),
                "smart_alerts": td.get("smart_alerts", []),
                "last_scan_top": td.get("last_scan_top", [])[:9],
                # Top performing neurons (for Brain heatmap in dashboard)
                "top_neurons": sorted(
                    [{"neuron": neuron_map.get(k, k), "key": k, "win_rate": max(itm.get("win_rate",0) for itm in v) if isinstance(v,list) and v else 0,
                      "best_state": (max(v, key=lambda x: x.get("win_rate",0)) if isinstance(v,list) and v else {}).get("state",""),
                      "total": sum(itm.get("total",0) for itm in v) if isinstance(v,list) else 0}
                     for k, v in lp.items() if isinstance(v,list) and any(isinstance(x,dict) and x.get("total",0)>=3 for x in v)],
                    key=lambda x: -x["win_rate"]
                )[:15],
                # Brain analytics: timing, momentum, regime insights from learned neurons
                "brain_analytics": {
                    k: {
                        "best_state": v.get("state", ""),
                        "win_rate": v.get("win_rate", 0),
                        "samples": v.get("samples", v.get("total", 0)),
                    } if isinstance(v, dict) and "state" in v else v
                    for k, v in {
                        "best_day_of_week":   next(
                            ({"state": x.get("state"), "win_rate": x.get("win_rate",0), "samples": x.get("total",0)}
                             for x in sorted([x for x in lp.get("day_of_week_perf",[]) if isinstance(x,dict) and x.get("total",0)>=3],
                                             key=lambda x: -x.get("win_rate",0))[:1]), None),
                        "best_session":       next(
                            ({"state": x.get("state"), "win_rate": x.get("win_rate",0), "samples": x.get("total",0)}
                             for x in sorted([x for x in lp.get("entry_session_quality_perf",[]) if isinstance(x,dict) and x.get("total",0)>=3],
                                             key=lambda x: -x.get("win_rate",0))[:1]), None),
                        "best_vix_regime":    next(
                            ({"state": x.get("state"), "win_rate": x.get("win_rate",0), "samples": x.get("total",0)}
                             for x in sorted([x for x in lp.get("market_regime_vix_perf",[]) if isinstance(x,dict) and x.get("total",0)>=3],
                                             key=lambda x: -x.get("win_rate",0))[:1]), None),
                    }.items() if v is not None
                },
            }
            logger.info(f"Loaded {len(current_positions)} positions, {len(last_scan_top)} scan candidates, {len(td.get('weekend_watchlist', []))} watchlist items from trades.json")
    except Exception as _te:
        logger.warning(f"Could not load trades.json: {_te}")

    # Include held ticker symbols in the analysis universe
    held_tickers = [p.get("ticker", "") for p in current_positions if p.get("ticker")]
    if held_tickers:
        held_set = set(held_tickers)
        # Prepend held tickers so AI analysis covers them specifically
        for ht in reversed(held_tickers):
            if ht not in {t for t, _ in top_tickers}:
                top_tickers = [(ht, 10)] + top_tickers  # high weight for held positions
        logger.info(f"Added held positions to analysis: {', '.join(held_tickers)}")

    # ── AI Analysis ───────────────────────────────────────────────
    logger.info("Calling Claude AI...")
    analysis = None

    if Config.ANTHROPIC_API_KEY:
        analysis = _safe(
            lambda: claude_analyzer.run_analysis(
                api_key=Config.ANTHROPIC_API_KEY,
                model=Config.CLAUDE_MODEL,
                ticker_mentions=top_tickers,
                reddit_posts=reddit_posts,
                news_articles=news_articles,
                stock_data=stock_data,
                sec_filings=sec_filings,
                fear_greed=fear_greed,
                earnings_calendar=earnings_cal,
                unusual_options=unusual_opts,
                short_squeeze_candidates=short_squeeze,
                market_indices=market_indices,
                aggregate_sentiment=agg_sentiment,
                stocktwits_trending=stocktwits_data,
                technical_data=ta_data,
                congressional_buys=congress_buys,
                market_breadth=market_breadth,
                put_call_ratio=put_call_ratio,
                insider_buys=insider_buys,
                current_positions=current_positions,    # what the bot holds now
                scan_top=last_scan_top,                  # what was scanned last cycle
                live_market_context=live_market_context, # day type, timing, performance stats
            ),
            default=None,
            label="ClaudeAI",
        )
    else:
        logger.warning("ANTHROPIC_API_KEY not set — skipping AI analysis")

    # ── Fallback to cached analysis ───────────────────────────────
    if not analysis:
        cached = history.get("last_analysis")
        if cached:
            logger.info("Using cached last_analysis from history.json")
            analysis = cached
        else:
            logger.warning("No analysis and no cache — writing minimal fallback")
            analysis = {
                "market_sentiment": "UNKNOWN",
                "market_regime": "UNCERTAIN",
                "week_summary": "Intelligence data loading...",
                "buy_signals": [],
                "sell_signals": [],
                "watch_list": [],
                "notable_trends": [],
                "macro_risks": [],
                "rok_message": "Connecting to AI analysis — check back shortly.",
            }
    else:
        history["last_analysis"] = analysis

    # ── Build price lookup from stock_data ────────────────────────
    price_lookup = {s["ticker"]: s["price"] for s in stock_data if s}
    stock_data_lookup = {s["ticker"]: s for s in stock_data if s}

    # ── Enrich signals ────────────────────────────────────────────
    signal_lookup = {}
    for sig in analysis.get("buy_signals", []):
        signal_lookup[sig["ticker"]] = {"type": "buy", "strength": sig.get("signal_strength", 5)}
    for sig in analysis.get("sell_signals", []):
        signal_lookup[sig["ticker"]] = {"type": "sell", "strength": sig.get("signal_strength", 5)}

    all_signals = (
        analysis.get("buy_signals", [])
        + analysis.get("sell_signals", [])
        + analysis.get("watch_list", [])
    )
    for sig in all_signals:
        t = sig.get("ticker", "")
        if not t:
            continue
        sd = stock_data_lookup.get(t)
        if sd:
            if not sig.get("current_price") and sd.get("price"):
                sig["current_price"] = sd["price"]
            if not sig.get("company") and sd.get("company_name"):
                sig["company"] = sd["company_name"]
            if not sig.get("price_target") and sd.get("analyst_target"):
                sig["price_target"] = sd["analyst_target"]
            if not sig.get("stop_loss") and sig.get("current_price"):
                sig["stop_loss"] = round(sig["current_price"] * 0.92, 2)
            if not sig.get("sector"):
                sig["sector"] = sd.get("sector", "")
            ta = ta_data.get(t, {}) if ta_data else {}
            if not sig.get("vol_ratio"):
                sig["vol_ratio"] = ta.get("volume_ratio") or sd.get("vol_ratio")
            if not sig.get("rsi"):
                sig["rsi"] = ta.get("rsi") or sd.get("rsi")

        # Price sparkline (last 30 days)
        sig["price_history"] = _safe(
            yahoo_finance.get_price_history, t, 30,
            default=list, label=f"Hist:{t}",
        ) or []

        if not sig.get("signal_strength"):
            sig["signal_strength"] = 6
        if not sig.get("time_horizon"):
            sig["time_horizon"] = "1-3 months"
        if not sig.get("risk_level"):
            sig["risk_level"] = "Medium"

    # ── History tracking ──────────────────────────────────────────
    history["runs"].append({
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "timestamp": datetime.utcnow().isoformat(),
        "sentiment": analysis.get("market_sentiment"),
        "regime": analysis.get("market_regime"),
        "buy_signals": [
            {"ticker": s["ticker"], "price": s.get("current_price"), "target": s.get("price_target")}
            for s in analysis.get("buy_signals", [])
        ],
        "sell_signals": [
            {"ticker": s["ticker"], "price": s.get("current_price")}
            for s in analysis.get("sell_signals", [])
        ],
    })
    history["runs"] = history["runs"][-96:]

    # ── Track record ──────────────────────────────────────────────
    track_record = []
    if len(history["runs"]) >= 2:
        prev_run = history["runs"][-2]
        for sig in prev_run.get("buy_signals", []):
            ticker = sig.get("ticker")
            entry = sig.get("price")
            if not ticker or not entry:
                continue
            current = price_lookup.get(ticker)
            if not current:
                sd2 = _safe(yahoo_finance.get_stock_data, ticker, default=lambda: None)
                current = sd2["price"] if sd2 else None
            if current and entry:
                pct = round((current - entry) / entry * 100, 1)
                track_record.append({
                    "ticker": ticker,
                    "entry_price": entry,
                    "current_price": round(current, 2),
                    "pct_change": pct,
                    "date": prev_run.get("date", ""),
                })
        track_record.sort(key=lambda x: abs(x["pct_change"]), reverse=True)

    # ── Market mood plain-language ────────────────────────────────
    mkt_sent = (analysis.get("market_sentiment") or "NEUTRAL").upper()
    fg_score = (fear_greed or {}).get("score", 50)
    buy_count = len(analysis.get("buy_signals", []))
    if mkt_sent == "BULLISH" and buy_count >= 5:
        market_mood = f"Markets are strong — ROK found {buy_count} stocks worth watching right now"
    elif mkt_sent == "BULLISH":
        market_mood = "Markets are leaning bullish — ROK sees some opportunities"
    elif mkt_sent == "BEARISH":
        market_mood = "Markets are under pressure — ROK recommends caution"
    elif fg_score and fg_score < 30:
        market_mood = "Fear is high — that often means buying opportunities are near"
    else:
        market_mood = "Markets are mixed — ROK is watching closely for clear signals"

    # ── Build intel_report output ─────────────────────────────────
    intel = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "market_regime": analysis.get("market_regime", "UNCERTAIN"),
        "market_sentiment": analysis.get("market_sentiment", "NEUTRAL"),
        "sentiment_score": analysis.get("sentiment_score", 5),
        "week_summary": analysis.get("week_summary", ""),
        "market_mood": market_mood,
        "rok_message": analysis.get("rok_message", ""),
        "buy_signals": analysis.get("buy_signals", []),
        "sell_signals": analysis.get("sell_signals", []),
        "watch_list": analysis.get("watch_list", []),
        "notable_trends": analysis.get("notable_trends", []),
        "macro_risks": analysis.get("macro_risks", []),
        "sector_heat": analysis.get("sector_heat", {}),
        "sector_rotation": analysis.get("sector_rotation", ""),
        "short_squeeze_alerts": analysis.get("short_squeeze_alerts", []),
        "earnings_plays": analysis.get("earnings_plays", []),
        "congressional_plays": analysis.get("congressional_plays", []),
        "technical_breakouts": analysis.get("technical_breakouts", []),
        "fear_greed": fear_greed or {},
        "market_indices": market_indices or {},
        "market_breadth": market_breadth or {},
        "put_call_ratio": put_call_ratio or {},
        "ticker_mentions": top_tickers[:24],
        "stocktwits_trending": stocktwits_data[:12],
        "congressional_buys": congress_buys[:8],
        "insider_buys": insider_buys[:12],
        "track_record": track_record[:10],
        "recent_runs": history["runs"][-8:],
        "news_items": [
            {
                "title": a.get("title", ""),
                "source": a.get("source", ""),
                "url": a.get("url", ""),
                "sentiment": a.get("sentiment_score", 0),
                "tickers": a.get("mentioned_tickers", []),
            }
            for a in news_articles[:30]
            if a.get("title")
        ],
        "source_stats": {
            "reddit": len(reddit_posts),
            "news": len(news_articles),
            "stocks": len(stock_data),
            "sec": len(sec_filings or []),
            "earnings_upcoming": len(earnings_cal or []),
            "unusual_options": len(unusual_opts or []),
            "congress_trades": len(congress_buys or []),
            "technical": len(ta_data or {}),
            "insider_buys": len(insider_buys or []),
        },
        "buy_count": buy_count,
        "sell_count": len(analysis.get("sell_signals", [])),
        "current_positions": current_positions[:10],  # pass-through for dashboard cross-reference
        "position_analysis": analysis.get("position_analysis", []),  # AI commentary on held positions
        "rs_leaders":      live_market_context.get("rs_leaders", []),
        "ema21_setups":    live_market_context.get("ema21_setups", []),
        "pocket_pivots":   live_market_context.get("pocket_pivots", []),
        "htf_stocks":      live_market_context.get("htf_stocks", []),
        "tt8_stocks":      live_market_context.get("tt8_stocks", []),
        "tt_leaders":      live_market_context.get("tt_leaders", []),
        "drawdown_halt":   live_market_context.get("drawdown_halt", False),
        "regime_max_pos":  live_market_context.get("regime_max_pos", 12),
        "scan_breadth_pct": live_market_context.get("scan_breadth_pct"),
        # Full live context for dashboard (weekend watchlist, bot state, etc.)
        "bot_state": {
            "strategy_mode":   live_market_context.get("strategy_mode", ""),
            "bot_conviction":  live_market_context.get("bot_conviction", 0),
            "neurons_active":  live_market_context.get("neurons_active", 0),
            "neurons_total":   live_market_context.get("neurons_total", 630),
            "market_open":     live_market_context.get("market_open", False),
            "win_rate":        live_market_context.get("win_rate", 0),
            "drawdown_pct":    live_market_context.get("drawdown_pct", 0),
            "intraday_wins":   live_market_context.get("intraday_wins", 0),
            "intraday_losses": live_market_context.get("intraday_losses", 0),
            "loss_streak":     live_market_context.get("loss_streak", 0),
            "daily_risk_mult": live_market_context.get("daily_risk_mult", 1.0),
        },
        "weekend_watchlist": live_market_context.get("weekend_watchlist", []),
        "market_open_plan":   live_market_context.get("market_open_plan", {}),
        "exit_intelligence":  live_market_context.get("exit_intelligence", {}),
        "next_market_open":   live_market_context.get("next_market_open", ""),
        "position_news":      live_market_context.get("position_news", {}),
        # Synaptic learning intelligence
        "top_synapses":       live_market_context.get("top_synapses", []),
        "top_triplets":       live_market_context.get("top_triplets", []),
        "sector_performance": live_market_context.get("sector_performance", {}),
        "next_entry_conditions": live_market_context.get("next_entry_conditions", {}),
        "smart_alerts":       live_market_context.get("smart_alerts", []),
        "last_scan_top":      live_market_context.get("last_scan_top", []),
        "top_neurons":        live_market_context.get("top_neurons", []),
    }

    # Sanitize all datetime objects before JSON serialization
    intel = _sanitize(intel)

    # ── Write output files ────────────────────────────────────────
    intel_json = json.dumps(intel, cls=_Encoder, indent=2)
    (docs_dir / "intel_report.json").write_text(intel_json, encoding="utf-8")
    logger.info(f"Intel report written → docs/intel_report.json ({len(intel_json)} chars)")

    # Update history
    history_path.write_text(json.dumps(_sanitize(history), cls=_Encoder, indent=2), encoding="utf-8")
    logger.info(f"History updated → docs/history.json ({len(history['runs'])} runs)")

    # Write prices.json for JS fallback — ALWAYS include index ETFs + VIX
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # Start with existing prices.json to preserve any bot-written data (positions, etc.)
    try:
        _existing_prices = json.loads((docs_dir / "prices.json").read_text()) if (docs_dir / "prices.json").exists() else {}
    except Exception:
        _existing_prices = {}
    prices_dict = dict(_existing_prices)  # preserve existing (bot positions)
    # Overwrite with fresh signal data
    for s in stock_data:
        t = s.get("ticker")
        if t and t:
            prices_dict[t] = {
                "price": s.get("price"),
                "change_pct": s.get("change_pct"),
                "updated": now_iso,
            }
    # Always fetch and include index ETFs + VIX (most critical for dashboard)
    try:
        import yfinance as _yf_rpt
        _idx_tickers = ["SPY", "QQQ", "DIA", "IWM"]
        _idx_df = _yf_rpt.download(
            _idx_tickers, period="5d", interval="1d",
            auto_adjust=True, progress=False, group_by="ticker", threads=False
        )
        for _tk in _idx_tickers:
            try:
                if hasattr(_idx_df.columns, "levels"):
                    _closes = _idx_df[_tk]["Close"].dropna()
                else:
                    _closes = _idx_df["Close"].dropna()
                if len(_closes) >= 1:
                    _px   = float(_closes.iloc[-1])
                    _prev = float(_closes.iloc[-2]) if len(_closes) >= 2 else _px
                    _chg  = round((_px - _prev) / _prev * 100, 2) if _prev else 0
                    prices_dict[_tk] = {"price": round(_px, 2), "change_pct": _chg, "updated": now_iso}
            except Exception:
                pass
        # VIX
        _vix_data = _yf_rpt.download("^VIX", period="5d", interval="1d",
                                      auto_adjust=True, progress=False, threads=False)
        if not _vix_data.empty:
            _vix_cls = _vix_data["Close"].dropna()
            if len(_vix_cls) >= 1:
                _vx = float(_vix_cls.iloc[-1])
                _vp = float(_vix_cls.iloc[-2]) if len(_vix_cls) >= 2 else _vx
                _vc = round((_vx - _vp) / _vp * 100, 2) if _vp else 0
                prices_dict["^VIX"] = {"price": round(_vx, 2), "change_pct": _vc, "updated": now_iso}
                prices_dict["VIX"]  = prices_dict["^VIX"]
    except Exception as _ep:
        logger.debug(f"Index ETF fetch in report: {_ep}")
    _idx_found = [k for k in ("SPY","QQQ","DIA","IWM","^VIX") if k in prices_dict]
    (docs_dir / "prices.json").write_text(json.dumps(prices_dict, cls=_Encoder), encoding="utf-8")
    logger.info(f"Prices written → docs/prices.json ({len(prices_dict)} tickers, indices={_idx_found})")

    # ── Weekly Bot Performance Report ────────────────────────────────
    try:
        _week_report = _build_weekly_bot_report(docs_dir)
        if _week_report:
            (docs_dir / "weekly_report.json").write_text(
                json.dumps(_sanitize(_week_report), cls=_Encoder, indent=2), encoding="utf-8"
            )
            logger.info(f"Weekly report written → docs/weekly_report.json")
    except Exception as _we:
        logger.warning(f"Weekly report failed: {_we}")

    logger.info(f"Summary: {buy_count} buys | {len(analysis.get('sell_signals', []))} sells")
    logger.info("ROK INTELLIGENCE PIPELINE COMPLETE")


if __name__ == "__main__":
    run()
