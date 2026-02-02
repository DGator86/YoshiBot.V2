#!/usr/bin/env python3
"""
Kalshi Bitcoin Hourly Prediction Oracle
Complete .env integration with physics-inspired modeling.
"""
import asyncio
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import ccxt
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from telegram import Bot

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()


@dataclass
class Config:
    """Configuration loaded from .env file."""

    # Telegram
    TELEGRAM_BOT_TOKEN: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")

    # Kalshi
    KALSHI_EMAIL: Optional[str] = os.getenv("KALSHI_EMAIL")
    KALSHI_PASSWORD: Optional[str] = os.getenv("KALSHI_PASSWORD")

    # Exchange
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")

    # System
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    SIMULATION_COUNT: int = int(os.getenv("SIMULATION_COUNT", "50000"))
    UPDATE_INTERVAL: int = int(os.getenv("UPDATE_INTERVAL_MINUTES", "5"))
    PREDICTION_HORIZON: int = int(os.getenv("PREDICTION_HORIZON_MINUTES", "60"))


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=getattr(logging, Config.LOG_LEVEL, logging.INFO),
    handlers=[logging.FileHandler("kalshi_oracle.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class BitcoinPhysicsOracle:
    """Physics-inspired Bitcoin price prediction using Monte Carlo simulation."""

    def __init__(self) -> None:
        exchange_config: Dict[str, Any] = {
            "enableRateLimit": True,
            "timeout": 30000,
            "options": {"defaultType": "spot"},
        }
        if Config.BINANCE_API_KEY and Config.BINANCE_API_SECRET:
            exchange_config.update(
                {"apiKey": Config.BINANCE_API_KEY, "secret": Config.BINANCE_API_SECRET}
            )
            logger.info("Using authenticated Binance API")
        else:
            logger.info("Using public Binance API")

        try:
            self.exchange = ccxt.binance(exchange_config)
        except Exception as exc:
            logger.error("Exchange initialization failed: %s", exc)
            raise

        self.force_params = {
            "funding_strength": 15.0,
            "imbalance_strength": 0.08,
            "momentum_decay": 0.4,
            "volatility_base": 0.015,
        }
        logger.info("Oracle initialized with %s simulations", f"{Config.SIMULATION_COUNT:,}")

    def fetch_market_data(self) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]], Optional[float]]:
        """Fetch comprehensive market data with retry logic."""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info("Fetching market data (attempt %s)...", attempt + 1)
                ohlcv = self.exchange.fetch_ohlcv("BTC/USDT", "1m", limit=1440)
                df = pd.DataFrame(
                    ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df["returns"] = df["close"].pct_change()

                orderbook = self.exchange.fetch_order_book("BTC/USDT", limit=100)

                funding_rate = 0.0
                try:
                    funding_data = self.exchange.fetch_funding_rate("BTC/USDT")
                    funding_rate = float(funding_data.get("fundingRate", 0.0))
                except Exception as exc:
                    logger.warning("Could not fetch funding rate: %s", exc)

                logger.info("Market data fetched successfully")
                return df, orderbook, funding_rate
            except Exception as exc:
                logger.warning("Attempt %s failed: %s", attempt + 1, exc)
                if attempt == max_retries - 1:
                    logger.error("All data fetch attempts failed")
                    return None, None, None
                time.sleep(5)
        return None, None, None

    def calculate_market_forces(
        self, df: pd.DataFrame, orderbook: Dict[str, Any], funding_rate: float
    ) -> Dict[str, Any]:
        """Calculate market forces that steer Bitcoin price movement."""

        funding_force = -funding_rate * self.force_params["funding_strength"]

        bids = orderbook.get("bids", [])[:50]
        asks = orderbook.get("asks", [])[:50]
        bid_volume = sum(level[1] for level in bids) if bids else 0
        ask_volume = sum(level[1] for level in asks) if asks else 0
        if bid_volume + ask_volume > 0:
            ob_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        else:
            ob_imbalance = 0.0
        imbalance_force = ob_imbalance * self.force_params["imbalance_strength"]

        recent_returns = df["returns"].tail(30).dropna()
        momentum = recent_returns.mean() if len(recent_returns) > 0 else 0.0
        momentum_force = momentum * self.force_params["momentum_decay"]

        vol_window = df["returns"].tail(60).dropna()
        if len(vol_window) > 10:
            realized_vol = float(vol_window.std() * np.sqrt(60))
        else:
            realized_vol = self.force_params["volatility_base"]

        vol_regime_factor = realized_vol / self.force_params["volatility_base"]
        final_volatility = max(
            realized_vol * (1 + vol_regime_factor * 0.3),
            self.force_params["volatility_base"],
        )

        total_drift = funding_force + imbalance_force + momentum_force

        return {
            "drift": total_drift,
            "volatility": final_volatility,
            "components": {
                "funding_force": funding_force,
                "imbalance_force": imbalance_force,
                "momentum_force": momentum_force,
                "ob_imbalance": ob_imbalance,
                "realized_vol": realized_vol,
                "funding_rate": funding_rate,
            },
        }

    def monte_carlo_simulation(
        self, current_price: float, drift: float, volatility: float, minutes_ahead: int
    ) -> np.ndarray:
        """Vectorized Monte Carlo simulation using geometric Brownian motion."""

        dt = 1.0 / 60.0
        n_steps = minutes_ahead
        n_sims = Config.SIMULATION_COUNT

        random_shocks = np.random.normal(0, 1, (n_sims, n_steps))
        log_returns = (
            (drift - 0.5 * volatility**2) * dt
            + volatility * np.sqrt(dt) * random_shocks
        )
        cumulative_returns = np.cumsum(log_returns, axis=1)
        final_prices = current_price * np.exp(cumulative_returns[:, -1])
        return final_prices

    def predict_hourly_distribution(self, target_time: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """Generate complete price distribution for target time."""

        now = datetime.now()
        if target_time is None:
            minutes_ahead = Config.PREDICTION_HORIZON - (now.minute % Config.PREDICTION_HORIZON)
            target_time = now + timedelta(minutes=minutes_ahead)
        else:
            minutes_ahead = int((target_time - now).total_seconds() / 60)

        df, orderbook, funding_rate = self.fetch_market_data()
        if df is None or orderbook is None:
            logger.error("Failed to fetch market data")
            return None

        current_price = float(df["close"].iloc[-1])
        field_analysis = self.calculate_market_forces(df, orderbook, funding_rate or 0.0)

        logger.info("Running %s Monte Carlo simulations...", f"{Config.SIMULATION_COUNT:,}")
        final_prices = self.monte_carlo_simulation(
            current_price, field_analysis["drift"], field_analysis["volatility"], minutes_ahead
        )

        base_strike = round(current_price / 250) * 250
        strikes = [base_strike + (i * 250) for i in range(-6, 7)]
        strike_probabilities = {strike: float(np.mean(final_prices >= strike)) for strike in strikes}

        return {
            "current_price": current_price,
            "target_time": target_time,
            "minutes_ahead": minutes_ahead,
            "median_prediction": float(np.median(final_prices)),
            "mean_prediction": float(np.mean(final_prices)),
            "strike_probabilities": strike_probabilities,
            "field_analysis": field_analysis,
            "confidence_intervals": {
                "50%": (
                    float(np.percentile(final_prices, 25)),
                    float(np.percentile(final_prices, 75)),
                ),
                "68%": (
                    float(np.percentile(final_prices, 16)),
                    float(np.percentile(final_prices, 84)),
                ),
                "90%": (
                    float(np.percentile(final_prices, 5)),
                    float(np.percentile(final_prices, 95)),
                ),
                "95%": (
                    float(np.percentile(final_prices, 2.5)),
                    float(np.percentile(final_prices, 97.5)),
                ),
            },
        }


class KalshiMarketAnalyzer:
    """Analyzes Kalshi markets for comparison with model predictions."""

    def __init__(self) -> None:
        self.base_url = "https://trading-api.kalshi.com/trade-api/v2"
        self.demo_url = "https://demo-api.kalshi.co/trade-api/v2"
        self.session = requests.Session()
        self.authenticated = False

        self.use_demo = not (Config.KALSHI_EMAIL and Config.KALSHI_PASSWORD)
        if self.use_demo:
            logger.info("Using Kalshi demo API (read-only)")

    def authenticate(self) -> bool:
        """Authenticate with Kalshi API."""

        if self.use_demo:
            return True

        try:
            response = self.session.post(
                f"{self.base_url}/login",
                json={"email": Config.KALSHI_EMAIL, "password": Config.KALSHI_PASSWORD},
                timeout=15,
            )
            if response.status_code == 200:
                data = response.json()
                self.session.headers.update({"Authorization": f"Bearer {data['token']}"})
                self.authenticated = True
                logger.info("Kalshi authentication successful")
                return True

            logger.error("Kalshi authentication failed: %s", response.text)
            return False
        except Exception as exc:
            logger.error("Kalshi authentication error: %s", exc)
            return False

    def get_bitcoin_markets(self) -> List[Dict[str, Any]]:
        """Fetch current Bitcoin markets from Kalshi."""

        api_url = self.demo_url if self.use_demo else self.base_url
        try:
            response = self.session.get(
                f"{api_url}/markets", params={"limit": 100, "status": "open"}, timeout=15
            )
            if response.status_code == 200:
                markets = response.json().get("markets", [])
                btc_markets = [
                    market
                    for market in markets
                    if any(keyword in market.get("ticker", "").upper() for keyword in ["BTC", "BITCOIN"])
                ]
                logger.info("Found %s Bitcoin markets on Kalshi", len(btc_markets))
                return btc_markets

            logger.warning("Could not fetch Kalshi markets: %s", response.text)
            return []
        except Exception as exc:
            logger.error("Error fetching Kalshi markets: %s", exc)
            return []


class TelegramNotifier:
    """Handles Telegram notifications with rich formatting."""

    def __init__(self) -> None:
        self.bot_token = Config.TELEGRAM_BOT_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self.enabled = bool(self.bot_token and self.chat_id)
        if not self.enabled:
            logger.warning("Telegram notifications disabled - credentials not provided")

    async def send_prediction_alert(self, prediction_data: Dict[str, Any]) -> None:
        """Send formatted prediction alert to Telegram."""

        if not self.enabled:
            return

        try:
            bot = Bot(token=self.bot_token)

            current_price = prediction_data["current_price"]
            target_time = prediction_data["target_time"]
            median_pred = prediction_data["median_prediction"]
            strikes = prediction_data["strike_probabilities"]
            field_analysis = prediction_data["field_analysis"]

            message = "🔮 Bitcoin Kalshi Oracle\n\n"
            message += f"⏰ Target: {target_time.strftime('%I:%M %p EST')}\n"
            message += f"💰 Current: ${current_price:,.0f}\n"
            message += f"📊 Predicted: ${median_pred:,.0f}\n"
            message += f"📈 Expected Move: {((median_pred/current_price - 1)*100):+.2f}%\n\n"

            components = field_analysis["components"]
            message += "🧲 Market Forces:\n"
            message += f"• Funding: {components['funding_force']*100:+.3f}%\n"
            message += f"• Order Book: {components['ob_imbalance']:+.3f}\n"
            message += f"• Momentum: {components['momentum_force']*100:+.3f}%\n"
            message += f"• Volatility: {field_analysis['volatility']*100:.2f}%\n\n"

            message += "🎯 Strike Probabilities:\n"
            relevant_strikes = {
                strike: prob for strike, prob in strikes.items() if 0.05 <= prob <= 0.95
            }
            for strike, prob in sorted(relevant_strikes.items()):
                prob_pct = prob * 100
                if prob_pct > 75:
                    icon = "🟢"
                elif prob_pct > 60:
                    icon = "🟡"
                elif prob_pct > 40:
                    icon = "⚪"
                elif prob_pct > 25:
                    icon = "🟠"
                else:
                    icon = "🔴"
                message += f"{icon} ≥ ${strike:,}: {prob_pct:.1f}%\n"

            ci_95 = prediction_data["confidence_intervals"]["95%"]
            message += f"\n📊 95% CI: ${ci_95[0]:,.0f} - ${ci_95[1]:,.0f}\n"
            message += "\n_Compare with Kalshi markets for trading edges_"

            await bot.send_message(chat_id=self.chat_id, text=message, parse_mode="Markdown")
            logger.info("Telegram alert sent successfully")
        except Exception as exc:
            logger.error("Failed to send Telegram alert: %s", exc)


class KalshiBitcoinOracle:
    """Main orchestrator class that coordinates all components."""

    def __init__(self) -> None:
        logger.info("Initializing Kalshi Bitcoin Oracle...")
        self.oracle = BitcoinPhysicsOracle()
        self.kalshi = KalshiMarketAnalyzer()
        self.notifier = TelegramNotifier()
        self.kalshi.authenticate()
        logger.info("Oracle initialization complete")

    def run_analysis_cycle(self) -> bool:
        """Run complete analysis cycle."""

        try:
            logger.info("=" * 80)
            logger.info("Starting analysis cycle...")

            prediction = self.oracle.predict_hourly_distribution()
            if prediction is None:
                logger.error("Failed to generate prediction")
                return False

            self._display_prediction(prediction)
            if self.notifier.enabled:
                asyncio.run(self.notifier.send_prediction_alert(prediction))

            kalshi_markets = self.kalshi.get_bitcoin_markets()
            if kalshi_markets:
                self._analyze_kalshi_opportunities(prediction, kalshi_markets)

            logger.info("Analysis cycle completed successfully")
            return True
        except Exception as exc:
            logger.error("Analysis cycle failed: %s", exc)
            return False

    def _display_prediction(self, prediction: Dict[str, Any]) -> None:
        """Display formatted prediction to console."""

        current_price = prediction["current_price"]
        target_time = prediction["target_time"]
        median_pred = prediction["median_prediction"]
        strikes = prediction["strike_probabilities"]
        field = prediction["field_analysis"]

        print(f"\n{'='*80}")
        print(f"🔮 KALSHI BITCOIN ORACLE - {target_time.strftime('%I:%M %p EST')}")
        print(f"{'='*80}")
        print(f"💰 Current Price: ${current_price:,.2f}")
        print(f"📊 Predicted Price: ${median_pred:,.2f}")
        print(f"📈 Expected Return: {((median_pred/current_price - 1)*100):+.2f}%")

        print("\n🧲 MARKET FORCES:")
        comp = field["components"]
        print(
            " • Funding Rate: "
            f"{comp['funding_rate']*100:+.4f}% → Force: {comp['funding_force']*100:+.3f}%"
        )
        print(
            " • Order Book: "
            f"{comp['ob_imbalance']:+.3f} → Force: {comp['imbalance_force']*100:+.3f}%"
        )
        print(f" • Momentum: {comp['momentum_force']*100:+.3f}%")
        print(f" • Volatility: {field['volatility']*100:.2f}%")

        print("\n🎯 STRIKE PROBABILITIES:")
        print("-" * 70)
        print(f"{'Strike':<15} {'Prob':<10} Confidence")
        print("-" * 70)

        for strike, prob in sorted(strikes.items()):
            prob_pct = prob * 100
            if prob_pct > 80:
                confidence = "🟢 Very High"
            elif prob_pct > 65:
                confidence = "🟡 High"
            elif prob_pct > 50:
                confidence = "⚪ Medium"
            elif prob_pct > 35:
                confidence = "🟠 Low"
            else:
                confidence = "🔴 Very Low"

            if 5 <= prob_pct <= 95:
                print(f"${strike:<14,} {prob_pct:>6.1f}% {confidence}")

        ci_95 = prediction["confidence_intervals"]["95%"]
        print("-" * 70)
        print(f"95% CI: ${ci_95[0]:,.0f} - ${ci_95[1]:,.0f}\n")

    def _analyze_kalshi_opportunities(
        self, prediction: Dict[str, Any], kalshi_markets: List[Dict[str, Any]]
    ) -> None:
        """Basic comparison between model probabilities and Kalshi markets."""

        logger.info("Analyzing %s Kalshi markets for model edges", len(kalshi_markets))
        strike_probs = prediction["strike_probabilities"]
        for market in kalshi_markets:
            ticker = market.get("ticker", "")
            strike = market.get("strike_price")
            if strike is None:
                continue
            model_prob = strike_probs.get(float(strike))
            if model_prob is None:
                continue
            logger.info("Market %s strike %s -> model prob %.2f%%", ticker, strike, model_prob * 100)

    def run_continuous(self) -> None:
        """Run oracle in continuous mode."""

        interval = Config.UPDATE_INTERVAL
        logger.info("Starting continuous mode (every %s minutes)...", interval)
        while True:
            try:
                success = self.run_analysis_cycle()
                if success:
                    logger.info("Sleeping for %s minutes...", interval)
                    time.sleep(interval * 60)
                else:
                    logger.warning("Analysis failed, shorter sleep...")
                    time.sleep(2 * 60)
            except KeyboardInterrupt:
                logger.info("Oracle stopped by user")
                break
            except Exception as exc:
                logger.error("Unexpected error in continuous mode: %s", exc)
                time.sleep(60)


def check_configuration() -> None:
    """Check and display configuration status."""

    print("\n🔧 CONFIGURATION CHECK:")
    print("=" * 60)
    if Config.TELEGRAM_BOT_TOKEN and Config.TELEGRAM_CHAT_ID:
        print("✅ Telegram notifications: ENABLED")
    else:
        print("⚠️ Telegram notifications: DISABLED")

    if Config.KALSHI_EMAIL and Config.KALSHI_PASSWORD:
        print("✅ Kalshi API: LIVE TRADING ENABLED")
    else:
        print("⚠️ Kalshi API: DEMO MODE ONLY")

    if Config.BINANCE_API_KEY and Config.BINANCE_API_SECRET:
        print("✅ Binance API: AUTHENTICATED (higher limits)")
    else:
        print("ℹ️ Binance API: PUBLIC (rate limited)")

    print(f"ℹ️ Monte Carlo simulations: {Config.SIMULATION_COUNT:,}")
    print(f"ℹ️ Update interval: {Config.UPDATE_INTERVAL} minutes")
    print(f"ℹ️ Prediction horizon: {Config.PREDICTION_HORIZON} minutes")
    print("=" * 60)


def main() -> None:
    """Main entry point."""

    print("🚀 Kalshi Bitcoin Oracle Starting...")
    if not os.path.exists(".env"):
        print("\n❌ ERROR: .env file not found!")
        print("Please create a .env file with your configuration.")
        print("Use the template provided in the documentation.\n")
        sys.exit(1)

    check_configuration()

    try:
        oracle = KalshiBitcoinOracle()
        if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
            oracle.run_continuous()
        else:
            oracle.run_analysis_cycle()
    except KeyboardInterrupt:
        print("\n👋 Stopped by user")
    except Exception as exc:
        logger.error("Fatal error: %s", exc)


if __name__ == "__main__":
    main()
