# Kalshi Bitcoin Oracle

A Bitcoin price prediction oracle that uses a physics-inspired Monte Carlo model, `.env`-based secrets, and optional Telegram alerts.

## Quick Start

1. **Create your `.env` file** (copy the template and fill in your new credentials):

   ```bash
   cp .env.example .env
   chmod 600 .env
   ```

2. **Install dependencies**:

   ```bash
   pip3 install -r requirements.txt
   ```

3. **Run a single prediction**:

   ```bash
   python3 kalshi_bitcoin_oracle.py
   ```

4. **Run continuous mode (every 5 minutes by default)**:

   ```bash
   python3 kalshi_bitcoin_oracle.py --continuous
   ```

## Security Notes

- **Never commit `.env`**. It is ignored by default in `.gitignore`.
- Rotate any exposed credentials immediately.

## Configuration

All configuration is loaded from `.env`. Use `.env.example` as the template.

## Kalshi API setup (overview)

Kalshi’s API onboarding and authentication requirements are documented here:
https://docs.kalshi.com/welcome

High-level flow:

1. Create a Kalshi account and complete any required verification.
2. Generate API credentials in the Kalshi dashboard.
3. Store credentials in `.env` (never commit them).
4. Run the oracle; without credentials it will use the demo endpoints.

> Note: This implementation currently authenticates using `KALSHI_EMAIL` and
> `KALSHI_PASSWORD` for the live API and falls back to demo mode when they are
> missing. If your Kalshi account requires key-based auth, follow the docs and
> update the authentication flow accordingly.

## Files

- `kalshi_bitcoin_oracle.py` – main oracle implementation.
- `.env.example` – template for required environment variables.
- `requirements.txt` – Python dependencies.
