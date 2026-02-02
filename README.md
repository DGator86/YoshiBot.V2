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

## Files

- `kalshi_bitcoin_oracle.py` – main oracle implementation.
- `.env.example` – template for required environment variables.
- `requirements.txt` – Python dependencies.
