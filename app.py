import os
import logging
import asyncio
import aiohttp
import json
import time
from polygon import RESTClient
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import yfinance as yf
import pytz  # Added for timezone handling

# Configuration Variables
MIN_PRICE = 10.0  # Minimum stock price
VOLUME_THRESHOLD = 2_000_000  # Minimum average daily volume
TIMEFRAMES = ['4hour', 'day', 'week', 'month', 'quarter']  # Timeframes including 4hour
BATCH_SIZE = 700  # Number of tickers to process concurrently
CACHE_FILE = "filtered_inside.json"
CACHE_EXPIRY = 86400  # Cache expiry in seconds (24 hours)
FLOAT_VOL_LOOKBACK = 10  # Number of periods to calculate float traded
FLOAT_TRADED_THRESHOLD = 0.03  # Minimum % float traded (3%)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('polygon_script.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize Polygon client
api_key = os.getenv("POLYGON_API_KEY") or "beu8u8JlVDmMRGyjc88EVei1rKQDxBPo"
client = RESTClient(api_key)

def color_bool(value):
    return f"\033[92m{value}\033[0m" if value else f"\033[91m{value}\033[0m"

async def fetch_stock_data(session, ticker, timeframe, start_date, end_date):
    if timeframe == '4hour':
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/4/hour/{start_date}/{end_date}"
    else:
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{timeframe}/{start_date}/{end_date}"
    params = {"apiKey": api_key, "limit": 5000}
    try:
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                if data.get('results'):
                    df = pd.DataFrame([{
                        'open': bar['o'],
                        'high': bar['h'],
                        'low': bar['l'],
                        'close': bar['c'],
                        'volume': bar['v'],
                        'timestamp': pd.to_datetime(bar['t'], unit='ms')
                    } for bar in data['results']])
                    return ticker, df
            logger.warning(f"No data for {ticker} on {timeframe}")
            return ticker, None
    except Exception as e:
        logger.error(f"Error fetching data for {ticker} on {timeframe}: {e}")
        return ticker, None

def is_inside_bar(current, previous):
    return (current['high'] <= previous['high'] and current['low'] >= previous['low'])

def is_engulfing_bar(current, previous):
    current_range = current['high'] - current['low']
    previous_range = previous['high'] - previous['low']
    return (current_range > previous_range and 
            ((current['close'] > current['open'] and current['high'] >= previous['high'] and 
              current['low'] <= previous['low'] and current['close'] > previous['open']) or 
             (current['close'] < current['open'] and current['high'] >= previous['high'] and 
              current['low'] <= previous['low'] and current['close'] < previous['open'])))

async def analyze_ticker(session, ticker, timeframe):
    end_date = datetime.now().strftime('%Y-%m-%d')
    if timeframe == '4hour':
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    else:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    ticker, df = await fetch_stock_data(session, ticker, timeframe, start_date, end_date)
    if df is None or len(df) < FLOAT_VOL_LOOKBACK:
        return None

    latest_price = df['close'].iloc[-1]
    if latest_price < MIN_PRICE:
        return None

    current_bar = df.iloc[-1]
    previous_bar = df.iloc[-2]

    try:
        fundamentals = client.get_ticker_details(ticker)
        shares_float = fundamentals.share_class_shares_outstanding
    except Exception as e:
        logger.warning(f"Couldn't fetch float for {ticker}: {e}")
        return None

    if not shares_float or shares_float == 0:
        return None

    float_traded_series = df['volume'].tail(FLOAT_VOL_LOOKBACK) / shares_float
    avg_float_traded = float_traded_series.mean()

    if avg_float_traded < FLOAT_TRADED_THRESHOLD:
        return None

    result = {
        'ticker': ticker,
        'timeframe': timeframe,
        'inside_bar': is_inside_bar(current_bar, previous_bar),
        'engulfing_bar': is_engulfing_bar(current_bar, previous_bar),
        'avg_float_traded': avg_float_traded
        # Removed earnings date fetching from here
    }
    return result

async def get_filtered_tickers(rest_client):
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            cache_data = json.load(f)
        if time.time() - cache_data['timestamp'] < CACHE_EXPIRY:
            logger.info("Using cached filtered tickers.")
            return cache_data['tickers']

    tickers = []
    all_tickers = [ticker.ticker for ticker in rest_client.list_tickers(market='stocks', type='CS', active=True, limit=1000)]
    logger.info(f"Retrieved {len(all_tickers)} active U.S. stock tickers.")

    async with aiohttp.ClientSession() as session:
        for i in range(0, len(all_tickers), BATCH_SIZE):
            batch = all_tickers[i:i + BATCH_SIZE]
            ticker_string = ",".join(batch)
            try:
                snapshot = rest_client.get_snapshot_all(market_type="stocks", tickers=ticker_string)
                price_filtered = []
                for snap in snapshot:
                    ticker = snap.ticker
                    last_price = snap.last_trade.price if snap.last_trade else 0
                    if last_price >= MIN_PRICE:
                        price_filtered.append(ticker)
                logger.info(f"Price filtered {len(price_filtered)} tickers in batch {i//BATCH_SIZE + 1}")

                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                tasks = [fetch_stock_data(session, ticker, 'day', start_date, end_date) for ticker in price_filtered]
                volume_results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in volume_results:
                    if isinstance(result, Exception):
                        continue
                    ticker, df = result
                    if df is None or len(df) == 0:
                        continue
                    avg_volume = df['volume'].mean()
                    if avg_volume >= VOLUME_THRESHOLD:
                        tickers.append(ticker)
                        logger.debug(f"Included {ticker}: Price={df['close'].iloc[-1]}, Avg Volume={avg_volume:,.0f}")

                logger.info(f"Processed batch {i//BATCH_SIZE + 1} of {len(all_tickers)//BATCH_SIZE + 1}")
            except Exception as e:
                logger.error(f"Error processing snapshot batch {i//BATCH_SIZE + 1}: {e}")
            await asyncio.sleep(0.05)

    logger.info(f"Filtered {len(tickers)} tickers")
    with open(CACHE_FILE, 'w') as f:
        json.dump({'timestamp': time.time(), 'tickers': tickers}, f)
    return tickers

async def process_batch(session, tickers):
    tasks = []
    for ticker in tickers:
        for timeframe in TIMEFRAMES:
            tasks.append(analyze_ticker(session, ticker, timeframe))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if r is not None and not isinstance(r, Exception)]

async def main():
    logger.info("Starting script...")
    rest_client = RESTClient(api_key=api_key)
    try:
        tickers = await get_filtered_tickers(rest_client)

        results = []
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(tickers), BATCH_SIZE):
                batch = tickers[i:i + BATCH_SIZE]
                logger.info(f"Processing batch {i//BATCH_SIZE + 1} with {len(batch)} tickers...")
                batch_results = await process_batch(session, batch)
                results.extend(batch_results)
                logger.info(f"Batch {i//BATCH_SIZE + 1} completed. Found {len(batch_results)} patterns.")
                await asyncio.sleep(0.2)

        logger.info(f"Total patterns found: {len(results)}")

        # Group results by ticker
        ticker_groups = {}
        timeframe_map = {'4hour': '4H', 'day': 'D', 'week': 'W', 'month': 'M', 'quarter': 'Q'}
        for result in results:
            ticker = result['ticker']
            if ticker not in ticker_groups:
                ticker_groups[ticker] = {
                    'inside': [],
                    'engulfing': [],
                    'avg_float_traded': result['avg_float_traded'],
                    'previous_earnings': None,  # Initialize as None
                    'next_earnings': None  # Initialize as None
                }
            if result['inside_bar']:
                ticker_groups[ticker]['inside'].append(timeframe_map[result['timeframe']])
            if result['engulfing_bar']:
                ticker_groups[ticker]['engulfing'].append(timeframe_map[result['timeframe']])

        # Filter tickers with no inside or engulfing bars
        filtered_tickers = {
            ticker: data for ticker, data in ticker_groups.items()
            if data['inside'] or data['engulfing']
        }

        # Fetch earnings dates for filtered tickers only
        logger.info(f"Fetching earnings dates for {len(filtered_tickers)} filtered tickers...")
        ny_tz = pytz.timezone('America/New_York')  # Match yfinance timezone
        current_date = datetime.now(ny_tz)
        for ticker in filtered_tickers:
            try:
                yf_ticker = yf.Ticker(ticker)
                earnings_dates = yf_ticker.get_earnings_dates(limit=12)  # Get up to 12 quarters
                if earnings_dates is not None and not earnings_dates.empty:
                    # Convert index to timezone-aware datetime
                    earnings_dates.index = pd.to_datetime(earnings_dates.index).tz_convert(ny_tz)
                    # Previous earnings: most recent date before current date
                    past_dates = earnings_dates[earnings_dates.index < current_date]
                    if not past_dates.empty:
                        filtered_tickers[ticker]['previous_earnings'] = past_dates.index.max()
                    # Next earnings: earliest date on or after current date
                    future_dates = earnings_dates[earnings_dates.index >= current_date]
                    if not future_dates.empty:
                        filtered_tickers[ticker]['next_earnings'] = future_dates.index.min()
            except Exception as e:
                logger.warning(f"Couldn't fetch earnings dates for {ticker}: {e}")

        # Sort filtered tickers by avg_float_traded
        sorted_tickers = sorted(filtered_tickers.items(), key=lambda x: x[1]['avg_float_traded'])

        # Print output
        print("\nSTOCK PATTERNS")
        print("-" * 70)
        print(f"{'Ticker':<8} {'Inside':<10} {'Engulfing':<12} {'Float Traded':>12} {'Prev Earnings':>15} {'Next Earnings':>15}")
        print("." * 70)
        for ticker, data in sorted_tickers:
            inside = ", ".join(data['inside']) if data['inside'] else ""
            engulfing = ", ".join(data['engulfing']) if data['engulfing'] else ""
            prev_earnings = data['previous_earnings'].strftime('%Y-%m-%d') if data['previous_earnings'] is not None else "N/A"
            next_earnings = data['next_earnings'].strftime('%Y-%m-%d') if data['next_earnings'] is not None else "N/A"
            print(f"{ticker:<8} {inside:<10} {engulfing:<12} {data['avg_float_traded']:>10.2%} {prev_earnings:>15} {next_earnings:>15}")
        print("-" * 70)
    finally:
        logger.info("Script execution completed.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Script interrupted by user. Shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        logger.info("Script execution completed.")
