import os
import logging
import asyncio
import aiohttp
import json
import time
from polygon.rest import RESTClient
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import streamlit as st
import schedule
import threading
import pytz

# Configuration Variables
MIN_PRICE = 10.0
VOLUME_THRESHOLD = 2_000_000
TIMEFRAMES = ['4hour', 'day', 'week', 'month', 'quarter']
BATCH_SIZE = 700
CACHE_FILE = "/tmp/filtered_inside.json"
CACHE_EXPIRY = 86400
FLOAT_VOL_LOOKBACK = 10
FLOAT_TRADED_THRESHOLD = 0.03

# Use environment variable for API key
API_KEY = os.getenv("POLYGON_API_KEY")

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/polygon_script.log')
    ]
)
logger = logging.getLogger(__name__)

# Streamlit UI
st.title("Stock Pattern Analysis")
st.markdown("""
This app analyzes U.S. stocks for inside and engulfing bar patterns across multiple timeframes using Polygon.io data.
The analysis runs automatically at 1:31 PM and 4:01 PM Eastern Time daily, with fresh data.
""")

# Initialize Polygon client
client = RESTClient(API_KEY)

# State to store analysis results
if 'analysis_df' not in st.session_state:
    st.session_state.analysis_df = None
if 'last_run' not in st.session_state:
    st.session_state.last_run = None
if 'scheduler_started' not in st.session_state:
    st.session_state.scheduler_started = False

async def fetch_stock_data(session, ticker, timeframe, start_date, end_date):
    if timeframe == '4hour':
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/4/hour/{start_date}/{end_date}"
    else:
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{timeframe}/{start_date}/{end_date}"
    params = {"apiKey": API_KEY, "limit": 5000}
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

@st.cache_data
def run_analysis(_api_key):
    async def main():
        logger.info("Starting script...")
        rest_client = RESTClient(api_key=_api_key)
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
                        'avg_float_traded': result['avg_float_traded']
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

            # Sort filtered tickers by avg_float_traded in descending order
            sorted_tickers = sorted(filtered_tickers.items(), key=lambda x: x[1]['avg_float_traded'], reverse=True)

            # Prepare DataFrame for display
            data = []
            for ticker, data_dict in sorted_tickers:
                inside = ", ".join(data_dict['inside']) if data_dict['inside'] else ""
                engulfing = ", ".join(data_dict['engulfing']) if data_dict['engulfing'] else ""
                data.append({
                    "Ticker": ticker,
                    "Inside": inside,
                    "Engulfing": engulfing,
                    "Float Traded": data_dict['avg_float_traded']
                })
            df = pd.DataFrame(data)
            return df
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
        finally:
            logger.info("Script execution completed.")

    # Run the async main function
    return asyncio.run(main())

def clear_cache_and_run():
    """Delete the JSON cache file and run analysis."""
    try:
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
            logger.info(f"Deleted cache file: {CACHE_FILE}")
        if API_KEY:
            logger.info("Running analysis...")
            df = run_analysis(API_KEY)
            st.session_state.analysis_df = df
            st.session_state.last_run = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S %Z')
            logger.info("Analysis completed.")
        else:
            logger.error("No Polygon API key provided.")
            st.error("No Polygon API key provided.")
    except Exception as e:
        logger.error(f"Error in clear_cache_and_run: {e}")
        st.error(f"Error during analysis: {str(e)}")

def run_scheduler():
    """Run the scheduler in a background thread."""
    eastern = pytz.timezone('US/Eastern')
    schedule.every().day.at("13:31", eastern).do(clear_cache_and_run)
    schedule.every().day.at("16:01", eastern).do(clear_cache_and_run)
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

# Run analysis on startup if no results exist
if st.session_state.analysis_df is None and API_KEY:
    with st.spinner("Running initial analysis... This may take a few minutes."):
        clear_cache_and_run()

# Start scheduler in a background thread
if not st.session_state.scheduler_started:
    st.session_state.scheduler_started = True
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    logger.info("Scheduler started.")

# Manual run button
if st.button("Run Analysis Now"):
    if API_KEY:
        with st.spinner("Running analysis... This may take a few minutes."):
            clear_cache_and_run()
    else:
        st.error("Please provide a valid Polygon API key via environment variable.")

# Display results
if st.session_state.analysis_df is not None:
    st.success(f"Found {len(st.session_state.analysis_df)} stocks with patterns. Last run: {st.session_state.last_run}")
    st.session_state.analysis_df['Float Traded'] = st.session_state.analysis_df['Float Traded'].apply(lambda x: f"{x:.2%}")
    st.dataframe(st.session_state.analysis_df, use_container_width=True, height=800)
else:
    st.info("No analysis results yet. Waiting for scheduled run or click 'Run Analysis Now'.")
