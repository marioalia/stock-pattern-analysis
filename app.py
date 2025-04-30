import os
import logging
import asyncio
import aiohttp
import json
import time
from polygon.rest import RESTClient
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import streamlit as st
import schedule
import threading
import pytz

# Configuration Variables
MIN_PRICE = 1.0  # Very low to include most stocks
VOLUME_THRESHOLD = 100_000  # Low to include more stocks
TIMEFRAMES = ['4hour', 'day', 'week', 'month', 'quarter']
BATCH_SIZE = 700  # Number of tickers to process per batch
CACHE_FILE = "/tmp/filtered_inside.json"
CACHE_EXPIRY = 86400  # Cache expiry in seconds (24 hours)
VOLUME_LOOKBACK = 10  # Days for relative volume calculation
RELATIVE_VOLUME_THRESHOLD = 0.5  # Very low to include most tickers

# Use environment variable for Polygon API key
API_KEY = os.getenv("POLYGON_API_KEY")

# Setup Logging with DEBUG level
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/polygon_script.log')
    ]
)
logger = logging.getLogger(__name__)

# Streamlit UI
st.title("Pattern Analysis")
st.markdown("Analyzing stocks for inside and engulfing patterns")

# Initialize Polygon client
client = RESTClient(API_KEY) if API_KEY else None

# State to store analysis results
if 'analysis_df' not in st.session_state:
    st.session_state.analysis_df = None
if 'last_run' not in st.session_state:
    st.session_state.last_run = None
if 'scheduler_started' not in st.session_state:
    st.session_state.scheduler_started = False

async def fetch_stock_data(session, ticker, timeframe, start_date, end_date, max_retries=3):
    """Fetch stock data from Polygon API."""
    if timeframe == '4hour':
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/4/hour/{start_date}/{end_date}"
    else:
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{timeframe}/{start_date}/{end_date}"
    params = {"apiKey": API_KEY, "limit": 5000}
    for attempt in range(max_retries):
        try:
            async with session.get(url, params=params) as response:
                logger.debug(f"Fetching {ticker} on {timeframe}: HTTP {response.status}")
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
                        logger.debug(f"Fetched {len(df)} bars for {ticker} on {timeframe}")
                        return ticker, df
                    else:
                        logger.debug(f"No results for {ticker} on {timeframe}")
                        return ticker, None
                elif response.status == 429:
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limit hit for {ticker} on {timeframe}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.warning(f"Failed to fetch {ticker} on {timeframe}: HTTP {response.status}")
                    return ticker, None
        except Exception as e:
            logger.error(f"Error fetching {ticker} on {timeframe}: {e}")
            return ticker, None
    logger.error(f"Max retries ({max_retries}) exceeded for {ticker} on {timeframe}")
    return ticker, None

def is_inside_bar(current, previous):
    """Check if the current bar is an inside bar."""
    return (current['high'] <= previous['high'] and current['low'] >= previous['low'])

def is_engulfing_bar(current, previous):
    """Check if the current bar is an engulfing bar."""
    current_range = current['high'] - current['low']
    previous_range = previous['high'] - previous['low']
    return (current_range > previous_range and 
            ((current['close'] > current['open'] and current['high'] >= previous['high'] and 
              current['low'] <= previous['low'] and current['close'] > previous['open']) or 
             (current['close'] < current['open'] and current['high'] >= previous['high'] and 
              current['low'] <= previous['low'] and current['close'] < previous['open'])))

async def analyze_ticker(session, ticker, timeframe):
    """Analyze a ticker for patterns, relative volume, and earnings."""
    logger.debug(f"Analyzing {ticker} on {timeframe}")
    end_date = datetime.now().strftime('%Y-%m-%d')
    if timeframe == '4hour':
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    else:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    # Fetch price data for pattern analysis
    ticker, df = await fetch_stock_data(session, ticker, timeframe, start_date, end_date)
    if df is None or len(df) < 2:
        logger.debug(f"No valid price data for {ticker} on {timeframe}")
        return None

    latest_price = df['close'].iloc[-1]
    if latest_price < MIN_PRICE:
        logger.debug(f"{ticker} price {latest_price} below {MIN_PRICE}")
        return None

    current_bar = df.iloc[-1]
    previous_bar = df.iloc[-2]

    # Fetch daily data for relative volume
    volume_start_date = (datetime.now() - timedelta(days=VOLUME_LOOKBACK + 1)).strftime('%Y-%m-%d')
    _, volume_df = await fetch_stock_data(session, ticker, 'day', volume_start_date, end_date)
    relative_volume = None
    if volume_df is not None and len(volume_df) >= 5:  # Relaxed to 5 days
        current_volume = volume_df['volume'].iloc[-1]
        past_volumes = volume_df['volume'].iloc[:-1]
        if past_volumes.mean() > 0:
            relative_volume = current_volume / past_volumes.mean()
            if relative_volume < RELATIVE_VOLUME_THRESHOLD:
                logger.debug(f"{ticker} relative volume {relative_volume:.2f}x below {RELATIVE_VOLUME_THRESHOLD}x")
                return None
        else:
            logger.debug(f"No valid volume data for {ticker}")
            return None
    else:
        logger.debug(f"Insufficient volume data for {ticker}: {len(volume_df) if volume_df is not None else 0} days")
        return None

    # Fetch earnings data using yfinance
    last_earnings_date = None
    next_earnings_date = None
    try:
        stock = yf.Ticker(ticker)
        earnings_dates = stock.earnings_dates
        if earnings_dates is not None and not earnings_dates.empty:
            last_earnings_date = earnings_dates.index[0].strftime('%Y-%m-%d')
            last_date = datetime.strptime(last_earnings_date, '%Y-%m-%d')
            next_earnings_date = (last_date + timedelta(days=90)).strftime('%Y-%m-%d')
        else:
            logger.debug(f"No earnings data for {ticker}")
    except Exception as e:
        logger.debug(f"Error fetching earnings for {ticker}: {e}")

    result = {
        'ticker': ticker,
        'timeframe': timeframe,
        'inside_bar': is_inside_bar(current_bar, previous_bar),
        'engulfing_bar': is_engulfing_bar(current_bar, previous_bar),
        'relative_volume': relative_volume,
        'last_earnings_date': last_earnings_date,
        'next_earnings_date': next_earnings_date
    }
    logger.debug(f"Result for {ticker} on {timeframe}: {result}")
    return result

async def get_filtered_tickers(rest_client):
    """Get a list of filtered tickers based on price and volume."""
    logger.debug("Checking cache")
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
            if time.time() - cache_data['timestamp'] < CACHE_EXPIRY and cache_data['tickers']:
                logger.info(f"Using cached tickers: {len(cache_data['tickers'])}")
                return cache_data['tickers']
            else:
                logger.debug("Cache expired or empty")
        except Exception as e:
            logger.warning(f"Error reading cache: {e}")
            os.remove(CACHE_FILE) if os.path.exists(CACHE_FILE) else None

    tickers = []
    try:
        all_tickers = [ticker.ticker for ticker in rest_client.list_tickers(market='stocks', type='CS', active=True, limit=1000)]
        logger.info(f"Retrieved {len(all_tickers)} active U.S. stock tickers")
    except Exception as e:
        logger.error(f"Error fetching ticker list: {e}")
        return tickers

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
                logger.debug(f"Batch {i//BATCH_SIZE + 1}: {len(price_filtered)} tickers passed price filter")

                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                tasks = [fetch_stock_data(session, ticker, 'day', start_date, end_date) for ticker in price_filtered]
                volume_results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in volume_results:
                    if isinstance(result, Exception):
                        logger.warning(f"Volume fetch error: {result}")
                        continue
                    ticker, df = result
                    if df is None or len(df) == 0:
                        logger.debug(f"No volume data for {ticker}")
                        continue
                    avg_volume = df['volume'].mean()
                    if avg_volume >= VOLUME_THRESHOLD:
                        tickers.append(ticker)
                        logger.debug(f"Included {ticker}: Price={df['close'].iloc[-1]}, Avg Volume={avg_volume:,.0f}")

                logger.info(f"Processed batch {i//BATCH_SIZE + 1}")
            except Exception as e:
                logger.error(f"Error in batch {i//BATCH_SIZE + 1}: {e}")
            await asyncio.sleep(0.1)

    logger.info(f"Filtered {len(tickers)} tickers")
    if tickers:
        try:
            with open(CACHE_FILE, 'w') as f:
                json.dump({'timestamp': time.time(), 'tickers': tickers}, f)
            logger.info(f"Saved {len(tickers)} tickers to cache")
        except Exception as e:
            logger.warning(f"Error saving cache: {e}")
    return tickers

async def process_batch(session, tickers):
    """Process a batch of tickers for analysis."""
    tasks = []
    for ticker in tickers:
        for timeframe in TIMEFRAMES:
            tasks.append(analyze_ticker(session, ticker, timeframe))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    valid_results = [r for r in results if r is not None and not isinstance(r, Exception)]
    logger.debug(f"Batch processed: {len(valid_results)} valid results")
    return valid_results

@st.cache_data
def run_analysis(_api_key):
    """Run the main analysis to find patterns."""
    async def main():
        logger.info("Starting script...")
        if not _api_key:
            logger.error("No valid Polygon API key provided.")
            return pd.DataFrame(columns=["Ticker", "Inside", "Engulfing", "Relative Volume", "Last Earnings", "Next Earnings"])
        rest_client = RESTClient(api_key=_api_key)
        try:
            tickers = await get_filtered_tickers(rest_client)
            logger.info(f"Retrieved {len(tickers)} filtered tickers")
            logger.debug(f"Tickers: {tickers[:10]}...")

            if not tickers:
                logger.warning("No tickers meet the filtering criteria.")
                return pd.DataFrame(columns=["Ticker", "Inside", "Engulfing", "Relative Volume", "Last Earnings", "Next Earnings"])

            results = []
            async with aiohttp.ClientSession() as session:
                for i in range(0, len(tickers), BATCH_SIZE):
                    batch = tickers[i:i + BATCH_SIZE]
                    logger.info(f"Processing batch {i//BATCH_SIZE + 1} with {len(batch)} tickers...")
                    batch_results = await process_batch(session, batch)
                    results.extend(batch_results)
                    logger.info(f"Batch {i//BATCH_SIZE + 1} completed. Found {len(batch_results)} patterns.")
                    await asyncio.sleep(0.1)

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
                        'relative_volume': result['relative_volume'],
                        'last_earnings_date': result['last_earnings_date'],
                        'next_earnings_date': result['next_earnings_date']
                    }
                if result['inside_bar']:
                    ticker_groups[ticker]['inside'].append(timeframe_map[result['timeframe']])
                if result['engulfing_bar']:
                    ticker_groups[ticker]['engulfing'].append(timeframe_map[result['timeframe']])

            # Filter tickers with patterns
            filtered_tickers = {
                ticker: data for ticker, data in ticker_groups.items()
                if data['inside'] or data['engulfing']
            }
            logger.info(f"Tickers with patterns: {len(filtered_tickers)}")

            # Sort by relative volume
            sorted_tickers = sorted(filtered_tickers.items(), key=lambda x: x[1]['relative_volume'] or 0, reverse=True)

            # Prepare DataFrame
            data = []
            for ticker, data_dict in sorted_tickers:
                inside = ", ".join(data_dict['inside']) if data_dict['inside'] else ""
                engulfing = ", ".join(data_dict['engulfing']) if data_dict['engulfing'] else ""
                relative_volume = data_dict['relative_volume']
                relative_volume_str = f"{relative_volume:.2f}x" if relative_volume is not None else "N/A"
                data.append({
                    "Ticker": ticker,
                    "Inside": inside,
                    "Engulfing": engulfing,
                    "Relative Volume": relative_volume_str,
                    "Last Earnings": data_dict['last_earnings_date'] or "N/A",
                    "Next Earnings": data_dict['next_earnings_date'] or "N/A"
                })
            df = pd.DataFrame(data)
            logger.info(f"DataFrame created: {len(df)} rows")
            return df if not df.empty else pd.DataFrame(columns=["Ticker", "Inside", "Engulfing", "Relative Volume", "Last Earnings", "Next Earnings"])
        except Exception as e:
            logger.error(f"Error in run_analysis: {e}", exc_info=True)
            return pd.DataFrame(columns=["Ticker", "Inside", "Engulfing", "Relative Volume", "Last Earnings", "Next Earnings"])
        finally:
            logger.info("Script execution completed.")

    return asyncio.run(main())

def clear_cache_and_run():
    """Delete the JSON cache file and run analysis."""
    logger.info("Starting clear_cache_and_run")
    try:
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
            logger.info(f"Deleted cache file: {CACHE_FILE}")
        if not API_KEY:
            logger.error("No Polygon API key provided.")
            st.error("No Polygon API key provided.")
            return
        logger.info(f"Running analysis with relative volume threshold: {RELATIVE_VOLUME_THRESHOLD}x...")
        progress_bar = st.progress(0)
        df = run_analysis(API_KEY)
        progress_bar.progress(100)
        st.session_state.analysis_df = df
        st.session_state.last_run = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S %Z')
        logger.info("Analysis completed successfully.")
    except Exception as e:
        logger.error(f"Error in clear_cache_and_run: {e}", exc_info=True)
        st.error(f"Error during analysis: {str(e)}")
    finally:
        logger.info("Completed clear_cache_and_run")

def run_scheduler():
    """Run the scheduler in a background thread."""
    eastern = pytz.timezone('US/Eastern')
    schedule.every().day.at("13:31", eastern).do(clear_cache_and_run)
    schedule.every().day.at("16:01", eastern).do(clear_cache_and_run)
    while True:
        schedule.run_pending()
        time.sleep(60)

# Sidebar
with st.sidebar:
    st.header("Controls")
    st.markdown(f"**Last Run**: {st.session_state.get('last_run', 'Not yet run')}")
    st.markdown("---")
    st.markdown("**Run Manually**")
    if st.button("Run Analysis Now"):
        if API_KEY:
            with st.spinner("Running analysis..."):
                clear_cache_and_run()
        else:
            st.error("No Polygon API key provided.")

# Run analysis on startup if no results exist
if st.session_state.analysis_df is None and API_KEY:
    with st.spinner("Running initial analysis..."):
        clear_cache_and_run()

# Start scheduler in a background thread
if not st.session_state.scheduler_started and API_KEY:
    st.session_state.scheduler_started = True
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    logger.info("Scheduler started.")

# Main content
st.subheader("Results")
if st.session_state.analysis_df is not None and not st.session_state.analysis_df.empty:
    search = st.text_input("Search Tickers", "")
    df = st.session_state.analysis_df.copy()
    if search:
        df = df[df['Ticker'].str.contains(search, case=False, na=False)]
    st.success(f"Found {len(df)} stocks with patterns (Relative Volume ≥ {RELATIVE_VOLUME_THRESHOLD}x). Last run: {st.session_state.last_run}")
    if not df.empty:
        st.dataframe(df, use_container_width=True, height=800)
    else:
        st.warning("No tickers match the search criteria.")
else:
    st.info("No analysis results yet. Waiting for scheduled run or check API key.")