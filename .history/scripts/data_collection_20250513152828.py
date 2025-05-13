import requests
import json
import pandas as pd
from bs4 import BeautifulSoup
from google_play_scraper import app as play_app, Sort
from google_play_scraper import reviews as play_reviews
import numpy as np
from collections import Counter
from datetime import datetime, timedelta
from textblob import TextBlob  # Replacing NLTK with TextBlob for sentiment analysis
import threading  # For basic parallelization
from concurrent.futures import ThreadPoolExecutor  # For thread pool
import time
import random
import re

def fetch_app_store_data(app_id, country='us'):
    """Fetches detailed app data from the App Store."""
    try:
        # Use the iTunes API instead of app_store_scraper which has the 'NoneType' object error
        url = f"https://itunes.apple.com/lookup?id={app_id}&country={country}"
        response = requests.get(url)
        data = response.json()
        
        if not data or 'results' not in data or len(data['results']) == 0:
            print(f"No results found for iOS app ID: {app_id}")
            return None
            
        app_details = data['results'][0]
            
        # Get reviews separately via iTunes API
        reviews_url = f"https://itunes.apple.com/rss/customerreviews/id={app_id}/sortBy=mostRecent/json"
        reviews_response = requests.get(reviews_url)
        
        df_reviews = pd.DataFrame()
        average_rating = None
        review_count = 0
        
        try:
            reviews_data = reviews_response.json()
            
            if reviews_response.status_code == 200 and 'feed' in reviews_data and 'entry' in reviews_data['feed']:
                reviews = reviews_data['feed']['entry']
                # First entry is the app info, not a review
                if len(reviews) > 1:
                    reviews = reviews[1:]  # Skip the first entry which is app info
                    
                    review_data = []
                    for review in reviews:
                        try:
                            review_dict = {
                                'rating': float(review.get('im:rating', {}).get('label', 0)),
                                'title': review.get('title', {}).get('label', ''),
                                'content': review.get('content', {}).get('label', ''),
                                'author': review.get('author', {}).get('name', {}).get('label', ''),
                                'date': review.get('updated', {}).get('label', '')
                            }
                            review_data.append(review_dict)
                        except Exception as e:
                            # Skip individual review errors
                            pass
                    
                    if review_data:
                        df_reviews = pd.DataFrame(review_data)
                        average_rating = df_reviews['rating'].mean() if not df_reviews.empty else None
                        review_count = len(df_reviews)
        except Exception as e:
            print(f"Error fetching reviews for iOS app ID {app_id}: {e}")
            # Continue without reviews

        return {
            "platform": "iOS",
            "app_name": app_details.get('trackName'),
            "app_id": app_id,
            "category": app_details.get('primaryGenreName'),
            "rating": app_details.get('averageUserRating', average_rating),
            "rating_count": app_details.get('userRatingCount', review_count),
            "downloads": None,  # Not available in iTunes API
            "price": app_details.get('price'),
            "developer": app_details.get('artistName'),
            "release_date": app_details.get('releaseDate'),
            "updated_date": app_details.get('currentVersionReleaseDate'),
            "description": app_details.get('description'),
            "reviews": df_reviews,  # Include reviews
        }
    except Exception as e:
        print(f"Error fetching iOS data for {app_id}: {e}")
        return None

def fetch_play_store_data(app_id):
    """Fetches detailed app data and reviews from the Google Play Store."""

    try:
        app_details = play_app(app_id)
        app_reviews = play_reviews(app_id, sort=Sort.NEWEST, count=200)

        if app_reviews and app_reviews[0]:
            df_reviews = pd.DataFrame(app_reviews[0])
            average_rating = df_reviews['score'].mean()
            review_count = len(df_reviews)
        else:
            average_rating = None
            review_count = 0
            df_reviews = pd.DataFrame()

        return {
            "platform": "Android",
            "app_name": app_details.get('title'),
            "app_id": app_id,
            "category": app_details.get('genre'),
            "rating": average_rating,
            "rating_count": review_count,
            "downloads": app_details.get('installs'),
            "price": app_details.get('price'),
            "developer": app_details.get('developer'),
            "release_date": app_details.get('released'),
            "updated_date": app_details.get('updated'),
            "description": app_details.get('description'),
            "reviews": df_reviews,  # Include reviews
        }
    except Exception as e:
        print(f"Error fetching Play Store data for {app_id}: {e}")
        return None

def fetch_sensor_tower_data(app_id, platform):
    """Fetches data from Sensor Tower website through scraping."""
    try:
        # Format differs for iOS and Android
        if platform == "iOS":
            url = f"https://sensortower.com/ios/us/app/{app_id}/overview"
        else:
            url = f"https://sensortower.com/android/us/app/{app_id}/overview"
            
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch Sensor Tower data: {response.status_code}")
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract basic data (this is placeholder as actual site structure will vary)
        download_estimate = None
        revenue_estimate = None
        
        download_elem = soup.select_one('.downloads-value')
        if download_elem:
            download_estimate = download_elem.text.strip()
            
        revenue_elem = soup.select_one('.revenue-value')
        if revenue_elem:
            revenue_estimate = revenue_elem.text.strip()
            
        return {
            "download_estimate": download_estimate,
            "revenue_estimate": revenue_estimate,
            "source": "Sensor Tower"
        }
        
    except Exception as e:
        print(f"Error fetching Sensor Tower data for {app_id}: {e}")
        return None

def fetch_app_annie_data(app_id, platform):
    """Fetches data from App Annie (data.ai) through scraping."""
    try:
        if platform == "iOS":
            url = f"https://www.data.ai/apps/ios/{app_id}/app"
        else:
            url = f"https://www.data.ai/apps/google-play/{app_id}/app"
            
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch App Annie data: {response.status_code}")
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract data (placeholder, structure will differ)
        rank_data = None
        usage_data = None
        
        rank_elem = soup.select_one('.rank-value')
        if rank_elem:
            rank_data = rank_elem.text.strip()
            
        usage_elem = soup.select_one('.usage-stats')
        if usage_elem:
            usage_data = usage_elem.text.strip()
            
        return {
            "category_rank": rank_data,
            "usage_metrics": usage_data,
            "source": "App Annie"
        }
        
    except Exception as e:
        print(f"Error fetching App Annie data for {app_id}: {e}")
        return None

def fetch_appbrain_data(app_id):
    """Fetches Android app data from AppBrain."""
    try:
        url = f"https://www.appbrain.com/app/{app_id}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch AppBrain data: {response.status_code}")
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract data from AppBrain
        installs_elem = soup.select_one('.installs-text')
        installs = installs_elem.text.strip() if installs_elem else None
        
        rank_elem = soup.select_one('.rank-text')
        rank = rank_elem.text.strip() if rank_elem else None
        
        return {
            "appbrain_installs": installs,
            "appbrain_rank": rank,
            "source": "AppBrain"
        }
        
    except Exception as e:
        print(f"Error fetching AppBrain data for {app_id}: {e}")
        return None

def analyze_reviews(reviews_df):
    """Performs sentiment analysis and extracts keywords from reviews using TextBlob."""

    if reviews_df is None or reviews_df.empty:
        return 0, 0, [], 0  # Return defaults for empty reviews

    # Sentiment Analysis with TextBlob
    def get_sentiment(text):
        try:
            return TextBlob(str(text)).sentiment.polarity
        except:
            return 0
    
    reviews_df['sentiment_score'] = reviews_df['content'].apply(get_sentiment)
    
    positive_sentiment_ratio = (reviews_df['sentiment_score'] > 0.2).mean()  # Tunable threshold
    negative_sentiment_ratio = (reviews_df['sentiment_score'] < -0.2).mean()
    avg_sentiment = reviews_df['sentiment_score'].mean()

    # Keyword Extraction (Basic - Improve with TF-IDF, etc.)
    all_text = ' '.join(reviews_df['content'].astype(str).tolist())
    words = all_text.lower().split()
    keyword_counts = Counter(words).most_common(10)
    keywords = [word for word, count in keyword_counts]

    return positive_sentiment_ratio, negative_sentiment_ratio, keywords, avg_sentiment

def calculate_feature_engineering(app_data):
    """Calculates engineered features."""

    if app_data is None:
        return None

    # Data Validation and Cleaning
    try:
        if isinstance(app_data.get('downloads'), str):
            app_data['downloads'] = float(app_data['downloads'].replace('+', '').replace(',', ''))
        if isinstance(app_data.get('price'), str):
            app_data['price'] = float(app_data['price'].replace('$', ''))
    except (ValueError, TypeError) as e:
        print(f"Data conversion error: {e}. Setting downloads/price to NaN")
        app_data['downloads'] = np.nan
        app_data['price'] = np.nan

    reviews_df = app_data.pop('reviews', pd.DataFrame())  # Extract reviews

    positive_sentiment_ratio, negative_sentiment_ratio, keywords, avg_sentiment = analyze_reviews(reviews_df)

    app_data['positive_sentiment_ratio'] = positive_sentiment_ratio
    app_data['negative_sentiment_ratio'] = negative_sentiment_ratio
    app_data['avg_review_sentiment'] = avg_sentiment
    app_data['keywords'] = keywords

    # Time-Based Features
    try:
        release_date = pd.to_datetime(app_data.get('release_date'))
        if pd.notna(release_date):
            app_data['release_date'] = release_date
            app_data['days_since_release'] = (datetime.now().date() - release_date.date()).days
        else:
            app_data['days_since_release'] = np.nan
    except (TypeError, ValueError, AttributeError):
        app_data['days_since_release'] = np.nan  # Handle missing dates

    try:
        updated_date = pd.to_datetime(app_data.get('updated_date'))
        if pd.notna(updated_date):
            app_data['updated_date'] = updated_date
            app_data['days_since_last_update'] = (datetime.now().date() - updated_date.date()).days
        else:
            app_data['days_since_last_update'] = np.nan
    except (TypeError, ValueError, AttributeError):
        app_data['days_since_last_update'] = np.nan

    # Interaction Features - Add null checking to avoid NoneType errors
    rating = app_data.get('rating', 0)
    downloads = app_data.get('downloads', 0)
    price = app_data.get('price', 1e-9)
    
    # Convert None values to 0 or appropriate defaults
    rating = 0 if rating is None else rating
    downloads = 0 if downloads is None else downloads
    price = 1e-9 if price is None or price == 0 else price
    
    app_data['rating_x_downloads'] = rating * downloads
    app_data['rating_per_price'] = rating / price  # Already has check to avoid division by zero

    return app_data

def fetch_and_process_app_data(platform, app_id):
    """Fetches and processes data for a single app from multiple sources."""
    try:
        # Get primary data
        if platform == "iOS":
            raw_data = fetch_app_store_data(app_id)
        elif platform == "Android":
            raw_data = fetch_play_store_data(app_id)
        else:
            return None

        if not raw_data:
            return None
            
        # Add delay to avoid rate limiting
        time.sleep(random.uniform(1.5, 3))
        
        # Disable Sensor Tower and App Annie since they're consistently returning 403/429 errors
        # This can be enabled later if needed with proper rate limiting and user agents
        """
        # Get additional data from other sources
        try:
            sensor_tower_data = fetch_sensor_tower_data(app_id, platform)
            if sensor_tower_data:
                raw_data.update(sensor_tower_data)
        except Exception as e:
            print(f"Error fetching Sensor Tower data: {e}")
            
        try:
            app_annie_data = fetch_app_annie_data(app_id, platform)
            if app_annie_data:
                raw_data.update(app_annie_data)
        except Exception as e:
            print(f"Error fetching App Annie data: {e}")
        """
            
        # For Android apps, also try AppBrain
        if platform == "Android":
            try:
                appbrain_data = fetch_appbrain_data(app_id)
                if appbrain_data:
                    raw_data.update(appbrain_data)
            except Exception as e:
                print(f"Error fetching AppBrain data: {e}")
                
        # Process all collected data
        return calculate_feature_engineering(raw_data)
    except Exception as e:
        print(f"Error processing {platform} - {app_id}: {e}")
        return None

def main():
    # App IDs to collect data for (expanded list)
    app_store_ids = [
        # Social Media Apps
        '284882215',   # Facebook
        '389801252',   # Instagram
        '333903271',   # Twitter/X
        '544007664',   # LinkedIn
        '447188370',   # Snapchat
        '835599320',   # TikTok
        '310633997',   # WhatsApp
        '686449807',   # Telegram
        '1147741809',  # Pinterest
        '547702041',   # Clubhouse
        
        # Entertainment Apps
        '284035177',   # Pandora
        '324684580',   # Spotify
        '336353433',   # SoundCloud
        '1108187390',  # Amazon Music
        '510855668',   # Twitch
        '544007664',   # YouTube
        '363590051',   # Netflix
        '376510438',   # Hulu
        '545519333',   # Disney+
        '1116599773',  # HBO Max
        
        # Productivity
        '288429040',   # Dropbox
        '310633997',   # Google Drive
        '1274495053',  # Microsoft Teams
        '546505307',   # Slack
        '425424353',   # Evernote
        '422689480',   # Gmail
        '585829637',   # Todoist
        '904237743',   # Google Keep
        '568903335',   # Workflowy
        '775737172',   # Microsoft Office
        
        # Shopping
        '297606951',   # Amazon
        '335047888',   # eBay
        '363590051',   # Etsy
        '387428489',   # Wish
        '512142109',   # Aliexpress
        '878577184',   # Target
        '338186909',   # Walmart
        '342792525',   # Best Buy
        '342184805',   # Home Depot
        
        # Travel
        '284876795',   # TripAdvisor
        '284013126',   # Expedia
        '499726199',   # Airbnb
        '507874739',   # Hotel Tonight
        '284235722',   # Hotels.com
        '576649420',   # Uber
        '298330720',   # Lyft
        '288429040',   # Google Maps
        '510855668',   # Waze
        
        # Health & Fitness
        '454638411',   # MyFitnessPal
        '571800810',   # Fitbit
        '1026156606',  # Headspace
        '660010415',   # Strava
        '449099506',   # Sleep Cycle
        '557137771',   # 7 Minute Workout
        '553386675',   # Calm
        '1003655329',  # Nike Training Club
        '536141812',   # Peloton
        
        # Finance
        '283646709',   # PayPal
        '389801252',   # Venmo
        '284855289',   # Square Cash
        '331703417',   # Mint
        '286762344',   # Robinhood
        '298867247',   # Acorns
        '311867728',   # TD Ameritrade
        '311548241',   # E*TRADE
        '352509070',   # Wells Fargo
        '284855289',   # Chase
        
        # Games
        '553834731',   # Candy Crush Saga
        '553834720',   # Pokémon GO
        '406889139',   # Roblox
        '479516143',   # Clash of Clans
        '297606951',   # Minecraft
        '330633869',   # Among Us
        '413536863',   # PUBG Mobile
        '1229016807',  # Fortnite
        '835599320',   # Call of Duty Mobile
        '1246560442',  # Genshin Impact
    ]

    play_store_ids = [
        # Social Media
        'com.facebook.katana',
        'com.instagram.android',
        'com.twitter.android',
        'com.linkedin.android',
        'com.snapchat.android',
        'com.zhiliaoapp.musically',
        'com.whatsapp',
        'org.telegram.messenger',
        'com.pinterest',
        'com.clubhouse.app',
        
        # Entertainment
        'com.pandora.android',
        'com.spotify.music',
        'com.soundcloud.android',
        'com.amazon.mp3',
        'tv.twitch.android.app',
        'com.google.android.youtube',
        'com.netflix.mediaclient',
        'com.hulu.plus',
        'com.disney.disneyplus',
        'com.hbo.hbonow',
        
        # Productivity
        'com.dropbox.android',
        'com.google.android.apps.docs',
        'com.microsoft.teams',
        'com.Slack',
        'com.evernote',
        'com.google.android.gm',
        'com.todoist',
        'com.google.android.keep',
        'com.workflowy.android',
        'com.microsoft.office.officehubrow',
        
        # Shopping
        'com.amazon.mShop.android.shopping',
        'com.ebay.mobile',
        'com.etsy.android',
        'com.contextlogic.wish',
        'com.alibaba.aliexpresshd',
        'com.target.ui',
        'com.walmart.android',
        'com.bestbuy.android',
        'com.thehomedepot',
        
        # Travel
        'com.tripadvisor.tripadvisor',
        'com.expedia.bookings',
        'com.airbnb.android',
        'com.hoteltonight.android.prod',
        'com.hcom.android',
        'com.ubercab',
        'me.lyft.android',
        'com.google.android.apps.maps',
        'com.waze',
        
        # Health & Fitness
        'com.myfitnesspal.android',
        'com.fitbit.FitbitMobile',
        'com.getsomeheadspace.android',
        'com.strava',
        'com.urbandroid.sleep',
        'com.workoutapps.sevenmins',
        'com.calm.android',
        'com.nike.ntc',
        'com.onepeloton.callisto',
        
        # Finance
        'com.paypal.android.p2pmobile',
        'com.venmo',
        'com.squareup.cash',
        'com.mint',
        'com.robinhood.android',
        'com.acorns.android',
        'com.tdameritrade.mobile',
        'com.etrade.mobilepro.activity',
        'com.wf.wellsfargomobile',
        'com.chase.sig.android',
        
        # Games
        'com.king.candycrushsaga',
        'com.nianticlabs.pokemongo',
        'com.roblox.client',
        'com.supercell.clashofclans',
        'com.mojang.minecraftpe',
        'com.innersloth.spacemafia',
        'com.tencent.ig',
        'com.epicgames.fortnite',
        'com.activision.callofduty.shooter',
        'com.miHoYo.GenshinImpact'
    ]

    # Remove any duplicate IDs
    app_store_ids = list(set(app_store_ids))
    play_store_ids = list(set(play_store_ids))

    all_app_data = []
    failed_app_ids = []  # Track failed app IDs

    # Create folders if they don't exist
    import os
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/interim", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # Basic Threading for Parallelism with rate limiting
    with ThreadPoolExecutor(max_workers=3) as executor:  # Reduced workers to avoid rate limits
        futures = []
        for platform, ids in {"iOS": app_store_ids, "Android": play_store_ids}.items():
            for app_id in ids:
                futures.append(executor.submit(fetch_and_process_app_data, platform, app_id))
                time.sleep(2)  # Increased delay between submissions to avoid rate limits

        for future in futures:
            result = future.result()
            if result:
                all_app_data.append(result)

    # Create DataFrame only if we have data
    if all_app_data:
        df = pd.DataFrame(all_app_data)
        
        # Advanced Feature Engineering (after initial data collection)
        df = advanced_feature_engineering(df)

        # Data Validation (after feature engineering)
        df = validate_data(df)

        # Save raw data
        df.to_csv("data/raw/app_data_final.csv", index=False)
        print(f"Final data fetched and saved to data/raw/app_data_final.csv")
        print(f"Collected data for {len(df)} apps")
    else:
        print("No app data was successfully collected.")

def advanced_feature_engineering(df):
    """Performs advanced feature engineering on the combined DataFrame."""

    if df.empty:
        return df

    # Time-Based Lagged Features (example with downloads)
    df['downloads_lag_1'] = df.groupby('app_id')['downloads'].shift(1)
    df['downloads_lag_2'] = df.groupby('app_id')['downloads'].shift(2)

    # Rolling Statistics (example with rating)
    df['rating_rolling_mean'] = df.groupby('app_id')['rating'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean())
    df['rating_rolling_std'] = df.groupby('app_id')['rating'].transform(
        lambda x: x.rolling(window=3, min_periods=1).std())

    # Volatility (example with downloads)
    df['downloads_volatility'] = df.groupby('app_id')['downloads'].transform(
        lambda x: x.rolling(window=3, min_periods=1).std())

    # Complex Feature Interaction (example)
    df['engagement_score'] = (df['rating'] * df['rating_count']) + (df['downloads'] / 10000)

    # Additional competitive analysis features
    if 'category' in df.columns:
        df['category_competition'] = df.groupby('category')['app_id'].transform('count')
        
    # Market penetration indicator
    if 'downloads' in df.columns and 'days_since_release' in df.columns:
        df['download_rate'] = df['downloads'] / (df['days_since_release'] + 1)  # Add 1 to avoid division by zero
        
    # Example Feature Engineering for Category
    if 'category' in df.columns:
        df = pd.get_dummies(df, columns=['category'], prefix='category', dummy_na=True)

    return df

def validate_data(df):
    """Performs data validation."""

    if df.empty:
        return df

    # Example: Check for outliers in 'downloads' using z-score
    if 'downloads' in df.columns:  # Check if the column exists
        df['downloads_zscore'] = np.abs((df['downloads'] - df['downloads'].mean()) / df['downloads'].std())
        df = df[df['downloads_zscore'] < 3].drop(columns=['downloads_zscore'])

    # Example: Check for missing values
    print(f"Missing values before handling:\n{df.isnull().sum()}")
    
    # More advanced imputation methods for numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            # Use median for numeric columns with missing values
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
    
    # For categorical columns, fill with mode
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'keywords' and df[col].isnull().sum() > 0:
            # Set pandas option to avoid the FutureWarning
            pd.set_option('future.no_silent_downcasting', True)
            
            # Get the mode value if it exists
            if not df[col].mode().empty:
                mode_value = df[col].mode()[0]
            else:
                mode_value = "Unknown"
                
            # Fill the missing values and ensure it's an object type
            temp_series = df[col].copy()
            temp_series = temp_series.fillna(mode_value)
            df[col] = temp_series
    
    print(f"Missing values after handling:\n{df.isnull().sum()}")

    # Example: Check for inconsistencies (e.g., ratings outside valid range)
    if 'rating' in df.columns:  # Check if the column exists
        df = df[(df['rating'] >= 0) & (df['rating'] <= 5) | pd.isnull(df['rating'])]

    return df

if __name__ == "__main__":
    main()
