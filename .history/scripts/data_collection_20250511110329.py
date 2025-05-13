import requests
import json
import pandas as pd
from bs4 import BeautifulSoup
from google_play_scraper import app as play_app, Sort
from google_play_scraper import reviews as play_reviews
from app_store_scraper import AppStore  # pip install app-store-scraper
import numpy as np
from collections import Counter
from datetime import datetime, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # NLTK for Sentiment
import nltk
import threading  # For basic parallelization
from concurrent.futures import ThreadPoolExecutor  # For thread pool

# Download NLTK resources (if needed - do this once)
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

def fetch_app_store_data(app_id, country='us'):
    """Fetches detailed app data from the App Store."""
    try:
        app_store = AppStore(country=country, app_name=app_id)
        app_info = app_store.review()
        details = app_store.app_details

        if app_info and app_info['reviews']:
            df_reviews = pd.DataFrame(app_info['reviews'])
            average_rating = df_reviews['rating'].mean()
            review_count = len(df_reviews)
        else:
            average_rating = None
            review_count = 0
            df_reviews = pd.DataFrame()

        return {
            "platform": "iOS",
            "app_name": details.get('trackCensoredName'),
            "app_id": app_id,
            "category": details.get('genres'),
            "rating": average_rating,
            "rating_count": review_count,
            "downloads": details.get('sellerUrl'),  # Placeholder
            "price": details.get('price'),
            "developer": details.get('artistName'),
            "release_date": details.get('releaseDate'),
            "updated_date": details.get('currentVersionReleaseDate'),
            "description": details.get('description'),
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

def analyze_reviews(reviews_df):
    """Performs sentiment analysis and extracts keywords from reviews."""

    if reviews_df is None or reviews_df.empty:
        return 0, [], 0  # Return defaults for empty reviews

    analyzer = SentimentIntensityAnalyzer()

    # Sentiment Analysis
    reviews_df['compound'] = reviews_df['content'].astype(str).apply(lambda x: analyzer.polarity_score(x)['compound'])
    positive_sentiment_ratio = (reviews_df['compound'] > 0.2).mean()  # Tunable threshold
    negative_sentiment_ratio = (reviews_df['compound'] < -0.2).mean()
    avg_sentiment = reviews_df['compound'].mean()

    # Keyword Extraction (Basic - Improve with TF-IDF, etc.)
    all_text = ' '.join(reviews_df['content'].astype(str).tolist())
    words = all_text.lower().split()
    keyword_counts = Counter(words).most_common(10)
    keywords = [word for word, count in keyword_counts]

    return positive_sentiment_ratio, keywords, avg_sentiment

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

    positive_sentiment_ratio, keywords, avg_sentiment = analyze_reviews(reviews_df)

    app_data['positive_sentiment_ratio'] = positive_sentiment_ratio
    app_data['negative_sentiment_ratio'] = negative_sentiment_ratio
    app_data['avg_review_sentiment'] = avg_sentiment
    app_data['keywords'] = keywords

    # Time-Based Features
    try:
        app_data['release_date'] = pd.to_datetime(app_data.get('release_date'))
        app_data['days_since_release'] = (datetime.now().date() - app_data['release_date'].date()).days
    except (TypeError, ValueError):
        app_data['days_since_release'] = np.nan  # Handle missing dates

    try:
        app_data['updated_date'] = pd.to_datetime(app_data.get('updated_date'))
        app_data['days_since_last_update'] = (datetime.now().date() - app_data['updated_date'].date()).days
    except (TypeError, ValueError):
        app_data['days_since_last_update'] = np.nan

    # Interaction Features
    app_data['rating_x_downloads'] = app_data.get('rating', 0) * app_data.get('downloads', 0)
    app_data['rating_per_price'] = app_data.get('rating', 0) / (app_data.get('price', 1e-9) or 1e-9)  # Avoid division by zero

    return app_data

def fetch_and_process_app_data(platform, app_id):
    """Fetches and processes data for a single app."""
    try:
        if platform == "iOS":
            raw_data = fetch_app_store_data(app_id)
        elif platform == "Android":
            raw_data = fetch_play_store_data(app_id)
        else:
            return None

        if raw_data:
            return calculate_feature_engineering(raw_data)
        return None
    except Exception as e:
        print(f"Error processing {platform} - {app_id}: {e}")
        return None

def main():
    app_ids = {
        "iOS": ["com.facebook.Facebook", "com.zhiliaoapp.musically", "com.Slack", "com.spotify.Spotify"],
        "Android": ["com.facebook.katana", "com.ss.android.ugc.trill", "com.Slack", "com.spotify.music"]  # Changed id
    }
    all_app_data = []

    # Basic Threading for Parallelism
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for platform, ids in app_ids.items():
            for app_id in ids:
                futures.append(executor.submit(fetch_and_process_app_data, platform, app_id))

        for future in futures:
            result = future.result()
            if result:
                all_app_data.append(result)

    df = pd.DataFrame(all_app_data)

    # Advanced Feature Engineering (after initial data collection)
    df = advanced_feature_engineering(df)

    # Data Validation (after feature engineering)
    df = validate_data(df)

    df.to_csv("data/raw/app_data_final.csv", index=False)
    print("Final data fetched and saved to data/raw/app_data_final.csv")

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
    df = df.fillna(0)  # Or use a more sophisticated imputation method
    print(f"Missing values after handling:\n{df.isnull().sum()}")

    # Example: Check for inconsistencies (e.g., ratings outside valid range)
    if 'rating' in df.columns:  # Check if the column exists
        df = df[(df['rating'] >= 0) & (df['rating'] <= 5) | pd.isnull(df['rating'])]

    return df

if __name__ == "__main__":
    main()