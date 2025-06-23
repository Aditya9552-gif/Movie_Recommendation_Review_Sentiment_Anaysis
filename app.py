import os
import streamlit as st
import requests
import pandas as pd
import random
import urllib.request
from bs4 import BeautifulSoup
import re
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

# Load models and data
movies = joblib.load('movies.pkl')
transform_vectorizer = joblib.load('transform.pkl')
svc_model = joblib.load('svc_model.pkl')

# Get the TMDB API key from environment variable
tmdb_api_key = os.getenv('TMDB_API_KEY')

if not tmdb_api_key:
    raise ValueError('TMDB_API_KEY environment variable is not set')


def get_imdb_id(tmdb_id):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={tmdb_api_key}&language=en-US"
    response = requests.get(url)
    data = response.json()
    return data.get('imdb_id')


def scrape_imdb_reviews(imdb_movie_id):
    base_imdb_url = "https://www.imdb.com"
    reviews_found = []

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    try:
        if not imdb_movie_id:
            return []

        reviews_url = f"{base_imdb_url}/title/{imdb_movie_id}/reviews"
        reviews_response = requests.get(reviews_url, headers=headers)
        reviews_response.raise_for_status()
        reviews_soup = BeautifulSoup(reviews_response.text, 'html.parser')

        review_card_parents = reviews_soup.find_all('div', attrs={'data-testid': 'review-card-parent'})
        for review_parent_div in review_card_parents:
            review_text_div = review_parent_div.find('div', class_='ipc-html-content-inner-div')
            if review_text_div:
                text = review_text_div.get_text(strip=True)
                if text:
                    reviews_found.append(text)

        pagination_key = None
        load_more_container = reviews_soup.find('div', class_='load-more-reviews')
        if load_more_container:
            load_more_button = load_more_container.find('button', class_='ipc-btn')
            if load_more_button and 'data-key' in load_more_button.attrs:
                pagination_key = load_more_button['data-key']

        while pagination_key:
            ajax_url = f"{base_imdb_url}/title/{imdb_movie_id}/reviews/_ajax?ref_=undefined&paginationKey={pagination_key}"
            time.sleep(random.uniform(1.5, 3.5))
            ajax_response = requests.get(ajax_url, headers=headers)
            ajax_response.raise_for_status()
            ajax_soup = BeautifulSoup(ajax_response.text, 'html.parser')
            new_review_card_parents = ajax_soup.find_all('div', attrs={'data-testid': 'review-card-parent'})

            if not new_review_card_parents:
                break

            for review_parent_div in new_review_card_parents:
                review_text_div = review_parent_div.find('div', class_='ipc-html-content-inner-div')
                if review_text_div:
                    text = review_text_div.get_text(strip=True)
                    if text:
                        reviews_found.append(text)

            next_load_more_container = ajax_soup.find('div', class_='load-more-reviews')
            if next_load_more_container:
                next_load_more_button = next_load_more_container.find('button', class_='ipc-btn')
                if next_load_more_button and 'data-key' in next_load_more_button.attrs:
                    pagination_key = next_load_more_button['data-key']
                else:
                    pagination_key = None
            else:
                pagination_key = None

    except Exception as e:
        print(f"Error while scraping IMDb reviews: {e}")

    if len(reviews_found) > 10:
        return random.sample(reviews_found, 10)
    return reviews_found



def predict_sentiment(reviews):
    cleaned_reviews = [clean_text(review) for review in reviews]
    transformed_reviews = transform_vectorizer.transform(cleaned_reviews)
    sentiment_labels = svc_model.predict(transformed_reviews)
    sentiment_labels = ['Positive' if label == 1 else 'Negative' for label in sentiment_labels]
    return sentiment_labels

def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)  # Remove special characters
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = remove_stopwords(text)
    text = perform_Lemmatization(text)
    return text


def remove_stopwords(text):
    stopwords_english = stopwords.words('english')
    new_text = ' '.join([word for word in text.split() if word not in stopwords_english])
    return new_text

def perform_Lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    new_text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return new_text

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={tmdb_api_key}&language=en-US"
    data = requests.get(url).json()
    poster_path = data['poster_path']
    return f"https://image.tmdb.org/t/p/w500/{poster_path}"

def recommend(movie):
    index = movies[movies['movie_title'] == movie].index[0]
    cv = CountVectorizer()
    vectors = cv.fit_transform(movies['comb'])
    similarity = cosine_similarity(vectors)  
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].movie_title)
    return recommended_movie_names, recommended_movie_posters

st.header('🎬 Movie Recommendation and Review Sentiment Analysis')

selected_movie = st.selectbox("🎞️ Select a movie", movies['movie_title'].values)

if st.button('🔍 Show Recommendation & Analyze Reviews'):
    st.subheader('📽️ Recommended Movies Based on Your Selection:')
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie)
    cols = st.columns(5)

    for i in range(5):
        with cols[i]:
            poster = recommended_movie_posters[i] if recommended_movie_posters[i] else "https://via.placeholder.com/150x225?text=No+Image"
            st.image(poster, use_container_width=True)  # ✅ FIX: no deprecated use_column_width
            st.caption(recommended_movie_names[i])

    st.markdown("---")
    st.header('🗣️ Sentiment Analysis from IMDb Reviews')

    movie_id = movies[movies['movie_title'] == selected_movie].iloc[0]['movie_id']
    imdb_id = get_imdb_id(movie_id)

    if imdb_id:
        reviews = scrape_imdb_reviews(imdb_id)
        if reviews:
            sentiments = predict_sentiment(reviews)

            st.markdown("### 🧾 Reviews and Sentiments")
            for review, sentiment in zip(reviews, sentiments):
                if sentiment == 'Positive':
                    sentiment_bg = '#155724'  # dark green
                    sentiment_text = '#ffffff'
                else:
                    sentiment_bg = '#721c24'  # dark red
                    sentiment_text = '#ffffff'

                st.markdown(f"""
                    <div style="background-color:{sentiment_bg}; color:{sentiment_text}; 
                                padding:10px; border-radius:6px; margin-bottom:5px; font-weight:bold;">
                        Sentiment: {sentiment}
                    </div>
                    <div style="background-color:#ffffff; color:#000000; 
                                padding:10px; border-radius:6px; border:1px solid #ccc; margin-bottom:20px;">
                        {review}
                    </div>
                """, unsafe_allow_html=True)

        else:
            st.warning("⚠️ No reviews found for this movie on IMDb.")
    else:
        st.error("❌ Failed to fetch IMDb ID for the selected movie.")
