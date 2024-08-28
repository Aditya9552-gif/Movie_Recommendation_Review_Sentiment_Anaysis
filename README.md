# Movie Recommendation and Review Sentiment Analysis

This repository contains the code and resources for a movie recommendation and review sentiment analysis project. The project includes a recommendation system that suggests movies based on user preferences and a sentiment analysis tool that analyzes movie reviews to determine the sentiment (positive, negative, neutral).

The project has been deployed and can be accessed online at [Movie Recommendation and Review Sentiment Analysis](https://movie-recommendation-review-sentiment.onrender.com/).

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Preprocessing](#preprocessing)
- [Deployment](#deployment)

## Project Overview
This project combines two main functionalities:
1. **Movie Recommendation System:** Provides movie recommendations based on user inputs or preferences.
2. **Review Sentiment Analysis:** Analyzes user-submitted movie reviews to determine the sentiment (positive, negative, or neutral).

The project utilizes machine learning models for sentiment analysis and collaborative filtering techniques for movie recommendations.

## Features
- **Personalized Movie Recommendations:** Suggests movies based on user preferences.
- **Review Sentiment Analysis:** Analyzes the sentiment of user-submitted reviews.
- **Interactive Web Application:** Users can interact with the system through a web interface.

## Installation

To run this project locally, follow the steps below:

1. Clone the repository:
   ```bash
   git clone https://github.com/Aditya9552-gif/Movie_Recommendation_Review_Sentiment_Anaysis.git
   cd movierecommendation-sentiment

2. Create a virtual environment:
    python -m venv env

3. Activate the virtual environment:
     .\env\Scripts\activate

4. Install the required packages:
     pip install -r requirements.txt

5. Navigate to the final_app.py file and run it to start the local server:
     python app.py


     
## Usage
Movie Recommendation
*  Input your movie preferences or choose from the available options to receive personalized movie recommendations.

Sentiment Analysis
*  Submit a movie review, and the system will analyze and provide feedback on whether the review is positive, negative, or neutral.

Preprocessing
*  The preprocessing steps for both recommendation and sentiment analysis are divided into several notebooks:
*  preprocessing_1.ipynb: Handles extraction of movie details lke cast, genre, title etc for the "The Movies Dataset" avilable at kaggle.
*  preprocessing_2.ipynb: (Similar to preprocessing_1 movie details are extracted from "TMDB 5000 Movie Dataset" which is also available at kaggle)
*  preprocessing_3.ipynb: in this notebook movie detailed for years 2016 to 2023 are scraped from wikipedia.
*  preprocessing_4.ipynb: Final preprocessing and data cleaning is done in this notebook.
*  Movie_review_sentiment_analysis.ipynb: Sentiment Analysis is done on "IMDB Dataset of 50K Movie Reviews" Dataset avilable at kaggle.
*  Recommend.ipynb: Developed recommended system based on the extracted data from preprocessing files.
  
## Deployment
The project is deployed using Render. You can access the live version of the application [here](https://movie-recommendation-review-sentiment.onrender.com/).

To deploy the project yourself, follow these steps:

1. Create an account on Render (if you don't have one).
2. Create a new web service and link it to your GitHub repository.
3. Set up the environment variables as needed.
4. Deploy the service and access the provided URL.
