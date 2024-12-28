import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
import re
from datetime import datetime
import os

def load_and_clean_data(filename='Reviews.csv'):
    """
    Load and perform initial cleaning of the Amazon reviews dataset
    """
    # Read the data from the same directory
    df = pd.read_csv(filename)
    
    # Convert timestamps to datetime
    df['Time'] = pd.to_datetime(df['Time'], unit='s')
    
    # Calculate helpfulness ratio with handling division by zero
    df['helpfulness_ratio'] = np.where(
        df['HelpfulnessDenominator'] > 0,
        df['HelpfulnessNumerator'] / df['HelpfulnessDenominator'],
        0
    )
    
    # Convert float NaN to string empty
    df['Text'] = df['Text'].fillna('').astype(str)
    df['Summary'] = df['Summary'].fillna('').astype(str)
    
    # Basic text cleaning
    df['clean_text'] = df['Text'].apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()))
    df['clean_summary'] = df['Summary'].apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()))
    
    return df

def perform_sentiment_analysis(df):
    """
    Add sentiment analysis scores to the dataset
    """
    # Calculate sentiment scores
    df['sentiment_score'] = df['Text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['subjectivity_score'] = df['Text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    
    return df

def create_text_features(df):
    """
    Create features from review text
    """
    # Already converted to string in load_and_clean_data
    df['text_length'] = df['Text'].str.len()
    df['summary_length'] = df['Summary'].str.len()
    df['word_count'] = df['Text'].str.split().str.len()
    
    return df

def prepare_for_topic_modeling(df):
    """
    Prepare text data for LDA topic modeling
    """
    # Create TF-IDF features
    tfidf = TfidfVectorizer(max_features=1000,
                           stop_words='english')
    
    text_features = tfidf.fit_transform(df['clean_text'])
    feature_names = tfidf.get_feature_names_out()
    
    return text_features, feature_names

def save_processed_data(df, text_features, feature_names):
    """
    Save all processed data to files
    """
    # Create 'processed_data' directory if it doesn't exist
    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save main processed dataframe
    df.to_csv(f'processed_data/processed_reviews_{timestamp}.csv', index=False)
    
    # Save text features as sparse matrix
    import scipy.sparse as sparse
    sparse.save_npz(f'processed_data/text_features_{timestamp}.npz', text_features)
    
    # Save feature names
    pd.Series(feature_names).to_csv(f'processed_data/feature_names_{timestamp}.csv', index=False)
    
    print(f"Data saved in 'processed_data' directory with timestamp {timestamp}")
    return timestamp

def main_processing_pipeline(input_filename='Reviews.csv'):
    """
    Main processing pipeline that combines all steps
    """
    print("Starting data processing pipeline...")
    
    # Load and clean data
    print("Loading and cleaning data...")
    df = load_and_clean_data(input_filename)
    
    # Add sentiment analysis
    print("Performing sentiment analysis...")
    df = perform_sentiment_analysis(df)
    
    # Create text features
    print("Creating text features...")
    df = create_text_features(df)
    
    # Prepare for topic modeling
    print("Preparing text for topic modeling...")
    text_features, feature_names = prepare_for_topic_modeling(df)
    
    # Save all processed data
    print("Saving processed data...")
    timestamp = save_processed_data(df, text_features, feature_names)
    
    print("Processing complete!")
    return df, text_features, feature_names, timestamp

if __name__ == "__main__":
    # Run the pipeline
    df, text_features, feature_names, timestamp = main_processing_pipeline()