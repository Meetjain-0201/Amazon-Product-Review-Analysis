"""
updated_topic_modeling.py

This module implements enhanced topic modeling functionality for Amazon product reviews using
Latent Dirichlet Allocation (LDA) with coherence optimization and sentiment integration.

Dependencies:
- numpy
- pandas
- sklearn
- gensim
- pyLDAvis
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
import pickle
import os
from datetime import datetime
import scipy.sparse as sparse
import glob
from sklearn.feature_extraction.text import CountVectorizer
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', message=".*OpenSSL.*")
warnings.filterwarnings('ignore', category=UserWarning)

class TopicModeler:
    def __init__(self, min_topics=5, max_topics=15, max_iter=20, random_state=42):
        """Initialize the TopicModeler with given parameters."""
        self.min_topics = min_topics
        self.max_topics = max_topics
        self.max_iter = max_iter
        self.random_state = random_state
        self.vectorizer = CountVectorizer(max_features=1000)
        self.best_model = None
        self.optimal_topics = None
        self.coherence_scores = {}
        
    def load_processed_data(self):
        """Load the most recently processed data files."""
        processed_files = glob.glob('processed_data/processed_reviews_*.csv')
        feature_files = glob.glob('processed_data/text_features_*.npz')
        names_files = glob.glob('processed_data/feature_names_*.csv')
        
        if not (processed_files and feature_files and names_files):
            raise FileNotFoundError("Processed data files not found. Run amazon_review_processor.py first.")
            
        latest_processed = max(processed_files)
        latest_features = max(feature_files)
        latest_names = max(names_files)
        
        df = pd.read_csv(latest_processed)
        text_features = sparse.load_npz(latest_features)
        feature_names = pd.read_csv(latest_names).iloc[:, 0].tolist()
        
        print(f"Loaded processed data from: {latest_processed}")
        return df, text_features, feature_names

    def compute_coherence_values(self, texts, dictionary, corpus, step=5):
        """Compute coherence scores for different numbers of topics."""
        for num_topics in range(self.min_topics, self.max_topics + 1, step):
            print(f"Computing coherence for {num_topics} topics...")
            lda_model = LdaModel(
                corpus=corpus,
                num_topics=num_topics,
                id2word=dictionary,
                random_state=self.random_state
            )
            
            coherence_model = CoherenceModel(
                model=lda_model,
                texts=texts,
                dictionary=dictionary,
                coherence='c_v'
            )
            
            self.coherence_scores[num_topics] = coherence_model.get_coherence()
        
        return self.coherence_scores

    def find_optimal_topics(self, df):
        """Find the optimal number of topics using coherence scores."""
        print("Finding optimal number of topics...")
        
        # Prepare texts for coherence calculation
        texts = [text.split() for text in df['clean_text']]
        dictionary = Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        
        # Compute coherence scores
        self.coherence_scores = self.compute_coherence_values(texts, dictionary, corpus)
        
        # Find optimal number of topics
        self.optimal_topics = max(self.coherence_scores.items(), key=lambda x: x[1])[0]
        
        print(f"Optimal number of topics: {self.optimal_topics}")
        return self.optimal_topics

    def fit(self, text_features):
        """Fit the LDA model with optimal number of topics."""
        if self.optimal_topics is None:
            print("Warning: Using default number of topics. Run find_optimal_topics first for better results.")
            self.optimal_topics = 10
            
        print(f"Fitting LDA model with {self.optimal_topics} topics...")
        self.lda_model = LatentDirichletAllocation(
            n_components=self.optimal_topics,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.document_topics = self.lda_model.fit_transform(text_features)
        return self.document_topics

    def get_top_terms_per_topic(self, feature_names, n_terms=10):
        """Extract the top terms for each topic."""
        topics = {}
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_terms_idx = topic.argsort()[:-n_terms-1:-1]
            top_terms = [feature_names[i] for i in top_terms_idx]
            topics[f"Topic {topic_idx+1}"] = top_terms
        return topics

    def analyze_topic_sentiment(self, df):
        """Analyze sentiment distribution across topics."""
        if 'sentiment_score' not in df.columns:
            print("Warning: No sentiment scores found in data")
            return None
            
        topic_sentiments = pd.DataFrame(self.document_topics)
        topic_sentiments['sentiment'] = df['sentiment_score']
        
        # Calculate average sentiment per topic
        topic_sentiment_analysis = pd.DataFrame({
            'topic': range(self.optimal_topics),
            'avg_sentiment': [
                np.average(df['sentiment_score'], weights=self.document_topics[:, i])
                for i in range(self.optimal_topics)
            ]
        })
        
        return topic_sentiment_analysis

    def save_results(self, df, text_features, feature_names):
        """Save topic modeling results and visualizations."""
        # Create directories if they don't exist
        if not os.path.exists('topic_models'):
            os.makedirs('topic_models')
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save document-topic distributions
        topic_distributions = pd.DataFrame(
            self.document_topics,
            columns=[f'Topic_{i+1}' for i in range(self.optimal_topics)]
        )
        topic_distributions.to_csv(f'topic_models/topic_distributions_{timestamp}.csv', index=False)
        
        # Save top terms for each topic
        top_terms = self.get_top_terms_per_topic(feature_names)
        with open(f'topic_models/top_terms_{timestamp}.txt', 'w') as f:
            for topic, terms in top_terms.items():
                f.write(f"{topic}:\n{', '.join(terms)}\n\n")
        
        # Save coherence scores
        if self.coherence_scores:
            pd.DataFrame({
                'n_topics': list(self.coherence_scores.keys()),
                'coherence_score': list(self.coherence_scores.values())
            }).to_csv(f'topic_models/coherence_scores_{timestamp}.csv', index=False)
        
        print(f"Results saved in topic_models directory with timestamp {timestamp}")

def main():
    """Main function to run enhanced topic modeling."""
    try:
        # Initialize topic modeler
        modeler = TopicModeler(min_topics=5, max_topics=15)
        
        # Load processed data
        df, text_features, feature_names = modeler.load_processed_data()
        
        # Find optimal number of topics
        optimal_topics = modeler.find_optimal_topics(df)
        
        # Fit model
        document_topics = modeler.fit(text_features)
        
        # Print top terms
        top_terms = modeler.get_top_terms_per_topic(feature_names)
        print("\nTop terms per topic:")
        for topic, terms in top_terms.items():
            print(f"\n{topic}:")
            print(", ".join(terms))
        
        # Save all results
        modeler.save_results(df, text_features, feature_names)
        
        print("\nEnhanced topic modeling completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    main()