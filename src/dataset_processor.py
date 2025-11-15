"""
Dataset processing module for Clipzy
Processes raw datasets and extracts features for model training
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from sentence_transformers import SentenceTransformer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize NLTK components (will be downloaded if not present)
try:
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
except LookupError:
    logger.warning("NLTK stopwords not found. Run: python -m nltk.downloader stopwords")
    stop_words = set()

try:
    sia = SentimentIntensityAnalyzer()
except LookupError:
    logger.warning("NLTK VADER lexicon not found. Run: python -m nltk.downloader vader_lexicon")
    sia = None


class DatasetProcessor:
    """Processes raw datasets and extracts features for viral segment detection"""
    
    def __init__(self, embedding_model: str = None):
        """
        Initialize dataset processor
        
        Args:
            embedding_model: Name of sentence transformer model to use
        """
        self.embedding_model_name = embedding_model or config.EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        logger.info("Embedding model loaded successfully")
    
    def load_dataset(self, dataset_path: str = None) -> pd.DataFrame:
        """
        Load raw dataset from CSV file
        
        Args:
            dataset_path: Path to CSV file. If None, uses config default
            
        Returns:
            DataFrame with raw data
        """
        path = dataset_path or config.DATASET_PATH
        logger.info(f"Loading dataset from: {path}")
        
        if not Path(path).exists():
            raise FileNotFoundError(f"Dataset file not found: {path}. Please create the dataset first.")
        
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} records from dataset")
        return df
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\'"]', '', text)
        
        return text.strip()
    
    def extract_text_features(self, text: str) -> Dict:
        """
        Extract features from text segment
        
        Args:
            text: Text content of segment
            
        Returns:
            Dictionary of extracted features
        """
        text = self.clean_text(text)
        
        features = {
            'text_length': len(text),
            'word_count': len(word_tokenize(text)) if text else 0,
            'char_count': len(text),
        }
        
        # Sentiment analysis
        if sia:
            sentiment_scores = sia.polarity_scores(text)
            features['sentiment_positive'] = sentiment_scores['pos']
            features['sentiment_negative'] = sentiment_scores['neg']
            features['sentiment_neutral'] = sentiment_scores['neu']
            features['sentiment_compound'] = sentiment_scores['compound']
        else:
            features['sentiment_positive'] = 0.0
            features['sentiment_negative'] = 0.0
            features['sentiment_neutral'] = 1.0
            features['sentiment_compound'] = 0.0
        
        # Engagement indicators
        features['has_question'] = 1 if '?' in text else 0
        features['has_exclamation'] = 1 if '!' in text else 0
        features['question_count'] = text.count('?')
        features['exclamation_count'] = text.count('!')
        
        # Personal pronouns (engagement indicator)
        personal_pronouns = ['i', 'you', 'we', 'they', 'he', 'she', 'me', 'us', 'them', 'my', 'your', 'our']
        words = word_tokenize(text.lower()) if text else []
        features['personal_pronoun_count'] = sum(1 for word in words if word in personal_pronouns)
        
        # Keywords (non-stopwords)
        if text:
            words = [w.lower() for w in word_tokenize(text) if w.isalnum()]
            keywords = [w for w in words if w not in stop_words]
            features['keyword_count'] = len(keywords)
            features['keyword_density'] = len(keywords) / len(words) if words else 0
        else:
            features['keyword_count'] = 0
            features['keyword_density'] = 0
        
        return features
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Get text embedding using sentence transformer
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        text = self.clean_text(text)
        if not text:
            # Return zero vector if text is empty
            return np.zeros(self.embedding_model.get_sentence_embedding_dimension())
        
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        return embedding
    
    def process_dataset(self, dataset_path: str = None, save_path: str = None) -> pd.DataFrame:
        """
        Process entire dataset and extract all features
        
        Args:
            dataset_path: Path to raw dataset CSV
            save_path: Path to save processed dataset. If None, saves to processed directory
            
        Returns:
            DataFrame with processed features
        """
        logger.info("Starting dataset processing...")
        
        # Load raw dataset
        df = self.load_dataset(dataset_path)
        
        # Clean transcript column
        logger.info("Cleaning text data...")
        df['transcript_clean'] = df['transcript'].apply(self.clean_text)
        
        # Extract text features
        logger.info("Extracting text features...")
        text_features_list = []
        for idx, text in enumerate(df['transcript_clean']):
            if idx % 10 == 0:
                logger.info(f"Processing {idx+1}/{len(df)} segments...")
            text_features_list.append(self.extract_text_features(text))
        
        text_features_df = pd.DataFrame(text_features_list)
        
        # Extract embeddings
        logger.info("Extracting text embeddings (this may take a while)...")
        embeddings = []
        for idx, text in enumerate(df['transcript_clean']):
            if idx % 10 == 0:
                logger.info(f"Embedding {idx+1}/{len(df)} segments...")
            embeddings.append(self.get_text_embedding(text))
        
        # Add embedding columns
        embedding_dim = len(embeddings[0])
        for i in range(embedding_dim):
            df[f'embedding_{i}'] = [emb[i] for emb in embeddings]
        
        # Merge text features
        df = pd.concat([df, text_features_df], axis=1)
        
        # Calculate additional features
        logger.info("Calculating additional features...")
        if 'duration' in df.columns:
            df['words_per_second'] = df['word_count'] / df['duration'].clip(lower=0.1)
        else:
            df['words_per_second'] = 0
        
        # Ensure is_viral is binary
        if 'is_viral' in df.columns:
            df['is_viral'] = df['is_viral'].astype(int)
        
        logger.info(f"Dataset processing complete. Processed {len(df)} segments.")
        
        # Save processed dataset
        if save_path is None:
            save_path = config.PROCESSED_DATA_DIR / "processed_features.csv"
        else:
            save_path = Path(save_path)
        
        logger.info(f"Saving processed dataset to: {save_path}")
        df.to_csv(save_path, index=False)
        logger.info("Processed dataset saved successfully")
        
        return df
    
    def split_dataset(self, processed_df: pd.DataFrame, train_ratio: float = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train and test sets
        
        Args:
            processed_df: Processed DataFrame
            train_ratio: Ratio for training set (default from config)
            
        Returns:
            Tuple of (train_df, test_df)
        """
        ratio = train_ratio or config.TRAIN_TEST_SPLIT
        
        # Shuffle
        df_shuffled = processed_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split
        split_idx = int(len(df_shuffled) * ratio)
        train_df = df_shuffled[:split_idx]
        test_df = df_shuffled[split_idx:]
        
        logger.info(f"Dataset split: {len(train_df)} training, {len(test_df)} test samples")
        
        return train_df, test_df

