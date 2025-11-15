"""
NLP Analyzer module for detecting viral segments
Trains and uses ML models to identify interesting/viral podcast segments
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NLPAnalyzer:
    """NLP model for detecting viral/interesting segments in podcast transcripts"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize NLP Analyzer
        
        Args:
            model_path: Path to saved model. If None, uses config default
        """
        self.model_path = Path(model_path or config.MODEL_PATH)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels from processed dataset
        
        Args:
            df: Processed DataFrame with features
            
        Returns:
            Tuple of (features, labels)
        """
        # Get embedding columns
        embedding_cols = [col for col in df.columns if col.startswith('embedding_')]
        
        # Get other feature columns (exclude metadata)
        exclude_cols = ['transcript', 'transcript_clean', 'start_time', 'end_time', 
                       'source_video', 'topic', 'is_viral', 'views', 'likes', 'shares']
        
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and col not in embedding_cols]
        
        # Combine all features
        all_feature_cols = embedding_cols + feature_cols
        self.feature_columns = all_feature_cols
        
        # Extract features
        X = df[all_feature_cols].values
        
        # Extract labels
        if 'is_viral' not in df.columns:
            raise ValueError("Dataset must contain 'is_viral' column for training")
        
        y = df['is_viral'].values
        
        logger.info(f"Prepared {len(all_feature_cols)} features for {len(X)} samples")
        logger.info(f"Viral segments: {np.sum(y)}, Non-viral: {len(y) - np.sum(y)}")
        
        return X, y
    
    def train(self, processed_dataset_path: str = None, save_model: bool = True) -> Dict:
        """
        Train the viral segment detection model
        
        Args:
            processed_dataset_path: Path to processed dataset CSV
            save_model: Whether to save the trained model
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Starting model training...")
        
        # Load processed dataset
        if processed_dataset_path:
            df = pd.read_csv(processed_dataset_path)
        else:
            processed_path = config.PROCESSED_DATA_DIR / "processed_features.csv"
            if not processed_path.exists():
                raise FileNotFoundError(
                    f"Processed dataset not found: {processed_path}. "
                    "Please run dataset_processor.py first."
                )
            df = pd.read_csv(processed_path)
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        logger.info("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        logger.info("Training RandomForest classifier...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate
        logger.info("Evaluating model...")
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_precision = precision_score(y_test, y_pred_test, zero_division=0)
        test_recall = recall_score(y_test, y_pred_test, zero_division=0)
        test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='f1')
        
        metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std()
        }
        
        logger.info("Training Metrics:")
        logger.info(f"  Train Accuracy: {train_accuracy:.3f}")
        logger.info(f"  Test Accuracy: {test_accuracy:.3f}")
        logger.info(f"  Test Precision: {test_precision:.3f}")
        logger.info(f"  Test Recall: {test_recall:.3f}")
        logger.info(f"  Test F1: {test_f1:.3f}")
        logger.info(f"  CV F1: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred_test))
        
        # Save model
        if save_model:
            self.save_model()
        
        return metrics
    
    def save_model(self, model_path: str = None):
        """
        Save trained model to disk
        
        Args:
            model_path: Path to save model. If None, uses config default
        """
        if not self.is_trained:
            raise ValueError("Model is not trained. Train the model first.")
        
        path = Path(model_path or self.model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and scaler together
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to: {path}")
    
    def load_model(self, model_path: str = None):
        """
        Load trained model from disk
        
        Args:
            model_path: Path to saved model. If None, uses config default
        """
        path = Path(model_path or self.model_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}. Train the model first.")
        
        logger.info(f"Loading model from: {path}")
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.is_trained = True
        
        logger.info("Model loaded successfully")
    
    def predict_viral_score(self, features: Dict) -> float:
        """
        Predict viral score for a segment based on extracted features
        
        Args:
            features: Dictionary of features (from dataset_processor)
            
        Returns:
            Viral score between 0 and 1
        """
        if not self.is_trained:
            if self.model_path.exists():
                self.load_model()
            else:
                raise ValueError("Model not trained. Train or load a model first.")
        
        # Convert features to array matching feature_columns
        feature_vector = []
        for col in self.feature_columns:
            if col in features:
                feature_vector.append(features[col])
            elif col.startswith('embedding_'):
                # Extract embedding value
                idx = int(col.split('_')[1])
                if 'embedding' in features:
                    feature_vector.append(features['embedding'][idx])
                else:
                    feature_vector.append(0.0)
            else:
                feature_vector.append(0.0)
        
        feature_array = np.array([feature_vector])
        
        # Scale
        feature_scaled = self.scaler.transform(feature_array)
        
        # Predict probability
        prob = self.model.predict_proba(feature_scaled)[0]
        viral_score = prob[1] if len(prob) > 1 else prob[0]
        
        return float(viral_score)
    
    def predict_segment(self, text: str, duration: float = None, 
                       dataset_processor=None) -> Dict:
        """
        Predict viral score for a text segment
        
        Args:
            text: Transcript text
            duration: Duration in seconds (optional)
            dataset_processor: DatasetProcessor instance (optional, will create if not provided)
            
        Returns:
            Dictionary with prediction and features
        """
        # Extract features
        if dataset_processor is None:
            from src.dataset_processor import DatasetProcessor
            dataset_processor = DatasetProcessor()
        
        # Extract text features
        text_features = dataset_processor.extract_text_features(text)
        
        # Get embedding
        embedding = dataset_processor.get_text_embedding(text)
        
        # Add embedding to features
        for i, emb_val in enumerate(embedding):
            text_features[f'embedding_{i}'] = emb_val
        
        # Add duration if provided
        if duration:
            text_features['duration'] = duration
            text_features['words_per_second'] = text_features['word_count'] / max(duration, 0.1)
        
        # Predict score
        viral_score = self.predict_viral_score(text_features)
        
        return {
            'viral_score': viral_score,
            'is_viral': viral_score >= config.MIN_SEGMENT_SCORE,
            'features': text_features
        }

