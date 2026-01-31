"""
NYC Taxi Data Ingestion Module
Handles fetching, validating, and preprocessing NYC Taxi time-series data
"""
import pandas as pd
import numpy as np
import os
from typing import Optional, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class NYCTaxiDataIngestor:
    """Handles NYC Taxi data ingestion and preprocessing"""
    
    def __init__(self, raw_data_path: str = "data/raw/nyc_taxi/fare_per_day.csv"):
        """
        Initialize data ingestor
        
        Args:
            raw_data_path: Path to raw NYC Taxi data file
        """
        self.raw_data_path = raw_data_path
        self.data = None
        self.is_loaded = False
    
    def load_data(self) -> pd.DataFrame:
        """
        Load raw NYC Taxi data
        
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data format is invalid
        """
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Raw data file not found: {self.raw_data_path}")
        
        try:
            # Load the data
            self.data = pd.read_csv(self.raw_data_path)
            logger.info(f"Loaded {len(self.data)} records from {self.raw_data_path}")
            
            # Validate data structure
            self._validate_data_structure()
            
            # Convert date column to datetime
            self._process_dates()
            
            self.is_loaded = True
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise ValueError(f"Failed to load data: {e}")
    
    def _validate_data_structure(self):
        """Validate the data structure and required columns"""
        required_columns = ['date', 'total_fare']
        
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for reasonable data ranges
        if self.data['total_fare'].min() < 0:
            raise ValueError("Negative fare values detected")
        
        logger.info("Data structure validation passed")
    
    def _process_dates(self):
        """Process and validate date column"""
        # Convert date column to datetime
        self.data['date'] = pd.to_datetime(self.data['date'])
        
        # Sort by date
        self.data = self.data.sort_values('date').reset_index(drop=True)
        
        # Check for missing dates
        date_range = pd.date_range(
            start=self.data['date'].min(), 
            end=self.data['date'].max(), 
            freq='D'
        )
        
        missing_dates = set(date_range) - set(self.data['date'])
        if missing_dates:
            logger.warning(f"Missing {len(missing_dates)} dates in the dataset")
    
    def get_data_summary(self) -> dict:
        """
        Get summary statistics of the loaded data
        
        Returns:
            Dictionary with data summary
        """
        if not self.is_loaded:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        return {
            'total_records': len(self.data),
            'date_range': {
                'start': self.data['date'].min(),
                'end': self.data['date'].max()
            },
            'fare_stats': {
                'min': self.data['total_fare'].min(),
                'max': self.data['total_fare'].max(),
                'mean': self.data['total_fare'].mean(),
                'std': self.data['total_fare'].std()
            },
            'missing_values': self.data.isnull().sum().to_dict()
        }
    
    def get_time_series_data(self, 
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get time series data for a specific date range
        
        Args:
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            
        Returns:
            Filtered DataFrame
        """
        if not self.is_loaded:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        filtered_data = self.data.copy()
        
        if start_date:
            start_dt = pd.to_datetime(start_date)
            filtered_data = filtered_data[filtered_data['date'] >= start_dt]
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            filtered_data = filtered_data[filtered_data['date'] <= end_dt]
        
        return filtered_data.reset_index(drop=True)
    
    def save_processed_data(self, 
                          output_path: str = "data/processed/nyc_taxi/preprocessed_data.csv",
                          features: Optional[list] = None) -> str:
        """
        Save processed data to file
        
        Args:
            output_path: Path to save processed data
            features: List of features to include (default: all numeric columns)
            
        Returns:
            Path where data was saved
        """
        if not self.is_loaded:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Create features if not specified
        if features is None:
            features = ['total_fare']
        
        # Select relevant columns
        processed_data = self.data[['date'] + features].copy()
        
        # Add temporal features
        processed_data = self._add_temporal_features(processed_data)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to file
        processed_data.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")
        
        return output_path
    
    def _add_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features to the data
        
        Args:
            data: Input DataFrame with date column
            
        Returns:
            DataFrame with additional temporal features
        """
        data = data.copy()
        
        # Extract temporal components
        data['year'] = data['date'].dt.year
        data['month'] = data['date'].dt.month
        data['day'] = data['date'].dt.day
        data['day_of_week'] = data['date'].dt.dayofweek
        data['day_of_year'] = data['date'].dt.dayofyear
        data['week_of_year'] = data['date'].dt.isocalendar().week
        
        # Cyclical encoding for seasonal patterns
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        data['day_sin'] = np.sin(2 * np.pi * data['day'] / 31)
        data['day_cos'] = np.cos(2 * np.pi * data['day'] / 31)
        data['dow_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['dow_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        
        return data
    
    def get_data_for_modeling(self, 
                            features: Optional[list] = None,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Get data in format suitable for modeling
        
        Args:
            features: List of feature columns to use
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            Tuple of (feature_array, metadata_dataframe)
        """
        if not self.is_loaded:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Get filtered data
        filtered_data = self.get_time_series_data(start_date, end_date)
        
        # Default features if not specified
        if features is None:
            features = ['total_fare']
        
        # Add temporal features
        processed_data = self._add_temporal_features(filtered_data)
        
        # Select all numeric features
        numeric_features = processed_data.select_dtypes(include=[np.number]).columns.tolist()
        feature_data = processed_data[numeric_features].values
        
        return feature_data, processed_data


def load_nyc_taxi_data(raw_path: str = "data/raw/nyc_taxi/fare_per_day.csv",
                      processed_path: str = "data/processed/nyc_taxi/preprocessed_data.csv",
                      save_processed: bool = True) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Convenience function to load and preprocess NYC Taxi data
    
    Args:
        raw_path: Path to raw data
        processed_path: Path to save processed data
        save_processed: Whether to save processed data
        
    Returns:
        Tuple of (feature_array, metadata_dataframe)
    """
    ingestor = NYCTaxiDataIngestor(raw_path)
    ingestor.load_data()
    
    # Get data for modeling
    feature_data, metadata = ingestor.get_data_for_modeling()
    
    # Save processed data if requested
    if save_processed:
        ingestor.save_processed_data(processed_path)
    
    return feature_data, metadata


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Load data
        ingestor = NYCTaxiDataIngestor()
        data = ingestor.load_data()
        
        # Print summary
        summary = ingestor.get_data_summary()
        print("Data Summary:")
        print(f"  Records: {summary['total_records']}")
        print(f"  Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        print(f"  Fare Range: ${summary['fare_stats']['min']:.2f} to ${summary['fare_stats']['max']:.2f}")
        
        # Save processed data
        ingestor.save_processed_data()
        print("Processed data saved successfully")
        
    except Exception as e:
        print(f"Error: {e}")