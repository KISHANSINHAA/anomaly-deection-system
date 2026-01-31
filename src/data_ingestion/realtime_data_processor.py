"""
Real-Time NYC Taxi Data Pipeline
Fetches latest data and prepares for ongoing anomaly detection
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import os
import time
from tqdm import tqdm

class RealTimeDataPipeline:
    def __init__(self, start_date='2024-01-01'):
        """
        Initialize real-time data pipeline
        
        Args:
            start_date: Start fetching data from this date (default: 2024-01-01)
        """
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.base_url = "https://data.cityofnewyork.us/resource/tlc-trip-record-data/yellow-tripdata"
        self.data_dir = 'data/nyc_taxi_realtime'
        os.makedirs(self.data_dir, exist_ok=True)
        
    def fetch_recent_data(self, days_back=30):
        """
        Fetch recent NYC Taxi data
        
        Args:
            days_back: Number of recent days to fetch
            
        Returns:
            DataFrame with recent trip data
        """
        print(f"Fetching NYC Taxi data for last {days_back} days...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        all_data = []
        
        # Fetch data day by day
        current_date = start_date
        with tqdm(total=days_back, desc="Fetching days") as pbar:
            while current_date <= end_date:
                # Format dates for API
                date_str = current_date.strftime('%Y-%m-%d')
                start_timestamp = f"{date_str}T00:00:00"
                end_timestamp = f"{date_str}T23:59:59"
                
                # API query
                query = f"?$where=pickup_datetime between '{start_timestamp}' and '{end_timestamp}'&$limit=50000"
                url = self.base_url + query
                
                try:
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        data = response.json()
                        if data:  # Only add if data exists
                            all_data.extend(data)
                            print(f"  {date_str}: {len(data)} records")
                        else:
                            print(f"  {date_str}: No data")
                    else:
                        print(f"  {date_str}: API error {response.status_code}")
                        
                except Exception as e:
                    print(f"  {date_str}: Error - {e}")
                
                current_date += timedelta(days=1)
                pbar.update(1)
                time.sleep(0.5)  # Be respectful to API
        
        if not all_data:
            print("No recent data found. Generating synthetic data for demo...")
            return self.generate_synthetic_recent_data(days_back)
        
        return all_data
    
    def generate_synthetic_recent_data(self, days_back=30):
        """Generate synthetic recent data when API fails"""
        print("Generating synthetic recent data...")
        
        end_date = datetime.now()
        dates = pd.date_range(end=end_date, periods=days_back, freq='D')
        
        data = []
        for date in dates:
            # Base values with current trends
            base_trips = 25000 + np.random.normal(0, 3000)
            base_fare = 12.5 + np.random.normal(0, 2.0)
            
            # Add some realistic patterns
            if date.weekday() >= 5:  # Weekend
                base_trips *= 1.2
                base_fare *= 1.1
            
            # Add some anomalies (2% chance)
            is_anomaly = 0
            if np.random.random() < 0.02:
                multiplier = np.random.choice([0, 5, 8])  # Drop to zero or spike
                if multiplier == 0:
                    base_trips = 0
                    base_fare = 0
                else:
                    base_trips *= multiplier
                    base_fare *= multiplier * 0.8
                is_anomaly = 1
            
            data.append({
                'pickup_datetime': date.strftime('%Y-%m-%d'),
                'total_fare': max(0, base_trips * base_fare),
                'trip_count': max(0, int(base_trips)),
                'avg_fare': max(0, base_fare),
                'is_anomaly': is_anomaly
            })
        
        return pd.DataFrame(data)
    
    def process_realtime_data(self, raw_data):
        """
        Process raw API data into time-series format
        
        Args:
            raw_data: Raw data from API or synthetic data
            
        Returns:
            Processed DataFrame ready for anomaly detection
        """
        if isinstance(raw_data, list):
            # Real API data
            df = pd.DataFrame(raw_data)
            if df.empty:
                return pd.DataFrame()
                
            # Convert datetime columns
            df['pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
            df['dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
            
            # Extract date for aggregation
            df['date'] = df['pickup_datetime'].dt.date
            
            # Aggregate by day
            daily_stats = df.groupby('date').agg({
                'fare_amount': ['sum', 'count', 'mean'],
                'tip_amount': 'sum',
                'trip_distance': 'sum'
            }).reset_index()
            
            # Flatten column names
            daily_stats.columns = ['date', 'total_fare', 'trip_count', 'avg_fare', 'total_tip', 'total_distance']
            
        else:
            # Synthetic DataFrame
            daily_stats = raw_data.copy()
            daily_stats['date'] = pd.to_datetime(daily_stats['pickup_datetime']).dt.date
            daily_stats = daily_stats[['date', 'total_fare', 'trip_count', 'avg_fare']]
        
        # Sort by date
        daily_stats = daily_stats.sort_values('date').reset_index(drop=True)
        
        return daily_stats
    
    def save_realtime_data(self, df, filename='recent_data.csv'):
        """Save processed data"""
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        print(f"Shape: {df.shape}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        return filepath
    
    def create_future_dates(self, days_ahead=7):
        """
        Create future dates for prediction
        
        Args:
            days_ahead: Number of future days to create
            
        Returns:
            DataFrame with future dates
        """
        last_date = datetime.now().date()
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=days_ahead,
            freq='D'
        )
        
        future_df = pd.DataFrame({
            'date': future_dates,
            'total_fare': np.nan,  # Will be predicted
            'trip_count': np.nan,
            'avg_fare': np.nan,
            'is_future': True
        })
        
        return future_df
    
    def run_pipeline(self, days_back=30, days_ahead=7):
        """
        Run complete real-time pipeline
        
        Args:
            days_back: How many recent days to fetch
            days_ahead: How many future days to prepare
            
        Returns:
            Dictionary with historical and future data
        """
        print("=" * 60)
        print("REAL-TIME NYC TAXI DATA PIPELINE")
        print("=" * 60)
        
        # 1. Fetch recent data
        print("\n[1/4] Fetching recent data...")
        raw_data = self.fetch_recent_data(days_back)
        
        # 2. Process data
        print("\n[2/4] Processing data...")
        if isinstance(raw_data, list) and raw_data:
            processed_df = self.process_realtime_data(raw_data)
        else:
            processed_df = self.process_realtime_data(raw_data)
        
        # 3. Save historical data
        print("\n[3/4] Saving historical data...")
        hist_path = self.save_realtime_data(processed_df, 'historical_data.csv')
        
        # 4. Create future dates
        print("\n[4/4] Creating future dates...")
        future_df = self.create_future_dates(days_ahead)
        future_path = self.save_realtime_data(future_df, 'future_dates.csv')
        
        print(f"\nPipeline completed!")
        print(f"Historical data: {hist_path}")
        print(f"Future dates: {future_path}")
        print(f"Ready for real-time anomaly detection")
        
        return {
            'historical': processed_df,
            'future': future_df,
            'historical_path': hist_path,
            'future_path': future_path
        }

def main():
    # Run pipeline for last 60 days + next 14 days
    pipeline = RealTimeDataPipeline(start_date='2024-01-01')
    results = pipeline.run_pipeline(days_back=60, days_ahead=14)
    
    # Show sample
    print("\nRecent data sample:")
    print(results['historical'].tail())
    
    print("\nFuture dates:")
    print(results['future'])

if __name__ == "__main__":
    main()