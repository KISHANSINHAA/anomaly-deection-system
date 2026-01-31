def generate_comprehensive_nyc_taxi_data():
    """Generate 365+ days of realistic NYC taxi fare data for last year"""
    
    print("🔄 Generating comprehensive NYC Taxi dataset for LAST YEAR (2024-2025)...")
    
    # Generate data for last year (2024-2025)
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)  # Full year 2024
    
    # Generate daily data
    dates = []
    total_fares = []
    trip_counts = []
    avg_fares = []
    
    current_date = start_date
    
    while current_date <= end_date:
        dates.append(current_date)
        
        # Base fare amount (adjusted for 2024-2025 inflation)
        base_fare = random.uniform(50000, 85000)  # Higher base for recent years
        
        # Weekly patterns (higher weekends)
        day_of_week = current_date.weekday()
        if day_of_week >= 5:  # Weekend
            base_fare *= random.uniform(1.2, 1.5)
        
        # Monthly seasonal patterns
        month = current_date.month
        if month in [6, 7, 8]:  # Summer - higher tourism
            base_fare *= random.uniform(1.1, 1.3)
        elif month in [11, 12]:  # Holiday season
            base_fare *= random.uniform(1.15, 1.4)
        elif month in [1, 2]:  # Winter - lower activity
            base_fare *= random.uniform(0.8, 1.0)
        
        # Add realistic noise and occasional anomalies
        noise = random.uniform(-0.15, 0.15)
        fare = base_fare * (1 + noise)
        
        # Random anomalies (2-3% of data) - more realistic for recent data
        if random.random() < 0.025:
            anomaly_type = random.choice(['spike', 'drop', 'trend'])
            if anomaly_type == 'spike':
                fare *= random.uniform(1.8, 3.5)  # Moderate spikes for recent data
            elif anomaly_type == 'drop':
                fare *= random.uniform(0.4, 0.7)  # Moderate drops
            else:  # trend
                # Gradual trend over several days
                trend_factor = random.uniform(0.95, 1.05)
                for i in range(min(5, len(total_fares))):
                    if len(total_fares) > i:
                        total_fares[-(i+1)] *= trend_factor
        
        total_fares.append(max(0, fare))
        
        # Trip count (correlated with fare amounts)
        base_trips = random.randint(22000, 38000)  # Higher for recent years
        trip_multiplier = fare / base_fare
        trips = int(base_trips * trip_multiplier * random.uniform(0.8, 1.2))
        trip_counts.append(max(1000, trips))
        
        # Average fare per trip
        avg_fare = fare / trips if trips > 0 else random.uniform(4.5, 9.0)
        avg_fares.append(max(2.5, avg_fare))
        
        current_date += timedelta(days=1)
    
    # Create DataFrame
    data = pd.DataFrame({
        'date': dates,
        'total_fare': total_fares,
        'trip_count': trip_counts,
        'avg_fare': avg_fares
    })
    
    # Save to CSV
    output_path = "data/raw/nyc_taxi/last_year_fare_data.csv"
    data.to_csv(output_path, index=False)
    
    print(f"✅ Generated {len(data)} days of LAST YEAR data")
    print(f"📅 Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"📊 Total fare range: ${data['total_fare'].min():,.2f} - ${data['total_fare'].max():,.2f}")
    print(f"🚗 Trip count range: {data['trip_count'].min():,} - {data['trip_count'].max():,}")
    print(f"💰 Average fare range: ${data['avg_fare'].min():.2f} - ${data['avg_fare'].max():.2f}")
    print(f"💾 Data saved to: {output_path}")
    
    return data