import pandas as pd
from geopy.distance import geodesic

def clean_uber_data(df, LONG_MIN, LONG_MAX, LAT_MIN, LAT_MAX):
    """
    Cleans the Uber dataset by removing rows with invalid data based on certain conditions:
    - Invalid or missing datetime
    - Coordinates outside the [-90, 90] range
    - Fare amount less than or equal to 0
    - Missing or NaN latitude/longitude values
    - Passenger count is 0 or greater than 10
    - Distance between pickup and dropoff less than or equal to 0
    
    Parameters:
        df (pd.DataFrame): The original Uber dataset.
        LONG_MIN, LONG_MAX (float): Longitude bounds for focusing on specific locations (e.g., NY).
        LAT_MIN, LAT_MAX (float): Latitude bounds for focusing on specific locations (e.g., NY).
    
    Returns:
        df_cleaned (pd.DataFrame): Cleaned DataFrame after removing invalid rows.
        dropped (pd.DataFrame): DataFrame containing the dropped invalid rows.
    """
    
    # Convert 'pickup_datetime' to datetime and extract the pickup hour
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
    df['pickup_hour'] = df['pickup_datetime'].dt.hour  # Extract hour from pickup_datetime
    
    # Create conditions for invalid data
    condition_invalid_datetime = df['pickup_datetime'].isna()
    
    # Filter out coordinates outside the [-90, 90] range (e.g., only focus on NY)
    condition_out_of_range = (
        (df['pickup_longitude'] < LONG_MIN) | (df['pickup_longitude'] > LONG_MAX) |
        (df['pickup_latitude'] < LAT_MIN) | (df['pickup_latitude'] > LAT_MAX) |
        (df['dropoff_longitude'] < LONG_MIN) | (df['dropoff_longitude'] > LONG_MAX) |
        (df['dropoff_latitude'] < LAT_MIN) | (df['dropoff_latitude'] > LAT_MAX)
    )
    
    # Filter out rows where fare_amount is less than or equal to 0
    condition_invalid_fare = df['fare_amount'] <= 0
    
    # Filter out rows with NaN latitude/longitude values
    condition_nan_coordinates = (
        df['pickup_longitude'].isna() |
        df['pickup_latitude'].isna() |
        df['dropoff_longitude'].isna() |
        df['dropoff_latitude'].isna()
    )
    
    # Filter out rows where passenger_count is 0 or greater than 10
    condition_zero_passenger = df['passenger_count'] <= 0
    condition_high_passenger_count = df['passenger_count'] > 10
    
    # Combine all conditions to filter out invalid rows
    initial_condition_combined = (
        condition_out_of_range |
        condition_invalid_fare |
        condition_nan_coordinates |
        condition_zero_passenger |
        condition_invalid_datetime |
        condition_high_passenger_count
    )
    
    # Extract rows that meet the invalid conditions into 'dropped' DataFrame
    dropped = df[initial_condition_combined].copy()
    
    # Remove invalid rows from the original DataFrame
    df_cleaned = df[~initial_condition_combined].copy()
    
    # Calculate 'distance_miles' using the geodesic formula on the cleaned DataFrame
    df_cleaned['distance_miles'] = df_cleaned.apply(lambda row: geodesic(
        (row['pickup_latitude'], row['pickup_longitude']),
        (row['dropoff_latitude'], row['dropoff_longitude'])
    ).miles, axis=1)
    
    # Create a condition to filter out rows where distance_miles is less than or equal to 0
    condition_invalid_distance = df_cleaned['distance_miles'] <= 0
    
    # Add invalid distance rows to the 'dropped' DataFrame
    dropped = pd.concat([dropped, df_cleaned[condition_invalid_distance]])
    
    # Remove rows with invalid distance from the cleaned DataFrame
    df_cleaned = df_cleaned[~condition_invalid_distance].copy()
    
    return df_cleaned, dropped
