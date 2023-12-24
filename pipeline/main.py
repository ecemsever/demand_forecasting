import pandas as pd
import numpy as np
import os
import pickle
import time
import xgboost as xgb


if os.path.exists("8_hour_predictions.csv"):
     os.remove("8_hour_predictions.csv")
if os.path.exists("168_hour_predictions.csv"):
     os.remove("168_hour_predictions.csv")

special_holidays = [
    "01-01",  # New Year's Day
    "02-01",  # New Year's Day (additional day)
    "26-01",  # Australia Day
    "13-03",  # Labour Day
    "07-04",  # Good Friday
    "08-04",  # Easter Saturday
    # Since you did not provide a date for "Easter Sunday", I'm omitting it.
    "10-04",  # Easter Monday
    "25-04",  # Anzac Day
    "12-06",  # King's Birthday
    # Omitting the AFL Grand Final date since it's TBC.
    "07-11",  # Melbourne Cup
    "25-12",  # Christmas Day
    "26-12"  # Boxing Day
]


# Helper function to determine the season based on the month
def get_season(month):
    if (12 <= month) or (month <= 2):
        return 'summer'
    elif 3 <= month <= 5:
        return 'autumn'
    elif 6 <= month <= 8:
        return 'winter'
    elif 9 <= month <= 11:
        return 'spring'

def periodic_encoding(x, max_val):
    # Calculate sine and cosine values and explicitly cast them as float
    sin = np.sin(2 * np.pi * x / max_val).astype('float64')
    cos = np.cos(2 * np.pi * x / max_val).astype('float64')
    return sin, cos

def create_forecasting_dataset(df, n_lag, n_next, special_holidays, 
                               day_embedding='dummy', week_embedding='dummy', season_embedding='dummy'):
    # Convert valid_start to datetime and extract components
    df['valid_start'] = pd.to_datetime(df['valid_start'])
    df['hour'] = df['valid_start'].dt.hour
    df['day'] = df['valid_start'].dt.day
    df['month'] = df['valid_start'].dt.month
    df['year'] = df['valid_start'].dt.year
    df['day_of_week'] = df['valid_start'].dt.dayofweek
    df['week_of_year'] = df['valid_start'].dt.isocalendar().week.astype('float64')
    df['season'] = df['month'].apply(get_season)
    
    # Identify special holidays
    df['date_str'] = df['valid_start'].dt.strftime('%d-%m')
    df['is_special_day'] = df['date_str'].isin(special_holidays).astype(int)
    
    # Periodic or Dummy Encoding
    if day_embedding == 'periodic':
        day_sin, day_cos = periodic_encoding(df['day_of_week'], 7)
        df['day_sin'] = day_sin
        df['day_cos'] = day_cos
    else:
        df = pd.get_dummies(df, columns=['day_of_week'])

    if week_embedding == 'periodic':
        # Encode the week_of_year column
        week_sin, week_cos = periodic_encoding(df['week_of_year'], 52)
        
        # Assign the encoded values ensuring they're in the correct float format
        df['week_sin'] = week_sin.astype('float64')
        df['week_cos'] = week_cos.astype('float64')
    else:
        # If not 'periodic', apply dummy encoding
        df = pd.get_dummies(df, columns=['week_of_year'])

    if season_embedding == 'periodic':
        seasons_map = {'spring': 0, 'summer': 1, 'autumn': 2, 'winter': 3}
        df['season'] = df['season'].map(seasons_map)
        season_sin, season_cos = periodic_encoding(df['season'], 4)
        df['season_sin'] = season_sin
        df['season_cos'] = season_cos
    else:
        df = pd.get_dummies(df, columns=['season'])

    # Create lagged demand values
    for lag in range(1, n_lag + 1):
        df[f'lag_{lag}_hours'] = df['total_demand'].shift(lag)

    # Create future demand values
    future_demands = [df['total_demand'].shift(-i) for i in range(n_next)]
    df_future = pd.concat(future_demands, axis=1)
    
    # Ensure no NaN values in future demands and lagged features
    mask = df_future.notna().all(axis=1) & df[[f'lag_{lag}_hours' for lag in range(1, n_lag + 1)]].notna().all(axis=1)
    df = df.loc[mask]
    df_future = df_future.loc[mask]

    # Drop columns not needed for features, except for 'valid_start'
    columns_to_drop = ['year', 'month', 'day', 'hour', 'date_str', 'total_demand']
    if 'season_mapped' in df.columns:
        columns_to_drop.append('season_mapped')
    df.drop(columns=columns_to_drop, axis=1, inplace=True)
    
    # Convert object columns to float
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(float)
    
    # Ensure all floating point numbers are in the correct format
    float_cols = df.select_dtypes(include=['float64', 'float32']).columns
    df[float_cols] = df[float_cols].astype('float64')

    # X includes 'valid_start' and other features, Y is future demands
    X = df
    Y = df_future.values
    # Ensure that the features (X) are returned as a DataFrame
    X = pd.DataFrame(X)
    # The future values (Y) should also be a DataFrame if you're going to use .iloc on it
    Y = pd.DataFrame(Y)

    return X, Y

# Load models
with open('../model/xgboost_model.pkl', 'rb') as file:
    model_8 = pickle.load(file)

with open('../model/xgboost_model_168_v2.pkl', 'rb') as file:
    model_168 = pickle.load(file)
# Load datasets
train_df = pd.read_csv('train_dataset.csv')
test_df = pd.read_csv('test_dataset.csv')


def update_prediction_files(index, n_lag, train_df, test_df, model_8, model_168, filename_8, filename_168):
    # Include all of the training data and the part of the test data up to the current index
    if index < n_lag:
        df_temp = pd.concat([train_df, test_df.iloc[:index]])
    else:
        df_temp = pd.concat([train_df, test_df.iloc[index - n_lag:index]])

    # Process the combined data for 8-hour prediction
    X_8, Y_8 = create_forecasting_dataset(df_temp, 168, 8, special_holidays, 'dummy', 'dummy', 'periodic')
    predictions_8 = model_8.predict(X_8.iloc[-1:].drop(columns=['valid_start']))

    # Process the combined data for 168-hour prediction
    X_168, Y_168 = create_forecasting_dataset(df_temp, 168*6, 168, special_holidays, 'dummy', 'dummy', 'periodic')
    predictions_168 = model_168.predict(X_168.iloc[-1:].drop(columns=['valid_start']))

    # Extract the actual demand for the current hour from test_df
    # actual_demand = df_temp.iloc[-1,5]
    actual_demand_8 = (np.array(Y_8.iloc[-1:]).reshape(-1))[0]
    actual_demand_168 = (np.array(Y_168.iloc[-1:]).reshape(-1))[0]
    # print()
    
    # Write to CSV
    timestamp = test_df.iloc[index]['valid_start']  # Timestamp from the current test row
    
    # Writing for 8-hour predictions
    with open(filename_8, 'a') as f_8:
        header_8 = 'timestamp,' + ','.join(['total_demand'] + [f'prediction_{i+1}' for i in range(8)])
        f_8.write(f"{header_8}\n") if index == 0 else None
        row_8 = f"{timestamp},{actual_demand_8},{','.join(map(str, predictions_8.flatten()))}"
        f_8.write(f"{row_8}\n")

    # Writing for 168-hour predictions
    with open(filename_168, 'a') as f_168:
        header_168 = 'timestamp,' + ','.join(['total_demand'] + [f'prediction_{i+1}' for i in range(168)])
        f_168.write(f"{header_168}\n") if index == 0 else None
        row_168 = f"{timestamp},{actual_demand_168},{','.join(map(str, predictions_168.flatten()))}"
        f_168.write(f"{row_168}\n")


# Function to run the update process at specified intervals
def run_update_process(interval, train_df, test_df, model_8, model_168, filename_8, filename_168):
    for index in range(len(test_df)):
        update_prediction_files(index, 168*6, train_df, test_df, model_8, model_168, filename_8, filename_168)
        time.sleep(interval)  # Wait for the specified interval
        


# Start the update process
run_update_process(0, train_df, test_df, model_8, model_168, '8_hour_predictions.csv', '168_hour_predictions.csv')