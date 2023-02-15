# Import required packages
from azureml.core import Run, Workspace, Datastore, Dataset
from azureml.data.datapath import DataPath
import pandas as pd
import os
import argparse
# from sklearn import preprocessing
import numpy as np

#Parse input arguments
parser = argparse.ArgumentParser("Get tabular data from attached datastore, split into separate files, and register as file dataset in AML workspace")
parser.add_argument('--train_dataset', dest='train_dataset', required=True)
parser.add_argument('--forecast_dataset', dest='forecast_dataset', required=True)
parser.add_argument('--source_dataset_name', type=str, required=True)
parser.add_argument('--group_column_names', type=str, required=True)
parser.add_argument('--timestamp_column', type=str, required=True)

args, _ = parser.parse_known_args()
train_dataset = args.train_dataset
forecast_dataset = args.forecast_dataset
source_dataset_name = args.source_dataset_name
group_column_names = args.group_column_names.split(';')
timestamp_column = args.timestamp_column

#Get current run
current_run = Run.get_context()

#Get associated AML workspace
ws = current_run.experiment.workspace

#Connect to default Blob Store
ds = ws.get_default_datastore()

# Make directory on mounted storage for output dataset
os.makedirs(train_dataset, exist_ok=True)
os.makedirs(forecast_dataset, exist_ok=True)

#Read dataset from AML Datastore
source_dataset = Dataset.get_by_name(ws, name=source_dataset_name)
source_df = source_dataset.to_pandas_dataframe()

# Custom transformations
source_df['Segment'] = source_df['Segment'].astype(str)
source_df.rename(columns = {' Bookings ':'Bookings'}, inplace = True)
source_df['Bookings'] = source_df['Bookings'].replace('[\$,]', '', regex=True)
source_df['Bookings'] = source_df['Bookings'].replace(" -   ", "0")
source_df['Bookings'] = pd.to_numeric(source_df['Bookings'])
source_df['Booking Period'] = pd.to_datetime(source_df['Booking Period'], format='%Y%m')
source_df

# Helper function to add X number of months to a time-series, then split into two subsets
# for training and forecasting
def create_train_forecast_df(filtered_df, months_to_add=0, months_for_forecasting=4, timestamp_column='Booking Period', target_column='Bookings'):
    filtered_df = filtered_df.sort_values(by=timestamp_column)
    min_date = min(filtered_df[timestamp_column])
    max_date = max(filtered_df[timestamp_column])
    future_date = max_date + np.timedelta64(months_to_add, 'M')
    future_row = filtered_df.iloc[0]
    future_row[timestamp_column] = future_date
    future_row[target_column]=0
    filtered_df = filtered_df.append(future_row)
    filtered_df[timestamp_column] = filtered_df[timestamp_column]
    filtered_df.set_index(timestamp_column, inplace=True)
    final_df = filtered_df.resample('1MS').mean()
    final_df = final_df.reset_index()
    final_df = final_df.fillna(0)
    final_df['Segment'] = filtered_df.iloc[0]['Segment']
    final_df['Customer ID'] = filtered_df.iloc[0]['Customer ID']
    new_max = max(final_df[timestamp_column]) - np.timedelta64(months_for_forecasting, 'M')
    train_df = final_df[:-months_for_forecasting]
    forecast_df = final_df.iloc[-months_for_forecasting:]

    return train_df, forecast_df

# Save all training/forecasting subsets for model training/forecasting
grouped_dfs = source_df.groupby(group_column_names)
for idx, new_df in grouped_dfs:
    train_filename = "_".join([str(x) for x in list(idx)]) + "_train.csv"
    forecast_filename = "_".join([str(x) for x in list(idx)]) + "_forecast.csv"
    train_df, forecast_df = create_train_forecast_df(new_df, 12, 12, 'Booking Period', 'Bookings')
    
    train_df.to_csv(os.path.join(train_dataset, train_filename), index=False)
    forecast_df.to_csv(os.path.join(forecast_dataset, forecast_filename), index=False)