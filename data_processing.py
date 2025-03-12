import pandas as pd
import shutil
import os
import kagglehub

def load_raw_data():
    # Download latest version
    path = kagglehub.dataset_download("neuromusic/avocado-prices")

    destination_folder = "./data"
    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Move all files from source to destination
    for filename in os.listdir(path):
        source_file = os.path.join(path, filename)
        destination_file = os.path.join(destination_folder, filename)
        shutil.move(source_file, destination_file)

    df = pd.read_csv('data/avocado.csv')
    return df

def preprocess_raw_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    # df = df.set_index('Date')
    df = df.drop(['Unnamed: 0', 'year'], axis=1)
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(' ', '')
    df.columns = df.columns.str[0].str.upper() + df.columns.str[1:]
    return df

def aggregate_types(group):
    total_volume = group['TotalVolume'].sum()
    weighted_avg = (group['AveragePrice'] * group['TotalVolume']).sum() / total_volume
    return pd.Series({
        'Date': group['Date'].iloc[0],
        'Region': group['Region'].iloc[0],
        'AveragePrice_combined': weighted_avg,
        'TotalVolume_combined': total_volume,
        '4046_combined': group['4046'].sum(),
        '4225_combined': group['4225'].sum(),
        '4770_combined': group['4770'].sum(),
        'TotalBags_combined': group['TotalBags'].sum(),
        'SmallBags_combined': group['SmallBags'].sum(),
        'LargeBags_combined': group['LargeBags'].sum(),
        'XLargeBags_combined': group['XLargeBags'].sum(),
    })

def group_by_region(df):
    combined_df = df.groupby(['Date', 'Region'])[df.columns].apply(aggregate_types).reset_index(drop=True)
    return combined_df

def pivot_and_merge_numerical_columns(df, grouped_df, target_name):
    numerical_columns = df.select_dtypes(include=['number']).columns
    pivot_df = df.pivot(index=['Date','Region'], columns='Type', values=numerical_columns).reset_index()
    pivot_df.columns = pivot_df.columns.map(lambda col: '_'.join(map(str, col)).strip('_'))
    merge_df = pd.merge(pivot_df, df[['Date', 'Type', 'Region']], on=['Date', 'Region'], how='left')
    merge_df = pd.merge(merge_df, grouped_df, on=['Date', 'Region'], how='left')
    merge_df = pd.merge(merge_df, df[['Region', 'Date', 'Type', target_name]], on=['Date', 'Region', 'Type'], how='left')
    return merge_df

def make_stage_1_data(configs):
    df = load_raw_data()
    df = preprocess_raw_data(df)
    grouped_df = group_by_region(df)
    merge_df = pivot_and_merge_numerical_columns(df, grouped_df, configs['target_name'])
    return merge_df
