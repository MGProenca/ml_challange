import pandas as pd
import numpy as np

def make_lags_single_column(sel_df, lags, lag_column, region_name=False):
    ''' Create lag features for a single region and lag column '''
    # TODO Raise error if sel_df has multiple regions
    lag_feats = pd.DataFrame(index=sel_df.index)
    for lag in lags:
        if region_name:
            lag_feats[f'{region_name}_{lag_column}_lag_{lag}'] = sel_df.groupby('Type')[lag_column].shift(lag)
        else:
            lag_feats[f'{lag_column}_lag_{lag}'] = sel_df.groupby('Type')[lag_column].shift(lag)
    return lag_feats

def make_region_lags(region_filt_df, region_name, columns_to_lag, lags ,region_name_in_col = False):
    lag_features = []
    for col_name in columns_to_lag:
        if region_name_in_col:
            lag_features.append(make_lags_single_column(region_filt_df, lags, col_name, region_name))
        else:
            lag_features.append(make_lags_single_column(region_filt_df, lags, col_name))
    return pd.concat(lag_features, axis=1)

def select_region(merge_df, region):
    sel_df = merge_df[merge_df['Region'] == region].reset_index(drop=True)
    return sel_df.drop(columns='Region')

def make_target_region_lags_df(merge_df, target_region, configs):
    region_filt_df = select_region(merge_df, target_region)

    columns_to_lag = merge_df.loc[:, merge_df.columns != configs['target_name']].select_dtypes(
        include=['number']).columns

    target_region_lags_df = make_region_lags(region_filt_df, target_region, columns_to_lag, configs['lags'])

    aux_regions_lags = []
    if configs['aux_regions']:
        for aux_region_name in configs['aux_regions']:
            if aux_region_name == target_region:
                continue
            
            aux_region_filt_df = select_region(merge_df, aux_region_name).reset_index()
            aux_regions_lags.append(
                make_region_lags(
                    aux_region_filt_df, 
                    aux_region_name, 
                    configs['aux_features'], 
                    configs['aux_lags'],
                    region_name_in_col=True))
        aux_regions_lags_df = pd.concat(aux_regions_lags, axis=1)
        target_region_lags_df = pd.concat([target_region_lags_df, aux_regions_lags_df], axis=1)

    return target_region_lags_df

def make_time_features(sel_df):
    time_feats = pd.DataFrame(index=sel_df.index)
    dates = sel_df['Date']
    time_feats['Year'] = dates.dt.year
    time_feats['MonthSin'] = np.sin(2 * np.pi * dates.dt.month/ 12)
    time_feats['MonthCos'] = np.cos(2 * np.pi * dates.dt.month/ 12)
    time_feats['Day'] = dates.dt.day
    time_feats['DayofWeekSin'] = np.sin(2 * np.pi * dates.dt.dayofweek/ 7)
    time_feats['DayofWeekCos'] = np.cos(2 * np.pi * dates.dt.dayofweek/ 7)
    time_feats["WeekofYearSin"] = np.sin(2 * np.pi * dates.dt.isocalendar().week / 52)
    time_feats["WeekofYearCos"] = np.cos(2 * np.pi * dates.dt.isocalendar().week / 52)
    time_feats["QuarterSin"] = np.sin(2 * np.pi * dates.dt.quarter / 4)
    time_feats["QuarterCos"] = np.cos(2 * np.pi * dates.dt.quarter / 4)
    time_feats['TimeIndex'] = ((dates - dates.min()).dt.days) // 7  # A simple trend feature

    return time_feats

def make_rolling(feat_df, window_sizes):
    for size in window_sizes:
        feat_df[f'rolling_{size}_mean'] = feat_df.groupby('Type')['AveragePrice_combined_lag_4'].transform(
            lambda x: x.rolling(size).mean())
        feat_df[f'rolling_{size}_mean'] = feat_df.groupby('Type')['AveragePrice_combined_lag_4'].transform(
            lambda x: x.rolling(size).std())
        feat_df[f'rolling_{size}_mean'] = feat_df.groupby('Type')['AveragePrice_combined_lag_4'].transform(
            lambda x: x.rolling(size).max())
        feat_df[f'rolling_{size}_mean'] = feat_df.groupby('Type')['AveragePrice_combined_lag_4'].transform(
            lambda x: x.rolling(size).min())
    return feat_df

def make_stage_2_data(merge_df, region, configs):
    lags_df = make_target_region_lags_df(merge_df, region, configs)
    region_filt_df = select_region(merge_df, region)
    feat_df = pd.concat([
        region_filt_df[['Date', 'Type', configs['target_name']]], 
        lags_df], axis=1)

    feat_df = pd.concat([feat_df, make_time_features(feat_df)], axis=1)
    feat_df = make_rolling(feat_df, configs['rolling_window_sizes'])
    feat_df = pd.get_dummies(feat_df, columns=['Type'], prefix='Type')

    feat_df = feat_df.loc[~feat_df['Date'].isnull(), :]
    dates = feat_df['Date']
    feat_df = feat_df.drop(columns='Date')
    
    X = feat_df[feat_df.columns.difference([configs['target_name']])]
    X = X.apply(pd.to_numeric, errors='coerce')
    
    y = feat_df[configs['target_name']]
    
    return X, y, dates