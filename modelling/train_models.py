import matplotlib.pyplot as plt
import os
import xgboost as xgb
import mlflow
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import time
from sklearn.model_selection import train_test_split
import json
import warnings
from dotenv import load_dotenv
import data_processing
import feature_eng
import configs

def convert_numbers(obj):
        for key, value in obj.items():
            if isinstance(value, str) and value.isdigit():
                obj[key] = int(value)
            elif isinstance(value, str):
                try:
                    obj[key] = float(value)
                except ValueError:
                    pass
        return obj

def load_best_params(region, from_json=True):
    if from_json:
        with open('modelling/region_best_params.json') as file:
            best_params = json.load(file, object_hook=convert_numbers)
        if region in best_params:
            return best_params[region]
        else:
            warnings.warn(f"No best parameters found for region: {region}, returning empty defaults")
            return {}
    
    # If not loading with a JSON, load with MLflow TODO
    return

def load_configs():
    # configs = {
    #     "target_name": "AveragePrice",
    #     "lags":[4, 8, 13, 26, 52],
    #     # "lags":[4],
    #     # 'target_regions':['Chicago','Albany'],
    #     'target_regions': [
    #         'Albany', 
    #         'Atlanta', 'BaltimoreWashington', 'Boise', 'Boston',
    #         'BuffaloRochester', 'California', 'Charlotte', 'Chicago',
    #         'CincinnatiDayton', 'Columbus', 'DallasFtWorth', 'Denver',
    #         'Detroit', 'GrandRapids', 'GreatLakes', 'HarrisburgScranton',
    #         'HartfordSpringfield', 'Houston', 'Indianapolis', 'Jacksonville',
    #         'LasVegas', 'LosAngeles', 'Louisville', 'MiamiFtLauderdale',
    #         'Midsouth', 'Nashville', 'NewOrleansMobile', 'NewYork',
    #         'Northeast', 'NorthernNewEngland', 'Orlando', 'Philadelphia',
    #         'PhoenixTucson', 'Pittsburgh', 'Plains', 'Portland',
    #         'RaleighGreensboro', 'RichmondNorfolk', 'Roanoke', 'Sacramento',
    #         'SanDiego', 'SanFrancisco', 'Seattle', 'SouthCarolina',
    #         'SouthCentral', 'Southeast', 'Spokane', 'StLouis', 'Syracuse',
    #         'Tampa', 'TotalUS', 'West', 
    #         'WestTexNewMexico'],
    #     "aux_regions": ['TotalUS', 'West', 'Midsouth', 'Northeast', 'Southeast', 'SouthCentral'],
    #     "aux_features": ['AveragePrice_combined', 'TotalVolume_combined', 
    #                      '4046_combined', '4225_combined', '4770_combined', 
    #                      'TotalBags_combined', 'SmallBags_combined', 
    #                      'LargeBags_combined', 'XLargeBags_combined'],
    #     "aux_lags": [4, 8, 13, 26, 52],
    #     # "aux_lags": [4],
    #     'experiment_name': 'Price Forecasting11 - Regions'
    # }
    
    return configs.configs

def plot_results(y_train, y_true, y_pred, target_name, dates, fold=None):
    # Plot the fold results
    plt.figure(figsize=(12, 6))
    plt.plot(dates.loc[y_train.index], y_train, label='Train')
    plt.plot(dates.loc[y_true.index], y_true, label='True')
    plt.plot(dates.loc[y_true.index], y_pred, label='Predicted')
    plt.legend()
    if fold:
        plt.title(f"Train, Validation, and Predicted Values - Fold {fold}")
    else:
        plt.title(f"Train, Test, and Predicted Values")
    plt.xlabel("Date")
    plt.ylabel(target_name)
    return plt

def main(experiment_name = os.getenv("EXPERIMENT_NAME")):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))  # Set this if using a tracking server

    configs = load_configs()
    print('loaded configs')
    merge_df = data_processing.make_stage_1_data(configs)
    print('Loaded data')
    print(os.getenv("MLFLOW_TRACKING_URI"))
    print(os.getenv("AAA"))
    mlflow.set_experiment(experiment_name)
    print('Set experiment name')
    for region in configs['target_regions']:
        print(f'Trainning: {region}')
        X, y, dates = feature_eng.make_stage_2_data(merge_df, region, configs)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Start one run per region
        with mlflow.start_run(run_name=f"Region: {region}") as region_run:
            mlflow.log_param("region", region)

            # get best params from grid search
            params = load_best_params(region, from_json=True)
            mlflow.log_params(params)

            # Evaluate final model on the hold-out test set using training data only
            final_model_cv = xgb.XGBRegressor(**params)
            final_model_cv.fit(X_train, y_train)
            y_test_pred = final_model_cv.predict(X_test)
            test_mse = mean_squared_error(y_test, y_test_pred)
            mlflow.log_metric("test_mse", test_mse)
            
            plot = plot_results(y_train, y_test, y_test_pred, configs['target_name'], dates)
            plot_path = f"plot_test.png"
            plot.savefig(plot_path)
            mlflow.log_artifact(plot_path)
            plot.close()
            # Delete the plot file after logging it
            if os.path.exists(plot_path):
                os.remove(plot_path)
            
            final_model = xgb.XGBRegressor(**params)
            final_model.fit(X, y)

            mlflow.xgboost.log_model(final_model, artifact_path="final_model", input_example=X.iloc[:1])
            
            # Register the model
            model_uri = mlflow.get_artifact_uri("final_model")
            tags = {"region": region}
            mlflow.register_model(model_uri, name=f'{region}_AVOCADO_FORECAST', tags=tags)

if __name__ =='__main__':
    load_dotenv()
    start = time.time()
    print('START!')
    main()
    end = time.time()
    elapsed_time = start - end
    print(f"Elapsed time for main training function: {elapsed_time:.2f} seconds")