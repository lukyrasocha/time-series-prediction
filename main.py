#Assignment 3 - Mlflow with Azure
#Author: Lukas Rasocha (inspired by a template provided by the lecturer)
import pandas as pd
import mlflow
from datetime import datetime
from azureml.core import Workspace
import sys
import numpy as np
import argparse

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from matplotlib import pyplot as plt 

#UNCOMMENT THESE TWO LINES IF YOU WISH TO EXPERIMENT REMOTELY BASED ON YOUR CONFIG FILE INCLUDED IN THE SAME DIRECTORY
#ws = Workspace.from_config()
#mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

#SETUP
mlflow.set_experiment("lukr - Assignment3")
experiment = mlflow.get_experiment_by_name("lukr - Assignment3")

parser = argparse.ArgumentParser()
parser.add_argument("-poly", "--polydegree",type=int)
parser.add_argument("-model", "--modelname",type=str)
parser.add_argument("-splits", "--number_of_splits",type=int)
parser.add_argument("-n_neigh", "--n_neighbours", type=int)
parser.add_argument("-w", "--weights", type=str)
args = parser.parse_args()

model = args.modelname
if model not in ['knn','lin_reg']:
    raise "Error: Argument not known: default{knn,lin_reg}"

# Start a run
name = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {model}" 
with mlflow.start_run(run_name=name, experiment_id=experiment.experiment_id):

    class cardinal_to_degrees(BaseEstimator, TransformerMixin):
        def fit(self,X,y=None):
            return self
        
        def transform(self,X,y=None):
            cardinal_directions = {
                'N':0,
                'NNE':22.5,
                'NE':45,
                'ENE':67.5,
                'E':90,
                'ESE':112.5,
                'SE':135,
                'SSE':157.5,
                'S':180,
                'SSW':202.5,
                'SW':225,
                'WSW':247.5,
                'W':270,
                'WNW':292.5,
                'NW':315,
                'NNW':337.5}
            X_ = X.copy()
        
            for direction in cardinal_directions:
                X_.loc[X_["Direction"] == direction, "Direction"] = cardinal_directions[direction]
                        
            return X_
        
    numeric_transformer = Pipeline(steps=[
           ('imputer', SimpleImputer(strategy='mean'))
          ,('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
       transformers=[
        ('numeric', numeric_transformer, ['Speed']),
        ('decode', cardinal_to_degrees(), ['Direction'])
    ]) 

    df = pd.read_json("data/dataset.json", orient="split")
    df = df.drop(['Source_time','Lead_hours','ANM','Non-ANM'],axis=1)
    df.dropna(subset=['Total','Direction'], inplace=True) #Speed will be imputed in the pipeline

    mlflow.log_param('model_name', model)
    if model == 'lin_reg':
        poly_degree = args.polydegree
        number_of_splits = args.number_of_splits
        pipeline = Pipeline(steps = [
                   ('preprocessor', preprocessor)
                  ,('poly_features', PolynomialFeatures(degree=poly_degree, include_bias=False))
                  ,('regressor',LinearRegression())
               ])

        mlflow.log_param("poly_degree", poly_degree)
        mlflow.log_param("number_of_splits", number_of_splits)
        

    else:
        n_neighbours= args.n_neighbours 
        weights = args.weights 
        number_of_splits = args.number_of_splits
        pipeline = Pipeline(steps = [
           ('preprocessor', preprocessor)
          ,('regressor',KNeighborsRegressor(n_neighbors=n_neighbours, weights=weights))
        ])

        mlflow.log_param('n_neighbours', n_neighbours)
        mlflow.log_param('weights', weights)
        mlflow.log_param("number_of_splits", number_of_splits)

    metrics = [
        ("MAE", mean_absolute_error, []),
        ("MSE", mean_squared_error, []),
        ("r2", r2_score, [])
    ]

    X = df[["Speed","Direction"]]
    y = df["Total"]
    i = 0
    for train, test in TimeSeriesSplit(number_of_splits).split(X,y):
        i += 1 
        pipeline.fit(X.iloc[train],y.iloc[train])
        predictions = pipeline.predict(X.iloc[test])
        truth = y.iloc[test]

        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        #plt.plot(truth.index, truth.values, label="Truth")
        #plt.plot(truth.index, predictions, label="Predictions")
        ax.set_title(f'Split {i}')
        ax.plot(truth.index, truth.values, label="Truth")
        ax.plot(truth.index, predictions, label="Predictions")
        ax.legend()
        mlflow.log_figure(fig, f'split_{i}.png')
        
        # Calculate and save the metrics for this fold
        for name, func, scores in metrics:
            score = func(truth, predictions)
            scores.append(score)
    
    # Log a summary of the metrics
    for name, _, scores in metrics:
            mean_score = sum(scores)/number_of_splits
            var_score = np.var(scores)
            mlflow.log_metric(f"mean_{name}", mean_score)
            mlflow.log_metric(f"Variance_{name}", var_score)

    run = mlflow.active_run()
    print('=============================================================================')
    print("===== Run with run_id {} successfully finished =====".format(run.info.run_id))
    print('=============================================================================')
    mlflow.sklearn.log_model(pipeline, "model")


