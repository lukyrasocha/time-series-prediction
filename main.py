import pandas as pd
import mlflow
from utils.transformers import preprocessor
from datetime import datetime
from azureml.core import Workspace
import sys
import numpy as np
import argparse

#UNCOMMENT THESE TWO LINES IF YOU WISH TO EXPERIMENT REMOTELY BASED ON YOUR CONFIG FILE INCLUDED IN THE SAME DIRECTORY
#ws = Workspace.from_config()
#mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

#SETUP
mlflow.set_experiment("please")
mlflow.sklearn.autolog(log_input_examples=True, silent=True)
#model = sys.argv[2] if len(sys.argv) > 1 else 'lin_reg' 
#if model not in ['knn','lin_reg']:
#    raise "Error: Argument not known: default{knn,lin_reg}"

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

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Start a run
name = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {model}" 
with mlflow.start_run(run_name=name):

    df = pd.read_json("data/dataset.json", orient="split")
    df = df.drop(['Source_time','Lead_hours','ANM','Non-ANM'],axis=1)
    df.dropna(subset=['Total','Direction'], inplace=True) #Speed will be imputed in the pipeline

    # TODO: Handle missing data

    mlflow.log_param('model_name', model)
    if model == 'lin_reg':
        poly_degree = args.polydegree#int(sys.argv[3])
        number_of_splits = args.number_of_splits #int(sys.argv[4])
        pipeline = Pipeline(steps = [
                   ('preprocessor', preprocessor)
                  ,('poly_features', PolynomialFeatures(degree=poly_degree, include_bias=False))
                  ,('regressor',LinearRegression())
               ])

        mlflow.log_param("poly_degree", poly_degree)
        mlflow.log_param("number_of_splits", number_of_splits)
        

    else:
        n_neighbours= args.n_neighbours #int(sys.argv[3])
        weights = args.weights #sys.argv[4]
        number_of_splits = args.number_of_splits #int(sys.argv[5])
        pipeline = Pipeline(steps = [
           ('preprocessor', preprocessor)
          ,('regressor',KNeighborsRegressor(n_neighbors=n_neighbours, weights=weights))
        ])

        mlflow.log_param('n_neighbours', n_neighbours)
        mlflow.log_param('weights', weights)
        mlflow.log_param("number_of_splits", number_of_splits)

    # TODO: Currently the only metric is MAE. You should add more. What other metrics could you use? Why?
    metrics = [
        ("MAE", mean_absolute_error, []),
        ("MSE", mean_squared_error, []),
        ("r2", r2_score, [])
    ]

    X = df[["Speed","Direction"]]
    y = df["Total"]


    #TODO: Log your parameters. What parameters are important to log?
    #HINT: You can get access to the transformers in your pipeline using `pipeline.steps`
    
    for train, test in TimeSeriesSplit(number_of_splits).split(X,y):
        pipeline.fit(X.iloc[train],y.iloc[train])
        predictions = pipeline.predict(X.iloc[test])
        truth = y.iloc[test]

 #       from matplotlib import pyplot as plt 
 #       plt.plot(truth.index, truth.values, label="Truth")
 #       plt.plot(truth.index, predictions, label="Predictions")
 #       plt.show()
        
        # Calculate and save the metrics for this fold
        for name, func, scores in metrics:
            score = func(truth, predictions)
            scores.append(score)
    
    # Log a summary of the metrics
    for name, _, scores in metrics:
            # NOTE: Here we just log the mean of the scores. 
            # Are there other summarizations that could be interesting?
            mean_score = sum(scores)/number_of_splits
            var_score = np.var(scores)
            mlflow.log_metric(f"mean_{name}", mean_score)
            mlflow.log_metric(f"Variance_{name}", var_score)

    mlflow.sklearn.log_model(pipeline, "model")
