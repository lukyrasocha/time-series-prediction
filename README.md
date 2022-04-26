# ml-flow-azure

Ml-flow ensures
- Tracking
- Reproducibility
- Deployment

Notes:
There are 3 ways you can track your experiments, either you can choose to update your experiments on "https://training.itu.dk:5000/". Oryou can use Microsoft Azure Machine Learning Studio, or you can save them locally and use `mlflow ui` to display them on your localhost.

For this assignment I decided to track my experiments on Microsoft Azure ML studio, since the UI looks nice and also wanted to get more familiar with azure.

To track your experiments in Azure ML studio you must create ML workspace, download the `config.json` and place it in the working local directory. 

Include these lines to `main.py`
```python
ws = Workspace.from_config()
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
```
Note: If you comment these lines out, `mlflow` will track your experiments and save them locally (you can view them on a `localhost` by running `mlflow ui`)

## Prerequisities
1. Conda [installed](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

## How to run it

1. Create a virtual environment (for example using `conda` or `pyenv`) and download all the dependencies in `conda.yaml`
```
git clone https://github.com/lukyrasocha/ml-flow-azure.git
conda env create -f conda.yaml
conda activate ml-flow-azure 
```
You can experiment with two different models `KNN Regressor` and `Linear Regressor`

KNN example
```
python main.py --modelname=knn --n_neighbours=8 --number_of_splits=5 --weights=uniform
```

Linear Regression example
```
python main.py --modelname=lin_reg --polydegree=3 --number_of_splits=5
```
If you are running the experiment locally you can run `mlflow ui` to view the runs (or in Azure ML studio > Experiments otherwise)

## Running it using `mlflow projects`

```
git clone https://github.com/lukyrasocha/ml-flow-azure.git
cd ml-flow-azure
mlflow run . --experiment-name='lukr - Assignment3'
```
- To make it work remotely, you must uncomment the `set_tracking_uri()` line in `main.py` and you also must set an enviroment variable for the `uri` before running `mlflow run .` (you can read why [here](https://lifesaver.codes/answer/runid-not-found-when-executing-mlflow-run-with-remote-tracking-server-608)).
- The `--experiment-name` must correspond to the one set in `main.py`

## Run with custom parameters
I set the default parameters to the ones that performed best, but if you wish to change them here is an example:
```
mlflow run . --experiment-name='lukr - Assignment3' -P modelname=... polydegree=... number_of_splits=...
```

