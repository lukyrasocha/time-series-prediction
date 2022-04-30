# Mlflow with Azure (tracking and deployment)

## Prerequisities
1. Conda [installed](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. `pip install mlflow`

## Notes

Ml-flow ensures tracking, reproducibility and deployment

There are 3 ways you could track the experiments, either you can choose to update your experiments on "https://training.itu.dk:5000/" (or any other personal url). Or you can use Microsoft Azure Machine Learning Studio, or you can save them locally and use `mlflow ui` to display them on your localhost.

For this assignment I decided to track my experiments on Microsoft Azure ML studio, since the UI looks nice and also wanted to get more familiar with azure.

To track your experiments in Azure ML studio you must create ML workspace, download the `config.json` and place it in the working local directory. 

Include these lines to `main.py`
```python
ws = Workspace.from_config()
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
```
Note: If you comment these lines out, `mlflow` will track your experiments and save them locally (you can view them on a `localhost` by running `mlflow ui`)

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

## Run using `mlflow projects`

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
mlflow run . --experiment-name='lukr - Assignment3' -P modelname=... -P polydegree=... -P number_of_splits=...
```

## Run and serve the model locally
Run the following command:
```
mlflow run https://github.com/lukyrasocha/ml-flow-azure --version main --experiment-name='lukr - Assignment3'
```
- `--version` is just the git branch (because mlflow defaultly assumes that the branch is `master`)
- `--experiment-name` must be the same as is defined in the `main.py` 

This will create a folder `mlruns` 

Navigate into `mlruns > 1 > YOUR_RUN_ID > artifacts`

Then run
```
mlflow models serve -m model
```
This will serve the model that was created by the specific run on your localhost (`127.0.0.1:5000`)


## Get predictions from the model
```
curl 127.0.0.1:5000/invocations -H Content-Type: application/json -d {"columns": ["Speed", "Direction"], "data": [[10,"W"]]}
```
This will return a list of predictions (e.g. `[7.817]`)

## Serve your model on a VM in Azure

First in the azure web interface (or using the azure sdk) create a VM.
SSH into it using your key-pair: `ssh -i <path_to_private_key> <username>@<PUBLIC_IP_OF_THE_VM>`

```
ssh -i ~/lukr lukyrasocha@20.67.184.90
```

Install miniconda onto your VM
```
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh
bash Miniconda3-py39_4.10.3-Linux-x86_64.sh
```

Install `mlflow`
```
conda install -c conda-forge mlflow
```

[Configure](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/nsg-quickstart-portal) your VM to accept requests on port `5000`

Git clone this repo and serve the model on the VM
```
git clone https://github.com/lukyrasocha/ml-flow-azure.git
cd ml-flow-azure
mlflow models serve -m best_model -h 0.0.0.0 -p 5000
```

I am serving the model on my VM, you can try it out
```
curl http://20.67.184.90:5000/invocations -H 'Content-Type: application/json' -d '{"columns": ["Speed", "Direction"], "data": [[10,"W"]]}'
```
