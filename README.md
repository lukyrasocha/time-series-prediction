# ml-flow-azure

Ml-flow ensures
- Tracking
- Reproducibility
- Deployment

Notes:
There are 3 ways you can track your experiments, either you can choose to update your experiments on "http://training.itu.dk:5000/". Oryou can use Microsoft Azure Machine Learning Studio, or you can save them locally and use `mlflow ui` to display them on your localhost.

For this assignment I decided to track my experiments on Microsoft Azure ML studio, since the UI looks nice and also wanted to get more familiar with azure.

To track your experiments in Azure ML studio you must create ML workspace, download the `config.json` and place it in the working local directory (where you run your main experiment script)

Include these lines to `main.py`
```python
ws = Workspace.from_config()
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
```
Note: If you comment these lines out, `mlflow` will track your experiments and save them locally (you can view them on `localhost` by running `mlflow ui`)

To track your experiments on another `url` one can use:
```python
 mlflow.set_tracking_uri("http://training.itu.dk:5000/")
```
