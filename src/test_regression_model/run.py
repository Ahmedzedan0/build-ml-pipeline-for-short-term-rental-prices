import argparse
import pandas as pd
import mlflow
import wandb
from sklearn.metrics import mean_squared_error, r2_score

def go(args):
    run = wandb.init(job_type="test_regression_model")
    run.config.update(args)

    # Download the model
    model_path = mlflow.artifacts.download_artifacts(args.mlflow_model)

    # Load the model
    model = mlflow.sklearn.load_model(model_path)

    # Download the test dataset
    test_path = wandb.use_artifact(args.test_artifact).file()
    test_data = pd.read_csv(test_path)

    # Separate features and target
    X_test = test_data.drop("price", axis=1)
    y_test = test_data["price"]

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Log metrics
    run.summary["mse"] = mse
    run.summary["r2"] = r2

    print(f"Test MSE: {mse}")
    print(f"Test R2: {r2}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test regression model")
    parser.add_argument("--mlflow_model", type=str, help="Path to the mlflow model", required=True)
    parser.add_argument("--test_artifact", type=str, help="Path to the test data artifact", required=True)
    args = parser.parse_args()
    go(args)
