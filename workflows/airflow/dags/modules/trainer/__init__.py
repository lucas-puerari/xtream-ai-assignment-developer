import pickle
import mlflow
from mlflow.models import infer_signature
import xgboost
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error


MLFLOW_EXPERIMENT_NAME = "diamonds"
SEED = 42


def train_linear_model(x_train, y_train):
    regressor = LinearRegression()

    regressor.fit(x_train, y_train)

    return pickle.dumps(regressor)


def train_xgboost_model(x_train, y_train, params):
    if params is None:
        params = {
            "enable_categorical": True,
            "random_state": SEED
        }

    xgb = xgboost.XGBRegressor(**params)

    xgb.fit(x_train, y_train)

    return pickle.dumps(xgb)


def evaluate_model(model, x_test, y_test):
    model = pickle.loads(model)
    model_name = model.__class__.__name__
    model_params = model.get_params()

    pred = model.predict(x_test)

    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run():
        # I'm not able to fix the following error
        # mlflow.exceptions.MlflowException: Failed to enforce schema of data
        # '[[ 32.   3.   0.   0. 345.]]' with schema '[Array(double) (required)]'.
        # Error: Expected data to be list or numpy array, got float.
        # Temporarily disable the template signature

        # model_signature = infer_signature(x_test, pred)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn-model",
            # signature=model_signature,
            registered_model_name=f"sklearn-{model_name}"
        )

        mlflow.log_params(model_params)

        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mean_absolute_error", mae)

    return r2, mae
