import pickle
import mlflow
import mlflow.sklearn
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
        mlflow.sklearn.log_model(model, model_name)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mean_absolute_error", mae)

        for param_key, param_value in model_params.items():
            mlflow.log_param(param_key, param_value)

    return r2, mae
