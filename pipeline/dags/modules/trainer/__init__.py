import pickle
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error


MLFLOW_EXPERIMENT_NAME = "diamonds"


def train_linear_model(x_train, y_train):
    regressor = LinearRegression()

    regressor.fit(x_train, y_train)

    return pickle.dumps(regressor)


def evaluate_model(model, x_test, y_test):
    model = pickle.loads(model)

    pred = model.predict(x_test)

    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mean_absolute_error", mae)

    return r2, mae
