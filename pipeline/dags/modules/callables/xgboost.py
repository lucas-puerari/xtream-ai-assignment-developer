from modules.processor import xgboost_data_preparation, post_data_preparation
from modules.optimizer import optmize_xgboost_model
from modules.trainer import train_xgboost_model, evaluate_model


def xgboost_data_preparation_callable(**kwargs):
    ti = kwargs['ti']
    features = ti.xcom_pull(key='features', task_ids='pre_data_preparation_task')
    features = xgboost_data_preparation(features)
    ti.xcom_push(key='features', value=features)


def post_xgboost_data_preparation_callable(**kwargs):
    ti = kwargs['ti']
    features = ti.xcom_pull(key='features', task_ids='xgboost_data_preparation_task')
    subsets = post_data_preparation(features)
    ti.xcom_push(key='subsets', value=subsets)


def optimize_xgboost_hyperparameters_callable(**kwargs):
    ti = kwargs['ti']
    subsets = ti.xcom_pull(key='subsets', task_ids='post_xgboost_data_preparation_task')
    x_train, x_test, y_train, y_test = subsets
    xgboost = optmize_xgboost_model(x_train, x_test, y_train, y_test)
    ti.xcom_push(key='hyperparameters', value=xgboost)


def train_xgboost_model_callable(**kwargs):
    ti = kwargs['ti']
    subsets = ti.xcom_pull(key='subsets', task_ids='post_xgboost_data_preparation_task')
    params = ti.xcom_pull(key='hyperparameters', task_ids='optimize_xgboost_model_task')
    # pylint: disable=unused-variable
    x_train, x_test, y_train, y_test = subsets
    xgboost = train_xgboost_model(x_train, y_train, params)
    ti.xcom_push(key='xgboost', value=xgboost)


def evaluate_xgboost_model_callable(**kwargs):
    ti = kwargs['ti']
    xgboost = ti.xcom_pull(key='xgboost', task_ids='train_xgboost_model_task')
    subsets = ti.xcom_pull(key='subsets', task_ids='post_xgboost_data_preparation_task')
    # pylint: disable=unused-variable
    x_train, x_test, y_train, y_test = subsets
    evaluate_model(xgboost, x_test, y_test)
