from modules.retriever import download_data
from modules.processor import pre_data_preparation, linear_data_preparation, post_data_preparation
from modules.trainer import train_linear_model, evaluate_model


# pylint: disable=line-too-long
DATA_URL = 'https://raw.githubusercontent.com/xtreamsrl/xtream-ai-assignment-engineer/main/datasets/diamonds/diamonds.csv'


def download_data_callable(**kwargs):
    ti = kwargs['ti']
    data = download_data(url=DATA_URL)
    ti.xcom_push(key='data', value=data)


def pre_data_preparation_callable(**kwargs):
    ti = kwargs['ti']
    data = ti.xcom_pull(key='data', task_ids='download_data_task')
    features = pre_data_preparation(data)
    ti.xcom_push(key='features', value=features)


def linear_data_preparation_callable(**kwargs):
    ti = kwargs['ti']
    features = ti.xcom_pull(key='features', task_ids='pre_data_preparation_task')
    features = linear_data_preparation(features)
    ti.xcom_push(key='features', value=features)


def post_data_preparation_callable(**kwargs):
    ti = kwargs['ti']
    features = ti.xcom_pull(key='features', task_ids='linear_data_preparation_task')
    subsets = post_data_preparation(features)
    ti.xcom_push(key='subsets', value=subsets)


def train_linear_model_callable(**kwargs):
    ti = kwargs['ti']
    subsets = ti.xcom_pull(key='subsets', task_ids='post_data_preparation_task')
    # pylint: disable=unused-variable
    x_train, x_test, y_train, y_test = subsets
    regressor = train_linear_model(x_train, y_train)
    ti.xcom_push(key='regressor', value=regressor)


def evaluate_linear_model_callable(**kwargs):
    ti = kwargs['ti']
    regressor = ti.xcom_pull(key='regressor', task_ids='train_linear_model_task')
    subsets = ti.xcom_pull(key='subsets', task_ids='post_data_preparation_task')
    # pylint: disable=unused-variable
    x_train, x_test, y_train, y_test = subsets
    evaluate_model(regressor, x_test, y_test)
