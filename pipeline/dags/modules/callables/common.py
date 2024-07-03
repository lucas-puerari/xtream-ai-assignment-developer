from modules.retriever import download_data
from modules.processor import pre_data_preparation


# pylint: disable=line-too-long
DATA_URL = 'https://raw.githubusercontent.com/xtreamsrl/xtream-ai-assignment-engineer/main/datasets/diamonds/diamonds.csv'


def download_data_callable(**kwargs):
    ti = kwargs['ti']
    # let's suppose the csv file is updated with new data
    data = download_data(url=DATA_URL)
    ti.xcom_push(key='data', value=data)


def pre_data_preparation_callable(**kwargs):
    ti = kwargs['ti']
    data = ti.xcom_pull(key='data', task_ids='download_data_task')
    features = pre_data_preparation(data)
    ti.xcom_push(key='features', value=features)
