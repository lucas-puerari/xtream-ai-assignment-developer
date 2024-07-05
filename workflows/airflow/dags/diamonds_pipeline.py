from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

# pylint: disable=wrong-import-order

from modules.callables.common import download_data_callable, pre_data_preparation_callable
from modules.callables.linear import linear_data_preparation_callable,\
    post_linear_data_preparation_callable, train_linear_model_callable,\
    evaluate_linear_model_callable
from modules.callables.xgboost import xgboost_data_preparation_callable,\
    post_xgboost_data_preparation_callable, optimize_xgboost_hyperparameters_callable,\
    train_xgboost_model_callable, evaluate_xgboost_model_callable


default_args = {
    'owner': 'airflow',
    'start_date': datetime.now(),
    'retries': 0
}

dag = DAG(
    'diamonds_pipeline', 
    default_args=default_args,
    description='A DAG to orchestrate the processing and modeling of diamond dataset'
)

# common operators

download_data_task = PythonOperator(
    task_id='download_data_task',
    python_callable=download_data_callable,
    dag=dag,
)

pre_data_preparation_task = PythonOperator(
    task_id='pre_data_preparation_task',
    python_callable=pre_data_preparation_callable,
    dag=dag,
)

# linear operators

linear_data_preparation_task = PythonOperator(
    task_id='linear_data_preparation_task',
    python_callable=linear_data_preparation_callable,
    dag=dag
)

post_linear_data_preparation_task = PythonOperator(
    task_id='post_linear_data_preparation_task',
    python_callable=post_linear_data_preparation_callable,
    dag=dag
)

train_linear_model_task = PythonOperator(
    task_id='train_linear_model_task',
    python_callable=train_linear_model_callable,
    dag=dag
)

evaluate_linear_model_task = PythonOperator(
    task_id='evaluate_linear_model_task',
    python_callable=evaluate_linear_model_callable,
    dag=dag
)

# xgboost operators

xgboost_data_preparation_task = PythonOperator(
    task_id='xgboost_data_preparation_task',
    python_callable=xgboost_data_preparation_callable,
    dag=dag
)

post_xgboost_data_preparation_task = PythonOperator(
    task_id='post_xgboost_data_preparation_task',
    python_callable=post_xgboost_data_preparation_callable,
    dag=dag
)

optimize_xgboost_model_task = PythonOperator(
    task_id='optimize_xgboost_model_task',
    python_callable=optimize_xgboost_hyperparameters_callable,
    dag=dag
)

train_xgboost_model_task = PythonOperator(
    task_id='train_xgboost_model_task',
    python_callable=train_xgboost_model_callable,
    dag=dag
)

evaluate_xgboost_model_task = PythonOperator(
    task_id='evaluate_xgboost_model_task',
    python_callable=evaluate_xgboost_model_callable,
    dag=dag
)

# linear pipeline

# pylint: disable=pointless-statement
download_data_task\
    >> pre_data_preparation_task >> linear_data_preparation_task\
    >> post_linear_data_preparation_task >> train_linear_model_task\
    >> evaluate_linear_model_task

# xgboost pipeline

# pylint: disable=pointless-statement
download_data_task\
    >> pre_data_preparation_task >> xgboost_data_preparation_task\
    >> post_xgboost_data_preparation_task >> optimize_xgboost_model_task\
    >> train_xgboost_model_task >> evaluate_xgboost_model_task
