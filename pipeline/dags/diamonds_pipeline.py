from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

from modules.callables import download_data_callable, pre_data_preparation_callable,\
    linear_data_preparation_callable, post_data_preparation_callable, train_linear_model_callable,\
    evaluate_linear_model_callable


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

linear_data_preparation_task = PythonOperator(
    task_id='linear_data_preparation_task',
    python_callable=linear_data_preparation_callable,
    dag=dag
)

post_data_preparation_task = PythonOperator(
    task_id='post_data_preparation_task',
    python_callable=post_data_preparation_callable,
    dag=dag
)

train_linear_model_task = PythonOperator(
    task_id='train_linear_model_task',
    python_callable=train_linear_model_callable,
    dag=dag
)

evaluate_linear_model_task = PythonOperator(
    task_id='evaluate_model_task',
    python_callable=evaluate_linear_model_callable,
    dag=dag
)

# pylint: disable=pointless-statement
download_data_task >> pre_data_preparation_task >> linear_data_preparation_task\
    >> post_data_preparation_task >> train_linear_model_task >> evaluate_linear_model_task
