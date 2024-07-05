setup:
	pre-commit install

freeze:
	pip freeze > requirements.txt

lint:
	pylint airflow/dags

test:
	pytest airflow/dags/modules
