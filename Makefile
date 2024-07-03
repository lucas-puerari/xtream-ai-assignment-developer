setup:
	pre-commit install

freeze:
	pip freeze > requirements.txt

lint:
	pylint pipeline

test:
	pytest pipeline/dags/modules

start:
	docker-compose up

stop:
	docker-compose down