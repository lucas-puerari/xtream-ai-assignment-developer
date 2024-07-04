setup:
	pre-commit install

freeze:
	pip freeze > requirements.txt

start:
	docker-compose up

stop:
	docker-compose down