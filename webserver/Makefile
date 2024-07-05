setup:
	pre-commit install

freeze:
	pip freeze > requirements.txt

lint:
	pylint src

test:
	pytest src

build:
	docker build . -t diamonds-webserver:0.1.0

start: 
	fastapi dev src/main.py
