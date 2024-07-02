setup:
	pre-commit install

lint:
	pylint src

test:
	pytest src