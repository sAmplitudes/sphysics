# Makefile


# Define the command to run pylint
lint:
	pylint --rcfile src/sphysics/pylintrc src/sphysics

# Define the command to run pre-commit tests
pre-commit:
	pre-commit run --hook-stage manual --all-files

# Define a command to run both pylint and pre-commit tests
test: pre-commit lint

.PHONY: lint pre-commit test
