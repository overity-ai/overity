default_python_version := "3.14"


###########################################
# Command recipes
###########################################

uv *args:
	PYTHONPATH="" uv {{args}}

pytest *args:
	just uv run pytest {{args}}

ruff *args:
	just uv run ruff {{args}}

black *args:
	just uv run black {{args}}


###########################################
# Project recipes
###########################################

build:
	just uv build

docs:
	just uv run mkdocs build


test py=default_python_version:
	just uv run --python {{py}} pytest -vv

lint-headers:
	just uv run scripts/header_check.py src

lint-rules:
	just ruff check src

lint-format:
	just black src
