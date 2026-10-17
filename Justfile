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

mypy *args:
	just uv run mypy {{args}}


###########################################
# Project recipes
###########################################

build:
	just uv build

docs:
	just uv run mkdocs build


test py=default_python_version:
	just uv run --python {{py}} pytest --cov=src/overity -vv

lint-headers:
	just uv run scripts/header_check.py src

lint-rules:
	just ruff check src

lint-format:
	just black src

lint-types:
	just mypy src

# Test all supported Python versions and print a summary table
test-all:
	#!/usr/bin/env bash
	set -euo pipefail
	python_versions=("3.12" "3.13" "3.14")
	results=()
	echo "Running tests for all supported Python versions..."
	echo ""
	for py in "${python_versions[@]}"; do
		echo "========================================"
		echo "Testing Python ${py}..."
		echo "========================================"
		if just test "${py}"; then
			results+=("${py}: SUCCESS")
			echo "✓ Python ${py} tests PASSED"
		else
			results+=("${py}: FAILED")
			echo "✗ Python ${py} tests FAILED"
		fi
		echo ""
	done
	echo "========================================"
	echo "           TEST SUMMARY"
	echo "========================================"
	printf "| %-10s | %-10s |\n" "Python" "Status"
	echo "|------------|------------|"
	for result in "${results[@]}"; do
		py="${result%%:*}"
		status="${result#*: }"
		if [ "${status}" = "SUCCESS" ]; then
			printf "| %-10s | %-10s |\n" "${py}" "✓ PASS"
		else
			printf "| %-10s | %-10s |\n" "${py}" "✗ FAIL"
		fi
	done
	echo "========================================"
	# Return non-zero exit code if any test failed
	for result in "${results[@]}"; do
		if [[ "${result}" == *"FAILED" ]]; then
			exit 1
		fi
	done
