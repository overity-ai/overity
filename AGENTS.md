# Development Commands

This file documents the commands used for development tasks in the Overity.ai project.

## Type Checking
- `just lint-types`: Run mypy for type checking.

## Linting and Formatting
- `just lint-rules`: Lint the code using ruff.
- `just lint-format`: Format the code using black.

## Documentation
- `just docs`: Build the project documentation using mkdocs.

## Project Structure

The `src/overity` folder contains the following subfolders:

- `api`: API definition that can be used in overity methods
- `backend`: Internal operations
- `bench`: Bench abstraction definition
- `exchange`: Exchange file parsers and encoders
- `frontend`: CLI related code
- `model`: Data structures (especially dataclasses) definition
- `storage`: Storage abstractions definition

## Testing

Test code is available in the `tests` folder

To run tests, use the `just test` command
