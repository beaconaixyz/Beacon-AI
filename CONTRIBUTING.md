# Contributing to BEACON

We love your input! We want to make contributing to BEACON as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code lints
6. Issue that pull request!

## Development Setup

1. Install Poetry (dependency management):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/beacon.git
   cd beacon
   ```

3. Install dependencies:
   ```bash
   poetry install
   ```

4. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Style

- We use `black` for code formatting
- We use `isort` for import sorting
- We use `mypy` for type checking
- We use `flake8` for linting

Run all style checks:
```bash
poetry run black .
poetry run isort .
poetry run mypy .
poetry run flake8 .
```

## Testing

Run tests with:
```bash
poetry run pytest
```

## Documentation

- Use Google-style docstrings
- Update the docs when you change code
- Add examples for new features

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Update the docs with details of changes if needed
3. The PR will be merged once you have the sign-off of two other developers

## Any contributions you make will be under the MIT Software License

In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using GitHub's [issue tracker](https://github.com/yourusername/beacon/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/yourusername/beacon/issues/new).

## License

By contributing, you agree that your contributions will be licensed under its MIT License.
