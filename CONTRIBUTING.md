# Contributing Guide

Thank you for your interest in contributing to the Customer Churn Analysis and Prediction project.

This repository is a portfolio and educational machine learning project. Contributions that improve the code quality, documentation, testing, reproducibility, or machine learning workflow are welcome.

## Before Contributing

Please read the project documentation in `README.md` to understand:

* The project objectives.
* The machine learning workflow.
* The dataset preparation process.
* The model evaluation approach.
* The project structure.

## Development Setup

Clone the repository and move into the project directory.

Install the project and its dependencies with:

```powershell
python -m pip install -e .
```

The project is developed and tested with Python 3.11.

## Making Changes

When contributing:

1. Create a separate branch for your changes.
2. Keep changes focused on a specific improvement.
3. Follow the existing project structure and coding style.
4. Avoid unnecessary changes to unrelated files.
5. Update documentation when a change affects project behavior or usage.
6. Add or update tests when appropriate.

## Testing

Run the complete test suite before submitting a contribution:

```powershell
python -m pytest -q
```

Please make sure the tests pass before opening a pull request.

## Machine Learning Changes

Changes to the machine learning workflow should be reproducible and clearly documented.

When modifying preprocessing, feature selection, model configuration, training, evaluation, or cross-validation:

* Explain the reason for the change.
* Preserve the existing train/test separation.
* Avoid data leakage.
* Keep random seeds reproducible where applicable.
* Update relevant tests and documentation.
* Report meaningful changes to evaluation results.

Do not replace the existing holdout evaluation with cross-validation. Cross-validation is used as an additional validation step in this project.

## Data and Generated Files

Please do not commit:

* Virtual environments.
* Python cache files.
* Logs.
* Generated model artifacts.
* Generated reports that are excluded by `.gitignore`.
* Credentials, API keys, or other secrets.
* Unnecessary temporary files.

Follow the repository's `.gitignore` rules when adding files.

## Commit Messages

Use clear and descriptive commit messages.

Examples:

```text
Add cross-validation tests
Update project documentation
Improve benchmark reporting
Fix preprocessing validation
```

Avoid vague messages such as:

```text
update
changes
fix stuff
```

## Pull Requests

Before opening a pull request:

1. Make sure the project tests pass.
2. Review your changes with Git.
3. Make sure no secrets or unnecessary generated files are included.
4. Update documentation if required.
5. Provide a clear description of what changed.

A pull request should include:

* A short summary of the change.
* The reason for the change.
* Testing performed.
* Any relevant limitations or considerations.

## Reporting Issues

If you find a bug or have a suggestion, open a GitHub Issue with enough information to understand the problem.

For bugs, include:

* A clear description of the problem.
* Steps to reproduce it.
* Expected behavior.
* Actual behavior.
* Relevant error messages or test output.

## Security Issues

Please do not publicly disclose sensitive security issues through GitHub Issues.

Follow the instructions in [`SECURITY.md`](SECURITY.md) for reporting potential security vulnerabilities.

## Code of Conduct

Please communicate respectfully and constructively when contributing to the project.

Contributors are expected to focus on improving the project and maintaining a professional development environment.

Thank you for helping improve the Customer Churn Analysis and Prediction project.