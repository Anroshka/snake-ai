# Contributing to Snake AI Learning Game

Thank you for considering contributing to the Snake AI Learning Game! We welcome contributions from the community and are grateful for your support.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [How to Contribute](#how-to-contribute)
3. [Development Environment Setup](#development-environment-setup)
4. [Code Style](#code-style)
5. [Pull Request Process](#pull-request-process)
6. [Issue Reporting](#issue-reporting)

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project, you agree to abide by its terms.

## How to Contribute

### Reporting Bugs

If you find a bug in the project, please open an issue on GitHub with detailed information about the bug, including steps to reproduce it, the expected behavior, and any relevant screenshots or logs.

### Requesting Features

If you have an idea for a new feature, please open an issue on GitHub with a detailed description of the feature, including its purpose, how it would be used, and any potential benefits.

### Submitting Pull Requests

1. Fork the repository and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. Ensure the test suite passes.
4. Make sure your code lints.
5. Submit a pull request with a clear description of your changes.

## Development Environment Setup

To set up the development environment for the Snake AI Learning Game, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/yourusername/snake-ai.git
cd snake-ai
```

2. Create a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. If you have an NVIDIA GPU, install PyTorch with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Code Style

We follow the PEP 8 style guide for Python code. Please ensure your code adheres to these guidelines. You can use tools like `flake8` and `black` to check and format your code.

## Pull Request Process

1. Ensure that your code follows the project's coding standards and passes all tests.
2. Write a clear and concise description of your changes in the pull request.
3. Reference any related issues in the pull request description.
4. Be responsive to feedback and make any necessary changes requested by reviewers.

## Issue Reporting

When reporting an issue, please provide as much detail as possible, including:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected and actual behavior
- Screenshots or logs, if applicable
- Any other relevant information

Thank you for contributing to the Snake AI Learning Game!
