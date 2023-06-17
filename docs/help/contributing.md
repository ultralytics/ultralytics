---
comments: true
description: Learn how to contribute to Ultralytics Open-Source YOLO Repositories with contributions guidelines, pull requests requirements, and GitHub CI tests.
keywords: Ultralytics YOLO, Open source, Contribution guidelines, Pull requests, CLA, GitHub Actions CI Tests, Google-style docstrings
---

# Contributing to Ultralytics Open-Source YOLO Repositories

First of all, thank you for your interest in contributing to Ultralytics open-source YOLO repositories! Your contributions will help improve the project and benefit the community. This document provides guidelines and best practices for contributing to Ultralytics YOLO repositories.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Pull Requests](#pull-requests)
    - [CLA Signing](#cla-signing)
    - [Google-Style Docstrings](#google-style-docstrings)
    - [GitHub Actions CI Tests](#github-actions-ci-tests)
- [Bug Reports](#bug-reports)
    - [Minimum Reproducible Example](#minimum-reproducible-example)
- [License and Copyright](#license-and-copyright)

## Code of Conduct

All contributors are expected to adhere to the [Code of Conduct](code_of_conduct.md) to ensure a welcoming and inclusive environment for everyone.

## Pull Requests

We welcome contributions in the form of pull requests. To make the review process smoother, please follow these guidelines:

1. **Fork the repository**: Fork the Ultralytics YOLO repository to your own GitHub account.

2. **Create a branch**: Create a new branch in your forked repository with a descriptive name for your changes.

3. **Make your changes**: Make the changes you want to contribute. Ensure that your changes follow the coding style of the project and do not introduce new errors or warnings.

4. **Test your changes**: Test your changes locally to ensure that they work as expected and do not introduce new issues.

5. **Commit your changes**: Commit your changes with a descriptive commit message. Make sure to include any relevant issue numbers in your commit message.

6. **Create a pull request**: Create a pull request from your forked repository to the main Ultralytics YOLO repository. In the pull request description, provide a clear explanation of your changes and how they improve the project.

### CLA Signing

Before we can accept your pull request, you need to sign a [Contributor License Agreement (CLA)](CLA.md). This is a legal document stating that you agree to the terms of contributing to the Ultralytics YOLO repositories. The CLA ensures that your contributions are properly licensed and that the project can continue to be distributed under the AGPL-3.0 license.

To sign the CLA, follow the instructions provided by the CLA bot after you submit your PR.

### Google-Style Docstrings

When adding new functions or classes, please include a [Google-style docstring](https://google.github.io/styleguide/pyguide.html) to provide clear and concise documentation for other developers. This will help ensure that your contributions are easy to understand and maintain.

Example Google-style docstring:

```python
def example_function(arg1: int, arg2: str) -> bool:
    """Example function that demonstrates Google-style docstrings.

    Args:
        arg1 (int): The first argument.
        arg2 (str): The second argument.

    Returns:
        bool: True if successful, False otherwise.

    Raises:
        ValueError: If `arg1` is negative or `arg2` is empty.
    """
    if arg1 < 0 or not arg2:
        raise ValueError("Invalid input values")
    return True
```

### GitHub Actions CI Tests

Before your pull request can be merged, all GitHub Actions Continuous Integration (CI) tests must pass. These tests include linting, unit tests, and other checks to ensure that your changes meet the quality standards of the project. Make sure to review the output of the GitHub Actions and fix any issues