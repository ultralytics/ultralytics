---
comments: true
description: Learn how to contribute to Ultralytics YOLO open-source repositories. Follow guidelines for pull requests, code of conduct, and bug reporting.
keywords: Ultralytics, YOLO, open-source, contribution, pull request, code of conduct, bug reporting, GitHub, CLA, Google-style docstrings
---

# Contributing to Ultralytics Open-Source YOLO Repositories

Thank you for your interest in contributing to Ultralytics open-source YOLO repositories! Your contributions will enhance the project and benefit the entire community. This document provides guidelines and best practices to help you get started.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Contributing via Pull Requests](#contributing-via-pull-requests)
   - [CLA Signing](#cla-signing)
   - [Google-Style Docstrings](#google-style-docstrings)
   - [GitHub Actions CI Tests](#github-actions-ci-tests)
3. [Reporting Bugs](#reporting-bugs)
4. [License](#license)
5. [Conclusion](#conclusion)

## Code of Conduct

All contributors must adhere to the [Code of Conduct](code_of_conduct.md) to ensure a welcoming and inclusive environment for everyone.

## Contributing via Pull Requests

We welcome contributions in the form of pull requests. To streamline the review process, please follow these guidelines:

1. **[Fork the repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo)**: Fork the Ultralytics YOLO repository to your GitHub account.

2. **[Create a branch](https://docs.github.com/en/desktop/making-changes-in-a-branch/managing-branches-in-github-desktop)**: Create a new branch in your forked repository with a descriptive name for your changes.

3. **Make your changes**: Ensure that your changes follow the project's coding style and do not introduce new errors or warnings.

4. **[Test your changes](https://github.com/ultralytics/ultralytics/tree/main/tests)**: Test your changes locally to ensure they work as expected and do not introduce new issues.

5. **[Commit your changes](https://docs.github.com/en/desktop/making-changes-in-a-branch/committing-and-reviewing-changes-to-your-project-in-github-desktop)**: Commit your changes with a descriptive commit message. Include any relevant issue numbers in your commit message.

6. **[Create a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)**: Create a pull request from your forked repository to the main Ultralytics YOLO repository. Provide a clear explanation of your changes and how they improve the project.

### CLA Signing

Before we can accept your pull request, you must sign a [Contributor License Agreement (CLA)](CLA.md). This legal document ensures that your contributions are properly licensed and that the project can continue to be distributed under the AGPL-3.0 license.

To sign the CLA, follow the instructions provided by the CLA bot after you submit your PR and add a comment in your PR saying:

```
I have read the CLA Document and I sign the CLA
```

### Google-Style Docstrings

When adding new functions or classes, include a [Google-style docstring](https://google.github.io/styleguide/pyguide.html) to provide clear and concise documentation for other developers. This helps ensure your contributions are easy to understand and maintain.

#### Google-style

This example shows a Google-style docstring. Note that both input and output `types` must always be enclosed by parentheses, i.e. `(bool)`.

```python
def example_function(arg1, arg2=4):
    """
    This example shows a Google-style docstring. Note that both input and output `types` must always be enclosed by
    parentheses, i.e., `(bool)`.

    Args:
        arg1 (int): The first argument.
        arg2 (int): The second argument. Default value is 4.

    Returns:
        (bool): True if successful, False otherwise.

    Examples:
        >>> result = example_function(1, 2)  # returns False
    """
    if arg1 == arg2:
        return True
    return False
```

#### Google-style with type hints

This example shows both a Google-style docstring and argument and return type hints, though both are not required, one can be used without the other.

```python
def example_function(arg1: int, arg2: int = 4) -> bool:
    """
    This example shows both a Google-style docstring and argument and return type hints, though both are not required;
    one can be used without the other.

    Args:
        arg1: The first argument.
        arg2: The second argument. Default value is 4.

    Returns:
        True if successful, False otherwise.

    Examples:
        >>> result = example_function(1, 2)  # returns False
    """
    if arg1 == arg2:
        return True
    return False
```

#### Single-line

Smaller or simpler functions can utilize a single-line docstring. Note the docstring must use 3 double-quotes, and be a complete sentence starting with a capital letter and ending with a period.

```python
def example_small_function(arg1: int, arg2: int = 4) -> bool:
    """Example function that demonstrates a single-line docstring."""
    return arg1 == arg2
```

### GitHub Actions CI Tests

Before your pull request can be merged, all GitHub Actions [Continuous Integration](https://docs.ultralytics.com/help/CI/) (CI) tests must pass. These tests include linting, unit tests, and other checks to ensure that your changes meet the quality standards of the project. Make sure to review the output of the GitHub Actions and fix any issues

## Reporting Bugs

We appreciate bug reports as they play a crucial role in maintaining the project's quality. When reporting bugs it is important to provide a [Minimum Reproducible Example](https://docs.ultralytics.com/help/minimum_reproducible_example/): a clear, concise code example that replicates the issue. This helps in quick identification and resolution of the bug.

## License

Ultralytics embraces the [GNU Affero General Public License v3.0 (AGPL-3.0)](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) for its repositories, promoting openness, transparency, and collaborative enhancement in software development. This strong copyleft license ensures that all users and developers retain the freedom to use, modify, and share the software. It fosters community collaboration, ensuring that any improvements remain accessible to all.

Users and developers are encouraged to familiarize themselves with the terms of AGPL-3.0 to contribute effectively and ethically to the Ultralytics open-source community.

## Conclusion

Thank you for your interest in contributing to [Ultralytics open-source](https://github.com/ultralytics) YOLO projects. Your participation is crucial in shaping the future of our software and fostering a community of innovation and collaboration. Whether you're improving code, reporting bugs, or suggesting features, your contributions make a significant impact.

We look forward to seeing your ideas in action and appreciate your commitment to advancing object detection technology. Let's continue to grow and innovate together in this exciting open-source journey. Happy coding! 🚀🌟
