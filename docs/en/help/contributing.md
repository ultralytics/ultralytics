---
comments: true
description: Learn how to contribute to Ultralytics YOLO open-source repositories. Follow guidelines for pull requests, code of conduct, and bug reporting.
keywords: Ultralytics, YOLO, open-source, contribution, pull request, code of conduct, bug reporting, GitHub, CLA, Google-style docstrings
---

# Contributing to Ultralytics Open-Source Projects

Welcome! We're thrilled that you're considering contributing to our [Ultralytics](https://www.ultralytics.com/) [open-source](https://github.com/ultralytics) projects. Your involvement not only helps enhance the quality of our repositories but also benefits the entire community. This guide provides clear guidelines and best practices to help you get started.

<a href="https://github.com/ultralytics/ultralytics/graphs/contributors">
<img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-open-source-contributors.avif" alt="Ultralytics open-source contributors"></a>

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Contributing via Pull Requests](#contributing-via-pull-requests)
    - [CLA Signing](#cla-signing)
    - [Google-Style Docstrings](#google-style-docstrings)
    - [GitHub Actions CI Tests](#github-actions-ci-tests)
3. [Reporting Bugs](#reporting-bugs)
4. [License](#license)
5. [Conclusion](#conclusion)
6. [FAQ](#faq)

## Code of Conduct

To ensure a welcoming and inclusive environment for everyone, all contributors must adhere to our [Code of Conduct](https://docs.ultralytics.com/help/code_of_conduct/). Respect, kindness, and professionalism are at the heart of our community.

## Contributing via Pull Requests

We greatly appreciate contributions in the form of pull requests. To make the review process as smooth as possible, please follow these steps:

1. **[Fork the repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo):** Start by forking the Ultralytics YOLO repository to your GitHub account.

2. **[Create a branch](https://docs.github.com/en/desktop/making-changes-in-a-branch/managing-branches-in-github-desktop):** Create a new branch in your forked repository with a clear, descriptive name that reflects your changes.

3. **Make your changes:** Ensure your code adheres to the project's style guidelines and does not introduce any new errors or warnings.

4. **[Test your changes](https://github.com/ultralytics/ultralytics/tree/main/tests):** Before submitting, test your changes locally to confirm they work as expected and don't cause any new issues.

5. **[Commit your changes](https://docs.github.com/en/desktop/making-changes-in-a-branch/committing-and-reviewing-changes-to-your-project-in-github-desktop):** Commit your changes with a concise and descriptive commit message. If your changes address a specific issue, include the issue number in your commit message.

6. **[Create a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request):** Submit a pull request from your forked repository to the main Ultralytics YOLO repository. Provide a clear and detailed explanation of your changes and how they improve the project.

### CLA Signing

Before we can merge your pull request, you must sign our [Contributor License Agreement (CLA)](https://docs.ultralytics.com/help/CLA/). This legal agreement ensures that your contributions are properly licensed, allowing the project to continue being distributed under the AGPL-3.0 license.

After submitting your pull request, the CLA bot will guide you through the signing process. To sign the CLA, simply add a comment in your PR stating:

```
I have read the CLA Document and I sign the CLA
```

### Google-Style Docstrings

When adding new functions or classes, please include [Google-style docstrings](https://google.github.io/styleguide/pyguide.html). These docstrings provide clear, standardized documentation that helps other developers understand and maintain your code.

!!! example "Example Docstrings"

    === "Google-style"

         This example illustrates a Google-style docstring. Ensure that both input and output `types` are always enclosed in parentheses, e.g., `(bool)`.

         ```python
         def example_function(arg1, arg2=4):
             """
             Example function demonstrating Google-style docstrings.

             Args:
                 arg1 (int): The first argument.
                 arg2 (int): The second argument, with a default value of 4.

             Returns:
                 (bool): True if successful, False otherwise.

             Examples:
                 >>> result = example_function(1, 2)  # returns False
             """
             if arg1 == arg2:
                 return True
             return False
         ```

    === "Google-style with type hints"

         This example includes both a Google-style docstring and type hints for arguments and returns, though using either independently is also acceptable.

         ```python
         def example_function(arg1: int, arg2: int = 4) -> bool:
             """
             Example function demonstrating Google-style docstrings.

             Args:
                 arg1: The first argument.
                 arg2: The second argument, with a default value of 4.

             Returns:
                 True if successful, False otherwise.

             Examples:
                 >>> result = example_function(1, 2)  # returns False
             """
             if arg1 == arg2:
                 return True
             return False
         ```

    === "Single-line"

         For smaller or simpler functions, a single-line docstring may be sufficient. The docstring must use three double-quotes, be a complete sentence, start with a capital letter, and end with a period.

         ```python
         def example_small_function(arg1: int, arg2: int = 4) -> bool:
             """Example function with a single-line docstring."""
             return arg1 == arg2
         ```

### GitHub Actions CI Tests

All pull requests must pass the GitHub Actions [Continuous Integration](https://docs.ultralytics.com/help/CI/) (CI) tests before they can be merged. These tests include linting, unit tests, and other checks to ensure that your changes meet the project's quality standards. Review the CI output and address any issues that arise.

## Reporting Bugs

We highly value bug reports as they help us maintain the quality of our projects. When reporting a bug, please provide a [Minimum Reproducible Example](https://docs.ultralytics.com/help/minimum_reproducible_example/)—a simple, clear code example that consistently reproduces the issue. This allows us to quickly identify and resolve the problem.

## License

Ultralytics uses the [GNU Affero General Public License v3.0 (AGPL-3.0)](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) for its repositories. This license promotes openness, transparency, and collaborative improvement in software development. It ensures that all users have the freedom to use, modify, and share the software, fostering a strong community of collaboration and innovation.

We encourage all contributors to familiarize themselves with the terms of the AGPL-3.0 license to contribute effectively and ethically to the Ultralytics open-source community.

## Conclusion

Thank you for your interest in contributing to [Ultralytics](https://www.ultralytics.com/) [open-source](https://github.com/ultralytics) YOLO projects. Your participation is essential in shaping the future of our software and building a vibrant community of innovation and collaboration. Whether you're enhancing code, reporting bugs, or suggesting new features, your contributions are invaluable.

We're excited to see your ideas come to life and appreciate your commitment to advancing object detection technology. Together, let's continue to grow and innovate in this exciting open-source journey. Happy coding! 🚀🌟

## FAQ

### Why should I contribute to Ultralytics YOLO open-source repositories?

Contributing to Ultralytics YOLO open-source repositories improves the software, making it more robust and feature-rich for the entire community. Contributions can include code enhancements, bug fixes, documentation improvements, and new feature implementations. Additionally, contributing allows you to collaborate with other skilled developers and experts in the field, enhancing your own skills and reputation. For details on how to get started, refer to the [Contributing via Pull Requests](#contributing-via-pull-requests) section.

### How do I sign the Contributor License Agreement (CLA) for Ultralytics YOLO?

To sign the Contributor License Agreement (CLA), follow the instructions provided by the CLA bot after submitting your pull request. This process ensures that your contributions are properly licensed under the AGPL-3.0 license, maintaining the legal integrity of the open-source project. Add a comment in your pull request stating:

```
I have read the CLA Document and I sign the CLA.
```

For more information, see the [CLA Signing](#cla-signing) section.

### What are Google-style docstrings, and why are they required for Ultralytics YOLO contributions?

Google-style docstrings provide clear, concise documentation for functions and classes, improving code readability and maintainability. These docstrings outline the function's purpose, arguments, and return values with specific formatting rules. When contributing to Ultralytics YOLO, following Google-style docstrings ensures that your additions are well-documented and easily understood. For examples and guidelines, visit the [Google-Style Docstrings](#google-style-docstrings) section.

### How can I ensure my changes pass the GitHub Actions CI tests?

Before your pull request can be merged, it must pass all GitHub Actions Continuous Integration (CI) tests. These tests include linting, unit tests, and other checks to ensure the code meets

the project's quality standards. Review the CI output and fix any issues. For detailed information on the CI process and troubleshooting tips, see the [GitHub Actions CI Tests](#github-actions-ci-tests) section.

### How do I report a bug in Ultralytics YOLO repositories?

To report a bug, provide a clear and concise [Minimum Reproducible Example](https://docs.ultralytics.com/help/minimum_reproducible_example/) along with your bug report. This helps developers quickly identify and fix the issue. Ensure your example is minimal yet sufficient to replicate the problem. For more detailed steps on reporting bugs, refer to the [Reporting Bugs](#reporting-bugs) section.
