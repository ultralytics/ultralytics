<a href="https://www.ultralytics.com/" target="_blank"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Contributing to Ultralytics Open-Source Projects

Welcome! We're thrilled that you're considering contributing to our [Ultralytics](https://www.ultralytics.com/) [open-source](https://github.com/ultralytics) projects. Your involvement not only helps enhance the quality of our repositories but also benefits the entire [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) community. This guide provides clear guidelines and best practices to help you get started.

[![Ultralytics open-source contributors](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/ultralytics/graphs/contributors)

## 🤝 Code of Conduct

To ensure a welcoming and inclusive environment for everyone, all contributors must adhere to our [Code of Conduct](https://docs.ultralytics.com/help/code-of-conduct/). **Respect**, **kindness**, and **professionalism** are at the heart of our community.

## 🚀 Contributing via Pull Requests

We greatly appreciate contributions in the form of [pull requests (PRs)](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests). To make the review process as smooth as possible, please follow these steps:

1. **[Fork the repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo):** Start by forking the relevant Ultralytics repository (e.g., [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)) to your GitHub account.
2. **[Create a branch](https://docs.github.com/en/desktop/making-changes-in-a-branch/managing-branches-in-github-desktop):** Create a new branch in your forked repository with a clear, descriptive name reflecting your changes (e.g., `fix-issue-123`, `add-feature-xyz`).
3. **Make your changes:** Implement your improvements or fixes. Ensure your code adheres to the project's style guidelines and doesn't introduce new errors or warnings.
4. **Test your changes:** Before submitting, test your changes locally to confirm they work as expected and don't cause [regressions](https://en.wikipedia.org/wiki/Software_regression). Add tests if you're introducing new functionality.
5. **[Commit your changes](https://docs.github.com/en/desktop/making-changes-in-a-branch/committing-and-reviewing-changes-to-your-project-in-github-desktop):** Commit your changes with concise and descriptive commit messages. If your changes address a specific issue, include the issue number (e.g., `Fix #123: Corrected calculation error.`).
6. **[Create a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request):** Submit a pull request from your branch to the `main` branch of the original Ultralytics repository. Provide a clear title and a detailed description explaining the purpose and scope of your changes.

### 📏 Contribution Scope & PR Size Guidelines

#### First-time Contributors

First-time contributors are expected to submit small, well-scoped pull requests. Large PRs (more than ~50 lines of change) are generally not accepted at this stage, except when the size is driven primarily by docstrings, documentation, or formatting rather than substantive code changes. Contributions should focus on speed improvements, bug fixes, documentation updates, and other minor issues. Feature PRs are not encouraged for first-time contributors. However, if you still wish to contribute a feature, follow the guidelines for outlined in the [Feature PRs](#feature-prs) section below.

#### Established Contributors

Pull requests from established contributors generally receive higher review priority. Actions and results are fundamental to the [Ultralytics Mission & Values](https://handbook.ultralytics.com/mission-vision-values/). There is no specific threshold to becoming an 'established contributor' as it's impossible to fit all individuals to the same standard. The Ultralytics Team notices those who make consistent, high-quality contributions that follow the Ultralytics standards.

Following our contributing guidance (this document) and [our Development Workflow](https://handbook.ultralytics.com/workflows/development/) is _the best_ way to improve your chances for your work to be reviewed, accepted, and/or recognized; this is not a guarantee. In addition, contributors with a strong track record of meaningful contributions to notable open-source projects may be treated as established contributors, even if they are technically first-time contributors to Ultralytics.

#### Feature PRs

Feature pull requests must be preceded by a feature request GitHub issue that has been sufficiently discussed and explicitly approved by the maintainers. This process helps avoid unnecessary effort on changes that are unlikely to be merged. Even after approval, feature PRs are expected to remain well-scoped and focused. All feature contributions are evaluated with attention to long-term maintenance costs and their overall usefulness to the Ultralytics user base.

#### PR Size and Review Time

The larger the proposed changes to the code, the longer the review process will take. Smaller, narrowly-scoped PRs that align with the style and structure of the Ultralytics codebase have significantly higher likelihood of timely review and merge.

### 📝 CLA Signing

Before we can merge your pull request, you must sign our [Contributor License Agreement (CLA)](https://docs.ultralytics.com/help/CLA/). This legal agreement ensures that your contributions are properly licensed, allowing the project to continue being distributed under the [AGPL-3.0 license](https://www.ultralytics.com/legal/agpl-3-0-software-license).

After submitting your pull request, the CLA bot will guide you through the signing process. To sign the CLA, simply add a comment in your PR stating:

```text
I have read the CLA Document and I sign the CLA
```

### ✍️ Google-Style Docstrings

When adding new functions or classes, please include [Google-style docstrings](https://google.github.io/styleguide/pyguide.html). These docstrings provide clear, standardized documentation that helps other developers understand and maintain your code.

#### Example Google-style

This example illustrates a Google-style docstring. Ensure that both input and output `types` are always enclosed in parentheses, e.g., `(bool)`.

```python
def example_function(arg1, arg2=4):
    """Example function demonstrating Google-style docstrings.

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

#### Example Google-style with type hints

This example includes both a Google-style docstring and [type hints](https://docs.python.org/3/library/typing.html) for arguments and returns, though using either independently is also acceptable.

```python
def example_function(arg1: int, arg2: int = 4) -> bool:
    """Example function demonstrating Google-style docstrings.

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

#### Example Single-line

For smaller or simpler functions, a single-line docstring may be sufficient. The docstring must use three double-quotes, be a complete sentence, start with a capital letter, and end with a period.

```python
def example_small_function(arg1: int, arg2: int = 4) -> bool:
    """Example function with a single-line docstring."""
    return arg1 == arg2
```

### ✅ GitHub Actions CI Tests

All pull requests must pass the [GitHub Actions](https://github.com/features/actions) [Continuous Integration](https://docs.ultralytics.com/help/CI/) (CI) tests before they can be merged. These tests include linting, unit tests, and other checks to ensure that your changes meet the project's quality standards. Review the CI output and address any issues that arise.

## ✨ Best Practices for Code Contributions

When contributing code to Ultralytics projects, keep these best practices in mind:

- **Avoid code duplication:** Reuse existing code wherever possible and minimize unnecessary arguments.
- **Make smaller, focused changes:** Focus on targeted modifications rather than large-scale changes. Smaller, narrow-scope pull requests are easier to review, less error-prone, and have a higher chance of being merged.
- **Simplify when possible:** Look for opportunities to simplify the code or remove unnecessary parts.
- **Consider compatibility:** Before making changes, consider whether they might break existing code using Ultralytics.
- **Use consistent formatting:** Tools like [Ruff Formatter](https://github.com/astral-sh/ruff) can help maintain stylistic consistency.
- **Add appropriate tests:** Include [tests](https://docs.ultralytics.com/guides/model-testing/) for new features to ensure they work as expected. New tests should be added to existing test files rather than creating new test files. Tests do not need to be exhaustive, but must be able to reasonably confirm correct behavior and detect regressions.

## 👀 Reviewing Pull Requests

Reviewing pull requests is another valuable way to contribute. When reviewing PRs:

- **Check for unit tests:** Verify that the PR includes tests for new features or changes.
- **Review documentation updates:** Ensure [documentation](https://docs.ultralytics.com/) is updated to reflect changes.
- **Evaluate performance impact:** Consider how changes might affect [performance](https://docs.ultralytics.com/guides/yolo-performance-metrics/).
- **Verify CI tests:** Confirm all [Continuous Integration tests](https://docs.ultralytics.com/help/CI/) are passing.
- **Provide constructive feedback:** Offer specific, clear feedback about any issues or concerns.
- **Recognize effort:** Acknowledge the author's work to maintain a positive collaborative atmosphere.

### 🤖 Automated Review

All pull requests undergo an automatic review by Ultralytics Assistant. Each suggestion from Ultralytics Assistant includes an associated level of importance. Contributors are expected to review all suggestions and either apply them or explicitly explain in a PR comment why a given suggestion is not applicable or should not be adopted.

If needed, you may re-request a review from Ultralytics Assistant via the GitHub reviewers panel.

## 🐞 Reporting Bugs

We highly value bug reports as they help us improve the quality and reliability of our projects. When reporting a bug via [GitHub Issues](https://github.com/ultralytics/ultralytics/issues):

- **Check existing issues:** Search first to see if the bug has already been reported.
- **Provide a [Minimum Reproducible Example](https://docs.ultralytics.com/help/minimum-reproducible-example/):** Create a small, self-contained code snippet that consistently reproduces the issue. This is crucial for efficient debugging.
- **Describe the environment:** Specify your operating system, Python version, relevant library versions (e.g., [`torch`](https://pytorch.org/), [`ultralytics`](https://github.com/ultralytics/ultralytics)), and hardware ([CPU](https://en.wikipedia.org/wiki/Central_processing_unit)/[GPU](https://www.ultralytics.com/glossary/gpu-graphics-processing-unit)).
- **Explain expected vs. actual behavior:** Clearly state what you expected to happen and what actually occurred. Include any error messages or tracebacks.

## 📜 License

Ultralytics uses the [GNU Affero General Public License v3.0 (AGPL-3.0)](https://www.ultralytics.com/legal/agpl-3-0-software-license) for its repositories. This license promotes [openness](https://en.wikipedia.org/wiki/Openness), [transparency](https://www.ultralytics.com/glossary/transparency-in-ai), and [collaborative improvement](https://en.wikipedia.org/wiki/Collaborative_software) in software development. It ensures that all users have the freedom to use, modify, and share the software, fostering a strong community of collaboration and innovation.

We encourage all contributors to familiarize themselves with the terms of the [AGPL-3.0 license](https://opensource.org/license/agpl-v3) to contribute effectively and ethically to the Ultralytics open-source community.

## 🌍 Open-Sourcing Your YOLO Project Under AGPL-3.0

Using Ultralytics YOLO models or code in your project? The [AGPL-3.0 license](https://opensource.org/license/agpl-v3) requires that your entire derivative work also be open-sourced under AGPL-3.0. This ensures modifications and larger projects built upon open-source foundations remain open.

### Why AGPL-3.0 Compliance Matters

- **Keeps Software Open:** Ensures that improvements and derivative works benefit the community.
- **Legal Requirement:** Using AGPL-3.0 licensed code binds your project to its terms.
- **Fosters Collaboration:** Encourages sharing and transparency.

If you prefer not to open-source your project, consider obtaining an [Enterprise License](https://www.ultralytics.com/license).

### How to Comply with AGPL-3.0

Complying means making the **complete corresponding source code** of your project publicly available under the AGPL-3.0 license.

1. **Choose Your Starting Point:**
   - **Fork Ultralytics YOLO:** Directly fork the [Ultralytics YOLO repository](https://github.com/ultralytics/ultralytics) if building closely upon it.
   - **Use Ultralytics Template:** Start with the [Ultralytics template repository](https://github.com/ultralytics/template) for a clean, modular setup integrating YOLO.

2. **License Your Project:**
   - Add a `LICENSE` file containing the full text of the [AGPL-3.0 license](https://opensource.org/license/agpl-v3).
   - Add a notice at the top of each source file indicating the license.

3. **Publish Your Source Code:**
   - Make your **entire project's source code** publicly accessible (e.g., on GitHub). This includes:
     - The complete larger application or system that incorporates the YOLO model or code.
     - Any modifications made to the original Ultralytics YOLO code.
     - Scripts for training, validation, inference.
     - [Model weights](https://www.ultralytics.com/glossary/model-weights) if modified or fine-tuned.
     - [Configuration files](https://docs.ultralytics.com/usage/cfg/), environment setups (`requirements.txt`, [`Dockerfiles`](https://docs.docker.com/reference/dockerfile/)).
     - Backend and frontend code if it's part of a [web application](https://en.wikipedia.org/wiki/Web_application).
     - Any [third-party libraries](<https://en.wikipedia.org/wiki/Library_(computing)#Third-party>) you've modified.
     - [Training data](https://www.ultralytics.com/glossary/training-data) if required to run/retrain _and_ redistributable.

4. **Document Clearly:**
   - Update your `README.md` to state that the project is licensed under AGPL-3.0.
   - Include clear instructions on how to set up, build, and run your project from the source code.
   - Attribute Ultralytics YOLO appropriately, linking back to the [original repository](https://github.com/ultralytics/ultralytics). Example:
     ```markdown
     This project utilizes code from [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), licensed under AGPL-3.0.
     ```

### Example Repository Structure

Refer to the [Ultralytics Template Repository](https://github.com/ultralytics/template) for a practical example structure:

```
my-yolo-project/
│
├── LICENSE               # Full AGPL-3.0 license text
├── README.md             # Project description, setup, usage, license info & attribution
├── pyproject.toml        # Dependencies (or requirements.txt)
├── scripts/              # Training/inference scripts
│   └── train.py
├── src/                  # Your project's source code
│   ├── __init__.py
│   ├── data_loader.py
│   └── model_wrapper.py  # Code interacting with YOLO
├── tests/                # Unit/integration tests
├── configs/              # YAML/JSON config files
├── docker/               # Dockerfiles, if used
│   └── Dockerfile
└── .github/              # GitHub specific files (e.g., workflows for CI)
    └── workflows/
        └── ci.yml
```

By following these guidelines, you ensure compliance with AGPL-3.0, supporting the open-source ecosystem that enables powerful tools like Ultralytics YOLO.

## 🎉 Conclusion

Thank you for your interest in contributing to [Ultralytics](https://www.ultralytics.com/) [open-source](https://github.com/ultralytics) YOLO projects. Your participation is essential in shaping the future of our software and building a vibrant community of innovation and collaboration. Whether you're enhancing code, reporting bugs, or suggesting new features, your contributions are invaluable.

We're excited to see your ideas come to life and appreciate your commitment to advancing [object detection](https://www.ultralytics.com/glossary/object-detection) technology. Together, let's continue to grow and innovate in this exciting open-source journey. Happy coding! 🚀🌟

## FAQ

### Why should I contribute to Ultralytics YOLO open-source repositories?

Contributing to Ultralytics YOLO open-source repositories improves the software, making it more robust and feature-rich for the entire community. Contributions can include code enhancements, bug fixes, documentation improvements, and new feature implementations. Additionally, contributing allows you to collaborate with other skilled developers and experts in the field, enhancing your own skills and reputation. For details on how to get started, refer to the [Contributing via Pull Requests](#-contributing-via-pull-requests) section.

### How do I sign the Contributor License Agreement (CLA) for Ultralytics YOLO?

To sign the Contributor License Agreement (CLA), follow the instructions provided by the CLA bot after submitting your pull request. This process ensures that your contributions are properly licensed under the AGPL-3.0 license, maintaining the legal integrity of the open-source project. Add a comment in your pull request stating:

```text
I have read the CLA Document and I sign the CLA
```

For more information, see the [CLA Signing](#-cla-signing) section.

### What are Google-style docstrings, and why are they required for Ultralytics YOLO contributions?

Google-style docstrings provide clear, concise documentation for functions and classes, improving code readability and maintainability. These docstrings outline the function's purpose, arguments, and return values with specific formatting rules. When contributing to Ultralytics YOLO, following Google-style docstrings ensures that your additions are well-documented and easily understood. For examples and guidelines, visit the [Google-Style Docstrings](#-google-style-docstrings) section.

### How can I ensure my changes pass the GitHub Actions CI tests?

Before your pull request can be merged, it must pass all GitHub Actions Continuous Integration (CI) tests. These tests include linting, unit tests, and other checks to ensure the code meets the project's quality standards. Review the CI output and fix any issues. For detailed information on the CI process and troubleshooting tips, see the [GitHub Actions CI Tests](#-github-actions-ci-tests) section.

### How do I report a bug in Ultralytics YOLO repositories?

To report a bug, provide a clear and concise [Minimum Reproducible Example](https://docs.ultralytics.com/help/minimum-reproducible-example/) along with your bug report. This helps developers quickly identify and fix the issue. Ensure your example is minimal yet sufficient to replicate the problem. For more detailed steps on reporting bugs, refer to the [Reporting Bugs](#-reporting-bugs) section.

### What does the AGPL-3.0 license mean if I use Ultralytics YOLO in my own project?

If you use Ultralytics YOLO code or models (licensed under AGPL-3.0) in your project, the AGPL-3.0 license requires that your entire project (the derivative work) must also be licensed under AGPL-3.0 and its complete source code must be made publicly available. This ensures that the open-source nature of the software is preserved throughout its derivatives. If you cannot meet these requirements, you need to obtain an [Enterprise License](https://www.ultralytics.com/license). See the [Open-Sourcing Your Project](#-open-sourcing-your-yolo-project-under-agpl-30) section for details.
