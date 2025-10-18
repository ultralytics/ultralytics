---
comments: true
description: Learn how Ultralytics leverages Continuous Integration (CI) for maintaining high-quality code. Explore our CI tests and the status of these tests for our repositories.
keywords: continuous integration, software development, CI tests, Ultralytics repositories, high-quality code, Docker Deployment, Broken Links, CodeQL, PyPi Publishing
---

# Continuous Integration (CI)

Continuous Integration (CI) is an essential aspect of software development which involves integrating changes and testing them automatically. CI allows us to maintain high-quality code by catching issues early and often in the development process. At Ultralytics, we use various CI tests to ensure the quality and integrity of our codebase.

## CI Actions

Here's a brief description of our CI actions:

- **CI:** This is our primary CI test that involves running unit tests, linting checks, and sometimes more comprehensive tests depending on the repository.
- **Docker Deployment:** This test checks the deployment of the project using Docker to ensure the Dockerfile and related scripts are working correctly.
- **Broken Links:** This test scans the codebase for any broken or dead links in our markdown or HTML files.
- **CodeQL:** CodeQL is a tool from GitHub that performs semantic analysis on our code, helping to find potential security vulnerabilities and maintain high-quality code.
- **PyPi Publishing:** This test checks if the project can be packaged and published to PyPi without any errors.

### CI Results

Below is the table showing the status of these CI tests for our main repositories:

| Repository                                                | CI                                                                                                                                                                        | Docker Deployment                                                                                                                                                                        | Broken Links                                                                                                                                                                      | CodeQL                                                                                                                                                                          | PyPi and Docs Publishing                                                                                                                                                                                      |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [yolov3](https://github.com/ultralytics/yolov3)           | [![YOLOv3 CI](https://github.com/ultralytics/yolov3/actions/workflows/ci-testing.yml/badge.svg)](https://github.com/ultralytics/yolov3/actions/workflows/ci-testing.yml)  | [![Publish Docker Images](https://github.com/ultralytics/yolov3/actions/workflows/docker.yml/badge.svg)](https://github.com/ultralytics/yolov3/actions/workflows/docker.yml)             | [![Check Broken links](https://github.com/ultralytics/yolov3/actions/workflows/links.yml/badge.svg)](https://github.com/ultralytics/yolov3/actions/workflows/links.yml)           | [![CodeQL](https://github.com/ultralytics/yolov3/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/ultralytics/yolov3/actions/workflows/codeql-analysis.yml) |                                                                                                                                                                                                               |
| [yolov5](https://github.com/ultralytics/yolov5)           | [![YOLOv5 CI](https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml/badge.svg)](https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml)  | [![Publish Docker Images](https://github.com/ultralytics/yolov5/actions/workflows/docker.yml/badge.svg)](https://github.com/ultralytics/yolov5/actions/workflows/docker.yml)             | [![Check Broken links](https://github.com/ultralytics/yolov5/actions/workflows/links.yml/badge.svg)](https://github.com/ultralytics/yolov5/actions/workflows/links.yml)           | [![CodeQL](https://github.com/ultralytics/yolov5/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/ultralytics/yolov5/actions/workflows/codeql-analysis.yml) |                                                                                                                                                                                                               |
| [ultralytics](https://github.com/ultralytics/ultralytics) | [![ultralytics CI](https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml/badge.svg)](https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml) | [![Publish Docker Images](https://github.com/ultralytics/ultralytics/actions/workflows/docker.yaml/badge.svg)](https://github.com/ultralytics/ultralytics/actions/workflows/docker.yaml) | [![Check Broken links](https://github.com/ultralytics/ultralytics/actions/workflows/links.yml/badge.svg)](https://github.com/ultralytics/ultralytics/actions/workflows/links.yml) | [![CodeQL](https://github.com/ultralytics/ultralytics/actions/workflows/codeql.yaml/badge.svg)](https://github.com/ultralytics/ultralytics/actions/workflows/codeql.yaml)       | [![Publish to PyPI and Deploy Docs](https://github.com/ultralytics/ultralytics/actions/workflows/publish.yml/badge.svg)](https://github.com/ultralytics/ultralytics/actions/workflows/publish.yml)            |
| [hub](https://github.com/ultralytics/hub)                 | [![HUB CI](https://github.com/ultralytics/hub/actions/workflows/ci.yaml/badge.svg)](https://github.com/ultralytics/hub/actions/workflows/ci.yaml)                         |                                                                                                                                                                                          | [![Check Broken links](https://github.com/ultralytics/hub/actions/workflows/links.yml/badge.svg)](https://github.com/ultralytics/hub/actions/workflows/links.yml)                 |                                                                                                                                                                                 |                                                                                                                                                                                                               |
| [docs](https://github.com/ultralytics/docs)               |                                                                                                                                                                           |                                                                                                                                                                                          |                                                                                                                                                                                   |                                                                                                                                                                                 | [![pages-build-deployment](https://github.com/ultralytics/docs/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/ultralytics/docs/actions/workflows/pages/pages-build-deployment) |

Each badge shows the status of the last run of the corresponding CI test on the `main` branch of the respective repository. If a test fails, the badge will display a "failing" status, and if it passes, it will display a "passing" status.

If you notice a test failing, it would be a great help if you could report it through a GitHub issue in the respective repository.

Remember, a successful CI test does not mean that everything is perfect. It is always recommended to manually review the code before deployment or merging changes.

## Code Coverage

Code coverage is a metric that represents the percentage of your codebase that is executed when your tests run. It provides insight into how well your tests exercise your code and can be crucial in identifying untested parts of your application. A high code coverage percentage is often associated with a lower likelihood of bugs. However, it's essential to understand that code coverage doesn't guarantee the absence of defects. It merely indicates which parts of the code have been executed by the tests.

### Integration with [codecov.io](https://codecov.io/)

At Ultralytics, we have integrated our repositories with [codecov.io](https://codecov.io/), a popular online platform for measuring and visualizing code coverage. Codecov provides detailed insights, coverage comparisons between commits, and visual overlays directly on your code, indicating which lines were covered.

By integrating with Codecov, we aim to maintain and improve the quality of our code by focusing on areas that might be prone to errors or need further testing.

### Coverage Results

To quickly get a glimpse of the code coverage status of the `ultralytics` python package, we have included a badge and and sunburst visual of the `ultralytics` coverage results. These images show the percentage of code covered by our tests, offering an at-a-glance metric of our testing efforts. For full details please see https://codecov.io/github/ultralytics/ultralytics.

| Repository                                                | Code Coverage                                                                                                                                           |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [ultralytics](https://github.com/ultralytics/ultralytics) | [![codecov](https://codecov.io/gh/ultralytics/ultralytics/branch/main/graph/badge.svg?token=HHW7IIVFVY)](https://codecov.io/gh/ultralytics/ultralytics) |

In the sunburst graphic below, the inner-most circle is the entire project, moving away from the center are folders then, finally, a single file. The size and color of each slice is representing the number of statements and the coverage, respectively.

<a href="https://codecov.io/github/ultralytics/ultralytics">
    <img src="https://codecov.io/gh/ultralytics/ultralytics/branch/main/graphs/sunburst.svg?token=HHW7IIVFVY" alt="Ultralytics Codecov Image">
</a>
