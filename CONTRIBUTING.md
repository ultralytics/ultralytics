## Contributing to YOLOv8 🚀

We love your input! We want to make contributing to YOLOv8 as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing a new feature
- Becoming a maintainer

YOLOv8 works so well due to our combined community effort, and for every small improvement you contribute you will be
helping push the frontiers of what's possible in AI 😃!

## Submitting a Pull Request (PR) 🛠️

Submitting a PR is easy! This example shows how to submit a PR for updating `requirements.txt` in 4 steps:

### 1. Select File to Update

Select `requirements.txt` to update by clicking on it in GitHub.

<p align="center"><img width="800" alt="PR_step1" src="https://user-images.githubusercontent.com/26833433/122260847-08be2600-ced4-11eb-828b-8287ace4136c.png"></p>

### 2. Click 'Edit this file'

Button is in top-right corner.

<p align="center"><img width="800" alt="PR_step2" src="https://user-images.githubusercontent.com/26833433/122260844-06f46280-ced4-11eb-9eec-b8a24be519ca.png"></p>

### 3. Make Changes

Change `matplotlib` version from `3.2.2` to `3.3`.

<p align="center"><img width="800" alt="PR_step3" src="https://user-images.githubusercontent.com/26833433/122260853-0a87e980-ced4-11eb-9fd2-3650fb6e0842.png"></p>

### 4. Preview Changes and Submit PR

Click on the **Preview changes** tab to verify your updates. At the bottom of the screen select 'Create a **new branch**
for this commit', assign your branch a descriptive name such as `fix/matplotlib_version` and click the green **Propose
changes** button. All done, your PR is now submitted to YOLOv8 for review and approval 😃!

<p align="center"><img width="800" alt="PR_step4" src="https://user-images.githubusercontent.com/26833433/122260856-0b208000-ced4-11eb-8e8e-77b6151cbcc3.png"></p>

### PR recommendations

To allow your work to be integrated as seamlessly as possible, we advise you to:

- ✅ Verify your PR is **up-to-date** with `ultralytics/ultralytics` `main` branch. If your PR is behind you can update
  your code by clicking the 'Update branch' button or by running `git pull` and `git merge main` locally.

<p align="center"><img width="751" alt="Screenshot 2022-08-29 at 22 47 15" src="https://user-images.githubusercontent.com/26833433/187295893-50ed9f44-b2c9-4138-a614-de69bd1753d7.png"></p>

- ✅ Verify all YOLOv8 Continuous Integration (CI) **checks are passing**.

<p align="center"><img width="751" alt="Screenshot 2022-08-29 at 22 47 03" src="https://user-images.githubusercontent.com/26833433/187296922-545c5498-f64a-4d8c-8300-5fa764360da6.png"></p>

- ✅ Reduce changes to the absolute **minimum** required for your bug fix or feature addition. _"It is not daily increase
  but daily decrease, hack away the unessential. The closer to the source, the less wastage there is."_  — Bruce Lee

### Docstrings

Not all functions or classes require docstrings but when they do, we
follow [google-style docstrings format](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
Here is an example:

```python
"""
    What the function does. Performs NMS on given detection predictions.

    Args:
        arg1: The description of the 1st argument
        arg2: The description of the 2nd argument

    Returns:
        What the function returns. Empty if nothing is returned.

    Raises:
        Exception Class: When and why this exception can be raised by the function.
"""
```

## Submitting a Bug Report 🐛

If you spot a problem with YOLOv8 please submit a Bug Report!

For us to start investigating a possible problem we need to be able to reproduce it ourselves first. We've created a few
short guidelines below to help users provide what we need in order to get started.

When asking a question, people will be better able to provide help if you provide **code** that they can easily
understand and use to **reproduce** the problem. This is referred to by community members as creating
a [minimum reproducible example](https://docs.ultralytics.com/help/minimum_reproducible_example/). Your code that reproduces
the problem should be:

- ✅ **Minimal** – Use as little code as possible that still produces the same problem
- ✅ **Complete** – Provide **all** parts someone else needs to reproduce your problem in the question itself
- ✅ **Reproducible** – Test the code you're about to provide to make sure it reproduces the problem

In addition to the above requirements, for [Ultralytics](https://ultralytics.com/) to provide assistance your code
should be:

- ✅ **Current** – Verify that your code is up-to-date with current
  GitHub [main](https://github.com/ultralytics/ultralytics/tree/main) branch, and if necessary `git pull` or `git clone`
  a new copy to ensure your problem has not already been resolved by previous commits.
- ✅ **Unmodified** – Your problem must be reproducible without any modifications to the codebase in this
  repository. [Ultralytics](https://ultralytics.com/) does not provide support for custom code ⚠️.

If you believe your problem meets all of the above criteria, please close this issue and raise a new one using the 🐛
**Bug Report** [template](https://github.com/ultralytics/ultralytics/issues/new/choose) and providing
a [minimum reproducible example](https://docs.ultralytics.com/help/minimum_reproducible_example/) to help us better
understand and diagnose your problem.

## License

By contributing, you agree that your contributions will be licensed under
the [AGPL-3.0 license](https://choosealicense.com/licenses/agpl-3.0/)
