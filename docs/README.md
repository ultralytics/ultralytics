<a href="https://www.ultralytics.com" target="_blank"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# 📚 Ultralytics Docs

Welcome to Ultralytics Docs, your comprehensive resource for understanding and utilizing our state-of-the-art [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) tools and models, including [Ultralytics YOLO](https://docs.ultralytics.com/models/yolo26). These documents are actively maintained and deployed to [https://docs.ultralytics.com](https://docs.ultralytics.com) for easy access.

[![Ultralytics Docs](https://github.com/ultralytics/ultralytics/actions/workflows/docs.yml/badge.svg)](https://github.com/ultralytics/ultralytics/actions/workflows/docs.yml)
[![Check Broken links](https://github.com/ultralytics/ultralytics/actions/workflows/links.yml/badge.svg)](https://github.com/ultralytics/ultralytics/actions/workflows/links.yml)
[![Ultralytics Actions](https://github.com/ultralytics/ultralytics/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/ultralytics/actions/workflows/format.yml)

<a href="https://discord.com/invite/ultralytics"><img alt="Discord" src="https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue"></a> <a href="https://community.ultralytics.com"><img alt="Ultralytics Forums" src="https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue"></a> <a href="https://www.reddit.com/r/ultralytics/"><img alt="Ultralytics Reddit" src="https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue"></a>

## 🛠️ Installation

[![PyPI - Version](https://img.shields.io/pypi/v/ultralytics?logo=pypi&logoColor=white)](https://pypi.org/project/ultralytics/)
[![Downloads](https://static.pepy.tech/badge/ultralytics)](https://clickpy.clickhouse.com/dashboard/ultralytics)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ultralytics?logo=python&logoColor=gold)](https://pypi.org/project/ultralytics/)

To install the `ultralytics` package in developer mode, which allows you to modify the source code directly, ensure you have [Git](https://git-scm.com/), [Python](https://www.python.org/), and [uv](https://docs.astral.sh/uv/getting-started/installation/) installed on your system. The package supports Python 3.8 or later; building the docs requires Python 3.10 or later. Then, follow these steps:

1.  Clone the `ultralytics` repository to your local machine using Git:

    ```bash
    git clone https://github.com/ultralytics/ultralytics.git
    ```

2.  Navigate to the cloned repository's root directory:

    ```bash
    cd ultralytics
    ```

3.  Install the package in editable mode (`-e`) along with its development dependencies (`[dev]`, which include the `zensical` and `minijinja` docs tooling) using [uv](https://docs.astral.sh/uv/):

    ```bash
    uv pip install -e ".[dev]"
    ```

    This command installs the `ultralytics` package such that changes to the source code are immediately reflected in your environment, ideal for development.

## 🗂️ Documentation Layout

- `docs/en/` holds every page; the root `mkdocs.yml` owns navigation and Markdown configuration.
- `docs/macros/` holds shared tables (training, prediction, and export arguments, performance tables, and more) that pages include with `{% include %}`; edit the macro, never a copy in a consuming page.
- `docs/en/reference/` is generated from Python docstrings by `docs/build_reference.py`, so API documentation changes belong in the docstrings. Run `python docs/build_reference.py` after adding, removing, or renaming public APIs.
- Model comparison pages under `docs/en/compare/` are imported from the [ultralytics/docs](https://github.com/ultralytics/docs) repository during the full build.

See [`docs/AGENTS.md`](AGENTS.md) for detailed contributor conventions.

## 🚀 Building and Serving Locally

### Full Validation (Recommended)

The `build_docs.py` script prepares the complete documentation tree by rendering Jinja macros, generating API reference pages, and pulling in model comparison pages (this step needs network access). It then runs `zensical build --strict`, matching the documentation validation performed in CI, and restores `docs/en/` and `docs/macros/` afterwards.

```bash
python docs/build_docs.py
```

The script builds the site into the `site/` directory. Preview it with `python -m http.server --directory site`; the local output intentionally omits production-owned banners, analytics, comments, and other site chrome.

### Quick Preview

For quick edits to pages that don't use `{% include %}` macros, you can use `zensical serve` from the repository root for faster iteration with live reloading:

```bash
zensical serve
```

Note that `zensical serve` does **not** render Jinja macros, generate API reference pages, or include comparison pages, so some pages (train, predict, val, export, tasks, and others) will display raw `{% include %}` tags instead of their actual content. Use the full build above to verify these pages.

### Formatting

Ultralytics Actions formats docs Markdown automatically on pull requests. To match it locally, run prettier with a 4-space tab width, which the docs Markdown dialect requires for list continuation:

```bash
npx prettier@3.9.9 --tab-width 4 --print-width 120 --write "docs/**/*.md"
```

## 📤 Deploying Your Documentation Site

The CI pipeline in `.github/workflows/docs.yml` regenerates the API reference, lints docstrings, and validates the full documentation build with Zensical on every pull request. After pushes to `main` that change `docs/`, `mkdocs.yml`, or `ultralytics/`, it triggers Ultralytics' centralized publisher, which renders and deploys [docs.ultralytics.com](https://docs.ultralytics.com).

## 💡 Contribute

We deeply value contributions from the open-source community to enhance Ultralytics projects. Your input helps drive innovation! Please review our [Contributing Guide](https://docs.ultralytics.com/help/contributing) for detailed information on how to get involved. You can also share your feedback and ideas through our [Survey](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey). A heartfelt thank you 🙏 to all our contributors for their dedication and support!

[![Ultralytics open-source contributors](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/ultralytics/graphs/contributors)

We look forward to your contributions!

## 📜 License

Ultralytics Docs are available under two licensing options to accommodate different usage scenarios:

- **AGPL-3.0 License**: Ideal for students, researchers, and enthusiasts involved in academic pursuits and open collaboration. See the [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) file for full details. This license promotes sharing improvements back with the community.
- **Enterprise License**: For development and production use, this license enables seamless integration of Ultralytics software and [AI models](https://docs.ultralytics.com/models) into business products and services, including internal tools, automated workflows, and production deployments, bypassing the open-source requirements of AGPL-3.0. To get started, please contact us via [Ultralytics Licensing](https://www.ultralytics.com/license).

## ✉️ Contact

For bug reports, feature requests, and other issues related to the documentation, please use [GitHub Issues](https://github.com/ultralytics/ultralytics/issues). For discussions, questions, and community support, join the conversation with peers and the Ultralytics team on our [Discord server](https://discord.com/invite/ultralytics)!

<br>
<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
