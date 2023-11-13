<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/supervision/roboflow-supervision-banner.png?ik-sdk-version=javascript-1.4.3&updatedAt=1674062891088"
      >
    </a>
  </p>
</div>

## 👋 Hello

We write your reusable computer vision tools. Whether you need to load your dataset from your hard drive, draw detections on an image or video, or count how many detections are in a zone. You can count on us!

## 💻 Install

You can install `supervision` with pip in a
[**3.11>=Python>=3.8**](https://www.python.org/) environment.

!!! example "pip install (recommended)"

    === "headless"
        The headless installation of `supervision` is designed for environments where graphical user interfaces (GUI) are not needed, making it more lightweight and suitable for server-side applications.

        ```bash
        pip install supervision
        ```

    === "desktop"
        If you require the full version of `supervision` with GUI support you can install the desktop version. This version includes the GUI components of OpenCV, allowing you to display images and videos on the screen.

        ```bash
        pip install supervision[desktop]
        ```

!!! example "git clone (for development)"

    === "virtualenv"

        ```bash
        # clone repository and navigate to root directory
        git clone https://github.com/roboflow/supervision.git
        cd supervision

        # setup python environment and activate it
        python3 -m venv venv
        source venv/bin/activate
        pip install --upgrade pip

        # headless install
        pip install -e "."

        # desktop install
        pip install -e ".[desktop]"
        ```

    === "poetry"

        ```bash
        # clone repository and navigate to root directory
        git clone https://github.com/roboflow/supervision.git
        cd supervision

        # setup python environment and activate it
        poetry env use python3.10
        poetry shell

        # headless install
        poetry install

        # desktop install
        poetry install --extras "desktop"
        ```
