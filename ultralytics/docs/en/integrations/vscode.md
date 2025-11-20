---
comments: true
description: An overview of how the Ultralytics-Snippets extension for Visual Studio Code can help developers accelerate their work with the Ultralytics Python package.
keywords: Visual Studio Code, VS Code, deep learning, convolutional neural networks, computer vision, Python, code snippets, Ultralytics, developer productivity, machine learning, YOLO, developers, productivity, efficiency, learning, programming, IDE, code editor, developer utilities, programming tools
---

# Ultralytics VS Code Extension

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/EXIpyYVEjoI"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to use Ultralytics Visual Studio Code Extension | Ready-to-Use Code Snippets | Ultralytics YOLO 🎉
</p>

## Features and Benefits

✅ Are you a data scientist or [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) engineer building computer vision applications with Ultralytics?

✅ Do you despise writing the same blocks of code repeatedly?

✅ Are you always forgetting the arguments or default values for the [export](../modes/export.md), [predict](../modes/predict.md), [train](../modes/train.md), [track](../modes/track.md), or [val](../modes/val.md) methods?

✅ Looking to get started with Ultralytics and wish you had an _easier_ way to reference or run code examples?

✅ Want to speed up your development cycle when working with Ultralytics?

If you use Visual Studio Code and answered 'yes' to any of the above, then the Ultralytics-snippets extension for VS Code is here to help! Read on to learn more about the extension, how to install it, and how to use it.

<p align="center">
  <br>
    <img src="https://github.com/ultralytics/docs/releases/download/0/snippet-prediction-preview.avif" alt="Snippet Prediction Preview">
  <br>
  Run example code using Ultralytics YOLO in under 20 seconds! 🚀
</p>

## Inspired by the Ultralytics Community

The inspiration to build this extension came from the Ultralytics Community. Questions from the Community around similar topics and examples fueled the development for this project. Additionally, many members of the Ultralytics team use VS Code to accelerate their own work ⚡.

## Why VS Code?

[Visual Studio Code](https://code.visualstudio.com/) is extremely popular with developers worldwide and has ranked most popular by the Stack Overflow Developer Survey in [2021](https://survey.stackoverflow.co/2021#section-most-popular-technologies-integrated-development-environment), [2022](https://survey.stackoverflow.co/2022/#section-most-popular-technologies-integrated-development-environment), [2023](https://survey.stackoverflow.co/2023/#section-most-popular-technologies-integrated-development-environment), and [2024](https://survey.stackoverflow.co/2024/technology#1-integrated-development-environment). Due to VS Code's high level of customization, built-in features, broad compatibility, and extensibility, it's no surprise that so many developers are using it. Given the popularity in the wider developer community and within the Ultralytics [Discord](https://discord.com/invite/ultralytics), [Discourse](https://community.ultralytics.com/), [Reddit](https://www.reddit.com/r/ultralytics/), and [GitHub](https://github.com/ultralytics) Communities, it made sense to build a VS Code extension to help streamline your workflow and boost your productivity.

Want to let us know what you use for developing code? Head over to our Discourse [community poll](https://community.ultralytics.com/t/what-do-you-use-to-write-code/89/1) and let us know! While you're there, maybe check out some of our favorite computer vision, machine learning, AI, and developer [memes](https://community.ultralytics.com/c/off-topic/memes-jokes/11), or even post your favorite!

## Installing the Extension

!!! note

    Any code environment that will allow for installing VS Code extensions _should be_ compatible with the Ultralytics-snippets extension. After publishing the extension, it was discovered that [neovim](https://neovim.io/) can be made compatible with VS Code extensions. To learn more see the [`neovim` install section](https://github.com/Burhan-Q/ultralytics-snippets?tab=readme-ov-file#use-with-neovim) of the Readme in the [Ultralytics-Snippets repository](https://github.com/Burhan-Q/ultralytics-snippets).

### Installing in VS Code

1. Navigate to the [Extensions menu in VS Code](https://code.visualstudio.com/docs/editor/extension-marketplace) or use the shortcut <kbd>Ctrl</kbd>+<kbd>Shift ⇑</kbd>+<kbd>x</kbd>, and search for Ultralytics-snippets.

2. Click the <kbd>Install</kbd> button.

<p align="center">
  <br>
    <img src="https://github.com/ultralytics/docs/releases/download/0/vs-code-extension-menu.avif" alt="VS Code extension menu">
  <br>
</p>

### Installing from the VS Code Extension Marketplace

1. Visit the [VS Code Extension Marketplace](https://marketplace.visualstudio.com/VSCode) and search for Ultralytics-snippets or go straight to the [extension page on the VS Code marketplace](https://marketplace.visualstudio.com/items?itemName=Ultralytics.ultralytics-snippets).

2. Click the <kbd>Install</kbd> button and allow your browser to launch a VS Code session.

3. Follow any prompts to install the extension.

<p align="center">
  <br>
    <img src="https://github.com/ultralytics/docs/releases/download/0/vscode-marketplace-extension-install.avif" alt="VS Code marketplace extension install">
  <br>
  Visual Studio Code Extension Marketplace page for <a href="https://marketplace.visualstudio.com/items?itemName=Ultralytics.ultralytics-snippets">Ultralytics-Snippets</a>
</p>

## Using the Ultralytics-Snippets Extension

- 🧠 **Intelligent Code Completion:** Write code faster and more accurately with advanced code completion suggestions tailored to the Ultralytics API.

- ⌛ **Increased Development Speed:** Save time by eliminating repetitive coding tasks and leveraging pre-built code block snippets.

- 🔬 **Improved Code Quality:** Write cleaner, more consistent, and error-free code with intelligent code completion.

- 💎 **Streamlined Workflow:** Stay focused on the core logic of your project by automating common tasks.

### Overview

The extension will only operate when the [Language Mode](https://code.visualstudio.com/docs/getstarted/tips-and-tricks#_change-language-mode) is configured for Python 🐍. This is to avoid snippets from being inserted when working on any other file type. All snippets have prefix starting with `ultra`, and simply typing `ultra` in your editor after installing the extension, will display a list of possible snippets to use. You can also open the VS Code [Command Palette](https://code.visualstudio.com/docs/getstarted/userinterface#_command-palette) using <kbd>Ctrl</kbd>+<kbd>Shift ⇑</kbd>+<kbd>p</kbd> and running the command `Snippets: Insert Snippet`.

### Code Snippet Fields

Many snippets have "fields" with default placeholder values or names. For instance, output from the [predict](../modes/predict.md) method could be saved to a Python variable named `r`, `results`, `detections`, `preds` or whatever else a developer chooses, which is why snippets include "fields". Using <kbd>Tab ⇥</kbd> on your keyboard after a snippet is inserted, your cursor will move between fields quickly. Once a field is selected, typing a new variable name will change that instance, but also every other instance in the snippet code for that variable!

<p align="center">
  <br>
    <img src="https://github.com/ultralytics/docs/releases/download/0/multi-update-field-and-options.avif" alt="Multi-update field and options">
  <br>
  After inserting snippet, renaming <code>model</code> as <code>world_model</code> updates all instances. Pressing <kbd>Tab ⇥</kbd> moves to the next field, which opens a dropdown menu and allows for selection of a model scale, and moving to the next field provides another dropdown to choose either <code>world</code> or <code>worldv2</code> model variant.
</p>

### Code Snippet Completions

!!! tip "Even _Shorter_ Shortcuts"

    It's **not** required to type the full prefix of the snippet, or even to start typing from the start of the snippet. See example in the image below.

The snippets are named in the most descriptive way possible, but this means there could be a lot to type and that would be counterproductive if the aim is to move _faster_. Luckily VS Code lets users type `ultra.example-yolo-predict`, `example-yolo-predict`, `yolo-predict`, or even `ex-yolo-p` and still reach the intended snippet option! If the intended snippet was _actually_ `ultra.example-yolo-predict-kwords`, then just using your keyboard arrows <kbd>↑</kbd> or <kbd>↓</kbd> to highlight the desired snippet and pressing <kbd>Enter ↵</kbd> or <kbd>Tab ⇥</kbd> will insert the correct block of code.

<p align="center">
  <br>
    <img src="https://github.com/ultralytics/docs/releases/download/0/incomplete-snippet-example.avif" alt="Incomplete Snippet Example">
  <br>
  Typing <code>ex-yolo-p</code> will <em>still</em> arrive at the correct snippet.
</p>

### Snippet Categories

These are the current snippet categories available to the Ultralytics-snippets extension. More will be added in the future, so make sure to check for updates and to enable auto-updates for the extension. You can also [request additional snippets](#how-do-i-request-a-new-snippet) to be added if you feel there's any missing.

| Category  | Starting Prefix  | Description                                                                                                                                                                                                           |
| :-------- | :--------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Examples  | `ultra.examples` | Example code to help learn or for getting started with Ultralytics. Examples are copies of or similar to code from documentation pages.                                                                               |
| Kwargs    | `ultra.kwargs`   | Speed up development by adding snippets for [train](../modes/train.md), [track](../modes/track.md), [predict](../modes/predict.md), and [val](../modes/val.md) methods with all keyword arguments and default values. |
| Imports   | `ultra.imports`  | Snippets to quickly import common Ultralytics objects.                                                                                                                                                                |
| Models    | `ultra.yolo`     | Insert code blocks for initializing various [models](../models/index.md) (`yolo`, `sam`, `rtdetr`, etc.), including dropdown configuration options.                                                                   |
| Results   | `ultra.result`   | Code blocks for common operations when [working with inference results](../modes/predict.md#working-with-results).                                                                                                    |
| Utilities | `ultra.util`     | Provides quick access to common utilities that are built into the Ultralytics package, learn more about these on the [Simple Utilities page](../usage/simple-utilities.md).                                           |

### Learning with Examples

The `ultra.examples` snippets are very useful for anyone looking to learn how to get started with the basics of working with Ultralytics YOLO. Example snippets are intended to run once inserted (some have dropdown options as well). An example of this is shown at the animation at the [top](#ultralytics-vs-code-extension) of this page, where after the snippet is inserted, all code is selected and run interactively using <kbd>Shift ⇑</kbd>+<kbd>Enter ↵</kbd>.

!!! example

    Just like the animation shows at the [top](#ultralytics-vs-code-extension) of this page, you can use the snippet `ultra.example-yolo-predict` to insert the following code example. Once inserted, the only configurable option is for the model scale which can be any one of: `n`, `s`, `m`, `l`, or `x`.

    ```python
    from ultralytics import ASSETS, YOLO

    model = YOLO("yolo11n.pt", task="detect")
    results = model(source=ASSETS / "bus.jpg")

    for result in results:
        print(result.boxes.data)
        # result.show()  # uncomment to view each result image
    ```

### Accelerating Development

The aim for snippets other than the `ultra.examples` are for making development easier and quicker when working with Ultralytics. A common code block to be used in many projects, is to iterate the list of `Results` returned from using the model [predict](../modes/predict.md) method. The `ultra.result-loop` snippet can help with this.

!!! example

    Using the `ultra.result-loop` will insert the following default code (including comments).

    ```python
    # reference https://docs.ultralytics.com/modes/predict/#working-with-results

    for result in results:
        result.boxes.data  # torch.Tensor array
    ```

However, since Ultralytics supports numerous [tasks](../tasks/index.md), when [working with inference results](../modes/predict.md#working-with-results) there are other `Results` attributes that you may wish to access, which is where the [snippet fields](#code-snippet-fields) will be powerful.

<p align="center">
  <br>
    <img src="https://github.com/ultralytics/docs/releases/download/0/results-loop-options.avif" alt="Results Loop Options">
  <br>
  Once tabbed to the <code>boxes</code> field, a dropdown menu appears to allow selection of another attribute as required.
</p>

### Keywords Arguments

There are over 💯 keyword arguments for all the various Ultralytics [tasks](../tasks/index.md) and [modes](../modes/index.md)! That's a lot to remember, and it can be easy to forget if the argument is `save_frame` or `save_frames` (it's definitely `save_frames` by the way). This is where the `ultra.kwargs` snippets can help out!

!!! example

    To insert the [predict](../modes/predict.md) method, including all [inference arguments](../modes/predict.md#inference-arguments), use `ultra.kwargs-predict`, which will insert the following code (including comments).

    ```python
    model.predict(
        source=src,  # (str, optional) source directory for images or videos
        imgsz=640,  # (int | list) input images size as int or list[w,h] for predict
        conf=0.25,  # (float) minimum confidence threshold
        iou=0.7,  # (float) intersection over union (IoU) threshold for NMS
        vid_stride=1,  # (int) video frame-rate stride
        stream_buffer=False,  # (bool) buffer incoming frames in a queue (True) or only keep the most recent frame (False)
        visualize=False,  # (bool) visualize model features
        augment=False,  # (bool) apply image augmentation to prediction sources
        agnostic_nms=False,  # (bool) class-agnostic NMS
        classes=None,  # (int | list[int], optional) filter results by class, i.e. classes=0, or classes=[0,2,3]
        retina_masks=False,  # (bool) use high-resolution segmentation masks
        embed=None,  # (list[int], optional) return feature vectors/embeddings from given layers
        show=False,  # (bool) show predicted images and videos if environment allows
        save=True,  # (bool) save prediction results
        save_frames=False,  # (bool) save predicted individual video frames
        save_txt=False,  # (bool) save results as .txt file
        save_conf=False,  # (bool) save results with confidence scores
        save_crop=False,  # (bool) save cropped images with results
        stream=False,  # (bool) for processing long videos or numerous images with reduced memory usage by returning a generator
        verbose=True,  # (bool) enable/disable verbose inference logging in the terminal
    )
    ```

    This snippet has fields for all the keyword arguments, but also for `model` and `src` in case you've used a different variable in your code. On each line containing a keyword argument, a brief description is included for reference.

### All Code Snippets

The best way to find out what snippets are available is to download and install the extension and try it out! If you're curious and want to take a look at the list beforehand, you can visit the [repo](https://github.com/Burhan-Q/ultralytics-snippets) or [extension page on the VS Code marketplace](https://marketplace.visualstudio.com/items?itemName=Ultralytics.ultralytics-snippets) to view the tables for all available snippets.

## Conclusion

The Ultralytics-Snippets extension for VS Code is designed to empower data scientists and machine learning engineers to build [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) applications using Ultralytics YOLO more efficiently. By providing pre-built code snippets and useful examples, we help you focus on what matters most: creating innovative solutions. Please share your feedback by visiting the [extension page on the VS Code marketplace](https://marketplace.visualstudio.com/items?itemName=Ultralytics.ultralytics-snippets) and leaving a review. ⭐

## FAQ

### How do I request a new snippet?

New snippets can be requested using the Issues on the Ultralytics-Snippets [repo](https://github.com/Burhan-Q/ultralytics-snippets).

### How much does the Ultralytics-Extension Cost?

It's 100% free!

### Why don't I see a code snippet preview?

VS Code uses the key combination <kbd>Ctrl</kbd>+<kbd>Space</kbd> to show more/less information in the preview window. If you're not seeing a snippet preview when you type in a code snippet prefix, using this key combination should restore the preview.

### How do I disable the extension recommendation in Ultralytics?

If you use VS Code and have started to see a message prompting you to install the Ultralytics-snippets extension, and don't want to see the message any more, there are two ways to disable this message.

1. Install Ultralytics-snippets and the message will no longer be shown 😆!

2. You can be using `yolo settings vscode_msg False` to disable the message from showing without having to install the extension. You can learn more about the [Ultralytics Settings](../quickstart.md#ultralytics-settings) on the [quickstart](../quickstart.md) page if you're unfamiliar.

### I have an idea for a new Ultralytics code snippet, how can I get one added?

Visit the Ultralytics-snippets [repo](https://github.com/Burhan-Q/ultralytics-snippets) and open an Issue or Pull Request!

### How do I uninstall the Ultralytics-Snippets Extension?

Like any other VS Code extension, you can uninstall it by navigating to the Extensions menu in VS Code. Find the Ultralytics-snippets extension in the menu and click the cog icon (⚙) and then click on "Uninstall" to remove the extension.

<p align="center">
  <br>
    <img src="https://github.com/ultralytics/docs/releases/download/0/vscode-extension-menu.avif" alt="VS Code extension menu">
  <br>
</p>
