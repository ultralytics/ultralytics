---
title: YOLO26 Experiment Tracking with DVCLive
comments: true
description: Track Ultralytics YOLO26 experiments with DVCLive. Log metrics, plots, and models automatically, then compare runs with DVC plots and Plotly.
keywords: YOLO26, DVCLive, experiment tracking, machine learning, model training, data visualization, Git integration
---

# Advanced YOLO26 Experiment Tracking with DVCLive

[Experiment tracking](https://www.ultralytics.com/glossary/experiment-tracking) in [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) is critical to model development and evaluation. It involves recording and analyzing various parameters, metrics, and outcomes from numerous training runs. This process is essential for understanding model performance and making data-driven decisions to refine and optimize models.

Integrating DVCLive with [Ultralytics YOLO26](../models/yolo26.md) automatically logs key experiment details, compares results across runs, and visualizes data for in-depth analysis. This guide shows how to set up and use the integration.

## DVCLive

<p align="center">
  <img width="640" src="https://cdn.ul.run/i/51f227a2cf9c3008135aa0c7cfeaec95.avif" alt="DVCLive experiment tracking integration">
</p>

[DVCLive](https://doc.dvc.org/dvclive), developed by DVC, is an open-source tool for experiment tracking in machine learning. It works with Git and DVC to [automate the logging of experiment data](https://www.ultralytics.com/blog/experiment-tracking-with-ultralytics-yolo11-and-dvc) like model parameters and training metrics, and makes it easy to compare and analyze multiple runs with [data visualization](https://www.ultralytics.com/glossary/data-visualization) and analysis tools.

## YOLO26 Training with DVCLive

YOLO26 training sessions can be monitored with DVCLive. DVC also provides tools for visualizing these experiments, including a report that compares metric plots across all tracked experiments.

## Installation

To install the required packages, run:

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install the required packages for YOLO26 and DVCLive
        pip install ultralytics dvclive
        ```

For detailed instructions and best practices related to the installation process, be sure to check our [YOLO26 Installation guide](../quickstart.md). While installing the required packages for YOLO26, if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

## Configuring DVCLive

Once you have installed the necessary packages, set up your project environment so DVCLive can record experiments alongside your code.

Begin by initializing a Git repository, as Git plays a crucial role in version control for both your code and DVCLive configurations.

!!! tip "Initial Environment Setup"

    === "CLI"

        ```bash
        # Initialize a Git repository
        git init -q

        # Configure Git with your details
        git config --local user.email "your-email"
        git config --local user.name "Your Name"

        # Initialize DVC in your project
        dvc init -q

        # Commit the DVC setup to your Git repository
        git commit -m "DVC init"
        ```

In these commands, ensure you replace `your-email` with the email address associated with your Git account, and "Your Name" with your Git account username.

## Usage

### Training YOLO26 Models with DVCLive

Start by running your YOLO26 training sessions. You can use different model configurations and training parameters to suit your project needs. For instance:

```bash
# Example training commands for YOLO26 with varying configurations
yolo train model=yolo26n.pt data=coco8.yaml epochs=5 imgsz=512
yolo train model=yolo26n.pt data=coco8.yaml epochs=5 imgsz=640
```

Adjust the model, data, [epochs](https://www.ultralytics.com/glossary/epoch), and imgsz parameters according to your specific requirements. For a detailed understanding of the model training process and best practices, refer to our [YOLO26 Model Training guide](../modes/train.md).

### Monitoring Experiments with DVCLive

DVCLive enhances the training process by enabling the tracking and visualization of key metrics. Once DVCLive is installed, its callback is enabled by default. It logs the training arguments, per-epoch losses, validation metrics and learning rates, the training and validation plots, a confusion matrix, and `best.pt` as a model artifact. Run `yolo settings dvc=False` to disable the integration. For a comprehensive understanding of the specific performance metrics used during training, be sure to explore [our detailed guide on performance metrics](../guides/yolo-performance-metrics.md).

### Analyzing Results

After your YOLO26 training sessions are complete, you can use DVCLive's visualization tools to analyze the results. Because the integration logs all training metrics, you can evaluate your model's performance across runs.

To start the analysis, you can extract the experiment data using DVC's API and process it with Pandas for easier handling and visualization:

```python
import dvc.api
import pandas as pd

# Define the columns of interest
columns = ["Experiment", "epochs", "imgsz", "model", "metrics.mAP50-95(B)"]

# Retrieve experiment data
df = pd.DataFrame(dvc.api.exp_show(), columns=columns)

# Clean the data
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Display the DataFrame
print(df)
```

The output of the code snippet above provides a clear tabular view of the different experiments conducted with YOLO26 models. Each row represents a different training run, detailing the experiment's name, the number of epochs, image size (imgsz), the specific model used, and the [mAP50-95(B)](https://www.ultralytics.com/glossary/mean-average-precision-map) metric. This metric is crucial for evaluating the model's [accuracy](https://www.ultralytics.com/glossary/accuracy), with higher values indicating better performance.

#### Visualizing Results with Plotly

For a more interactive and visual analysis of your experiment results, you can use Plotly's parallel coordinates plot. This type of plot is particularly useful for understanding the relationships and trade-offs between different parameters and metrics.

```python
from plotly.express import parallel_coordinates

# Create a parallel coordinates plot
fig = parallel_coordinates(df, columns, color="metrics.mAP50-95(B)")

# Display the plot
fig.show()
```

The output of the code snippet above generates a plot that will visually represent the relationships between epochs, image size, model type, and their corresponding mAP50-95(B) scores, enabling you to spot trends and patterns in your experiment data.

#### Generating Comparative Visualizations with DVC

DVC provides a useful command to generate comparative plots for your experiments. This can be especially helpful to compare the performance of different models over various training runs.

```bash
# Generate DVC comparative plots
dvc plots diff $(dvc exp list --names-only)
```

After executing this command, DVC generates plots comparing the metrics across different experiments, which are saved as HTML files. Below is an example image illustrating typical plots generated by this process. The image showcases various graphs, including those representing mAP, [recall](https://www.ultralytics.com/glossary/recall), [precision](https://www.ultralytics.com/glossary/precision), loss values, and more, providing a visual overview of key performance metrics:

<p align="center">
  <img width="640" src="https://cdn.ul.run/i/06bdca7444b703423cc9511506ea38b2.avif" alt="DVCLive training metrics comparison plots">
</p>

### Displaying DVC Plots

If you are using a Jupyter Notebook and you want to display the generated DVC plots, you can use the IPython display functionality.

```python
from IPython.display import HTML

# Display the DVC plots as HTML
HTML(filename="./dvc_plots/index.html")
```

This code will render the HTML file containing the DVC plots directly in your Jupyter Notebook, providing an easy and convenient way to analyze the visualized experiment data.

### Making Data-Driven Decisions

Use the insights gained from these visualizations to make informed decisions about model optimizations, [hyperparameter tuning](https://www.ultralytics.com/glossary/hyperparameter-tuning), and other modifications to enhance your model's performance.

### Iterating on Experiments

Based on your analysis, iterate on your experiments. Adjust model configurations, training parameters, or even the data inputs, and repeat the training and analysis process. This iterative approach is key to refining your model for the best possible performance.

## Summary

This guide has led you through integrating DVCLive with Ultralytics YOLO26, from setup to experiment monitoring, visualization, and analysis.

For further details on usage, visit [DVCLive's official documentation](https://doc.dvc.org/dvclive/ml-frameworks/yolo).

## FAQ

### How do I integrate DVCLive with Ultralytics YOLO26 for experiment tracking?

Integrating DVCLive with Ultralytics YOLO26 is straightforward. Start by installing the necessary packages:

!!! example "Installation"

    === "CLI"

        ```bash
        pip install ultralytics dvclive
        ```

Next, initialize a Git repository and configure DVCLive in your project:

!!! example "Initial Environment Setup"

    === "CLI"

        ```bash
        git init -q
        git config --local user.email "your-email"
        git config --local user.name "Your Name"
        dvc init -q
        git commit -m "DVC init"
        ```

Follow our [YOLO26 Installation guide](../quickstart.md) for detailed setup instructions.

### Why should I use DVCLive for tracking YOLO26 experiments?

Using DVCLive with YOLO26 provides several advantages, such as:

- **Automated Logging**: DVCLive automatically records key experiment details like model parameters and metrics.
- **Easy Comparison**: Facilitates comparison of results across different runs.
- **Visualization Tools**: Uses DVCLive's data visualization capabilities for in-depth analysis.

For further details, refer to our guide on [YOLO26 Model Training](../modes/train.md) and [YOLO Performance Metrics](../guides/yolo-performance-metrics.md) to maximize your experiment tracking efficiency.

### How can DVCLive improve my results analysis for YOLO26 training sessions?

After completing your YOLO26 training sessions, DVCLive helps in visualizing and analyzing the results effectively. Example code for loading and displaying experiment data:

```python
import dvc.api
import pandas as pd

# Define columns of interest
columns = ["Experiment", "epochs", "imgsz", "model", "metrics.mAP50-95(B)"]

# Retrieve experiment data
df = pd.DataFrame(dvc.api.exp_show(), columns=columns)

# Clean data
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Display DataFrame
print(df)
```

To visualize results interactively, use Plotly's parallel coordinates plot:

```python
from plotly.express import parallel_coordinates

fig = parallel_coordinates(df, columns, color="metrics.mAP50-95(B)")
fig.show()
```

Refer to our guide on [YOLO26 Training with DVCLive](#yolo26-training-with-dvclive) for more examples and best practices.

### What are the steps to configure my environment for DVCLive and YOLO26 integration?

To configure your environment for a smooth integration of DVCLive and YOLO26, follow these steps:

1. **Install Required Packages**: Use `pip install ultralytics dvclive`.
2. **Initialize Git Repository**: Run `git init -q`.
3. **Initialize DVC**: Execute `dvc init -q`.
4. **Commit to Git**: Use `git commit -m "DVC init"`.

These steps ensure proper version control and setup for experiment tracking.

### How do I visualize YOLO26 experiment results using DVCLive?

DVCLive offers tools to visualize the results of YOLO26 experiments. Here's how you can generate comparative plots:

!!! example "Generate Comparative Plots"

    === "CLI"

        ```bash
        dvc plots diff $(dvc exp list --names-only)
        ```

To display these plots in a Jupyter Notebook, use:

```python
from IPython.display import HTML

# Display plots as HTML
HTML(filename="./dvc_plots/index.html")
```

These visualizations help identify trends and optimize model performance.
