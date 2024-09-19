---
comments: true
description: Discover Ultralytics Explorer for semantic search, SQL queries, vector similarity, and natural language dataset exploration. Enhance your CV datasets effortlessly.
keywords: Ultralytics Explorer, CV datasets, semantic search, SQL queries, vector similarity, dataset visualization, python API, machine learning, computer vision
---

# Ultralytics Explorer

<p>
    <img width="1709" alt="Ultralytics Explorer Screenshot 1" src="https://github.com/ultralytics/docs/releases/download/0/explorer-dashboard-screenshot-1.avif">
</p>

<a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/docs/en/datasets/explorer/explorer.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
Ultralytics Explorer is a tool for exploring CV datasets using semantic search, SQL queries, vector similarity search and even using natural language. It is also a Python API for accessing the same functionality.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/3VryynorQeo"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Ultralytics Explorer API | Semantic Search, SQL Queries & Ask AI Features
</p>

### Installation of optional dependencies

Explorer depends on external libraries for some of its functionality. These are automatically installed on usage. To manually install these dependencies, use the following command:

```bash
pip install ultralytics[explorer]
```

!!! tip

    Explorer works on embedding/semantic search & SQL querying and is powered by [LanceDB](https://lancedb.com/) serverless vector database. Unlike traditional in-memory DBs, it is persisted on disk without sacrificing performance, so you can scale locally to large datasets like COCO without running out of memory.

### Explorer API

This is a Python API for Exploring your datasets. It also powers the GUI Explorer. You can use this to create your own exploratory notebooks or scripts to get insights into your datasets.

Learn more about the Explorer API [here](api.md).

## GUI Explorer Usage

The GUI demo runs in your browser allowing you to create embeddings for your dataset and search for similar images, run SQL queries and perform semantic search. It can be run using the following command:

```bash
yolo explorer
```

!!! note

    Ask AI feature works using OpenAI, so you'll be prompted to set the api key for OpenAI when you first run the GUI.
    You can set it like this - `yolo settings openai_api_key="..."`

<p>
    <img width="1709" alt="Ultralytics Explorer OpenAI Integration" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-explorer-openai-integration.avif">
</p>

## FAQ

### What is Ultralytics Explorer and how can it help with CV datasets?

Ultralytics Explorer is a powerful tool designed for exploring computer vision (CV) datasets through semantic search, SQL queries, vector similarity search, and even natural language. This versatile tool provides both a GUI and a Python API, allowing users to seamlessly interact with their datasets. By leveraging technologies like LanceDB, Ultralytics Explorer ensures efficient, scalable access to large datasets without excessive memory usage. Whether you're performing detailed dataset analysis or exploring data patterns, Ultralytics Explorer streamlines the entire process.

Learn more about the [Explorer API](api.md).

### How do I install the dependencies for Ultralytics Explorer?

To manually install the optional dependencies needed for Ultralytics Explorer, you can use the following `pip` command:

```bash
pip install ultralytics[explorer]
```

These dependencies are essential for the full functionality of semantic search and SQL querying. By including libraries powered by [LanceDB](https://lancedb.com/), the installation ensures that the database operations remain efficient and scalable, even for large datasets like COCO.

### How can I use the GUI version of Ultralytics Explorer?

Using the GUI version of Ultralytics Explorer is straightforward. After installing the necessary dependencies, you can launch the GUI with the following command:

```bash
yolo explorer
```

The GUI provides a user-friendly interface for creating dataset embeddings, searching for similar images, running SQL queries, and conducting semantic searches. Additionally, the integration with OpenAI's Ask AI feature allows you to query datasets using natural language, enhancing the flexibility and ease of use.

For storage and scalability information, check out our [installation instructions](#installation-of-optional-dependencies).

### What is the Ask AI feature in Ultralytics Explorer?

The Ask AI feature in Ultralytics Explorer allows users to interact with their datasets using natural language queries. Powered by OpenAI, this feature enables you to ask complex questions and receive insightful answers without needing to write SQL queries or similar commands. To use this feature, you'll need to set your OpenAI API key the first time you run the GUI:

```bash
yolo settings openai_api_key="YOUR_API_KEY"
```

For more on this feature and how to integrate it, see our [GUI Explorer Usage](#gui-explorer-usage) section.

### Can I run Ultralytics Explorer in Google Colab?

Yes, Ultralytics Explorer can be run in Google Colab, providing a convenient and powerful environment for dataset exploration. You can start by opening the provided Colab notebook, which is pre-configured with all the necessary settings:

<a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/docs/en/datasets/explorer/explorer.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

This setup allows you to explore your datasets fully, taking advantage of Google's cloud resources. Learn more in our [Google Colab Guide](../../integrations/google-colab.md).
