---
comments: true
description: Ultralytics Explorer explores CV datasets with semantic search, SQL queries, vector similarity, and natural language; available in ultralytics 8.3.11 and earlier.
keywords: Ultralytics Explorer, CV datasets, semantic search, SQL queries, vector similarity, dataset visualization, python API, machine learning, computer vision
---

# Ultralytics Explorer

!!! warning "Community Note ⚠️"

    As of **`ultralytics>=8.3.12`**, Ultralytics Explorer has been removed. To use Explorer, install `pip install ultralytics==8.3.11`. Similar (and expanded) dataset exploration features are available in [Ultralytics Platform](https://platform.ultralytics.com).

<p>
    <img width="1709" alt="Ultralytics Explorer dataset visualization GUI" src="https://cdn.ul.run/i/21d5ef59dc96909047855598246a1e42.avif">
</p>

Ultralytics Explorer is a tool for exploring CV datasets using [semantic search](https://www.ultralytics.com/glossary/semantic-search), SQL queries, vector similarity search, and natural language prompts. It also provides a Python API for accessing the same functionality.

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

## Installation of Optional Dependencies

Explorer depends on external libraries for some of its functionality. These are automatically installed when you use Explorer. To manually install these dependencies, use the following command:

```bash
pip install "ultralytics[explorer]==8.3.11"
```

!!! tip

    Explorer works on embedding/semantic search & SQL querying and is powered by the [LanceDB](https://www.lancedb.com/) serverless [vector database](https://www.ultralytics.com/glossary/vector-database). Unlike traditional in-memory DBs, it is persisted on disk without sacrificing performance, so you can scale locally to large datasets like COCO without running out of memory.

## Explorer API

This is a Python API for exploring your datasets. It also powers the GUI Explorer. You can use this to create your own exploratory notebooks or scripts to get insights into your datasets.

Explore the full capabilities and usage examples in the [Explorer API documentation](api.md).

## GUI Explorer Usage

The GUI demo runs in your browser allowing you to create [embeddings](https://www.ultralytics.com/glossary/embeddings) for your dataset and search for similar images, run SQL queries and perform semantic search. It can be run using the following command:

```bash
yolo explorer
```

!!! note

    The Ask AI feature uses OpenAI, so you'll be prompted to set the API key for OpenAI when you first run the GUI.
    You can set it like this - `yolo settings openai_api_key="..."`

<p>
    <img width="1709" alt="Ultralytics Explorer OpenAI Integration" src="https://cdn.ul.run/i/d5d6b20ee49dca3e705ef6a95dc71807.avif">
</p>

## FAQ

### What is Ultralytics Explorer and how can it help with CV datasets?

Ultralytics Explorer is a tool for exploring [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) (CV) datasets through semantic search, SQL queries, vector similarity search, and natural language. It provides both a GUI and a Python API. Because it stores embeddings on disk with [LanceDB](https://www.lancedb.com/), it scales to large datasets without excessive memory use.

Learn more about the [Explorer API](api.md).

### How do I install the dependencies for Ultralytics Explorer?

To manually install the optional dependencies needed for Ultralytics Explorer, you can use the following `pip` command:

```bash
pip install "ultralytics[explorer]==8.3.11"
```

These dependencies are required for semantic search and SQL querying. They include LanceDB, which keeps database operations efficient and scalable, even for large datasets like [COCO](../detect/coco.md).

### How can I use the GUI version of Ultralytics Explorer?

Using the GUI version of Ultralytics Explorer is straightforward. After installing the necessary dependencies, you can launch the GUI with the following command:

```bash
yolo explorer
```

The GUI provides a user-friendly interface for creating dataset embeddings, searching for similar images, running SQL queries, and conducting semantic searches. The Ask AI feature, powered by OpenAI, also lets you query datasets using natural language.

For storage and scalability information, check out our [installation instructions](#installation-of-optional-dependencies).

### What is the Ask AI feature in Ultralytics Explorer?

The Ask AI feature in Ultralytics Explorer allows users to interact with their datasets using natural language queries. Powered by [OpenAI](https://openai.com/), this feature lets you ask questions about your dataset without writing SQL queries or similar commands. To use this feature, you'll need to set your OpenAI API key the first time you run the GUI:

```bash
yolo settings openai_api_key="YOUR_API_KEY"
```

For more on this feature and how to integrate it, see our [GUI Explorer Usage](#gui-explorer-usage) section.

### Can I run Ultralytics Explorer in Google Colab?

Yes. Install the pinned release with `pip install ultralytics==8.3.11` in a Colab notebook and follow the [VOC Exploration Example](explorer.md) to build embeddings and run searches on Google's cloud resources. Learn more in our [Google Colab Guide](../../integrations/google-colab.md).
