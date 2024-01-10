---
comments: true
description: Discover the Ultralytics Explorer, a versatile tool and Python API for CV dataset exploration, enabling semantic search, SQL queries, and vector similarity searches.
keywords: Ultralytics Explorer, CV Dataset Tools, Semantic Search, SQL Dataset Queries, Vector Similarity, Python API, GUI Explorer, Dataset Analysis, YOLO Explorer, Data Insights
---

# Ultralytics Explorer

<p>
<img width="1709" alt="Screenshot 2024-01-08 at 7 19 48 PM (1)" src="https://github.com/AyushExel/assets/assets/15766192/e536b0eb-6bce-43fe-b800-3e79510d2e5b">
</p>

<a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/docs/en/datasets/explorer/explorer.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>  
Ultralytics Explorer is a tool for exploring CV datasets using semantic search, SQL queries, vector similarity search and even using natural language. It is also a Python API for accessing the same functionality.

### Installation of optional dependencies

Explorer depends on external libraries for some of its functionality. These are automatically installed on usage. To manually install these dependencies, use the following command:

```bash
pip install ultralytics[explorer]
```

### Explorer API

This is a Python API for Exploring your datasets. It also powers the GUI Explorer. You can use this to create your own exploratory notebooks or scripts to get insights into your datasets.

Learn more about the Explorer API [here](api.md).

## GUI Explorer Usage

The GUI demo runs in your browser allowing you to create embeddings for your dataset and search for similar images, run SQL queries and perform semantic search. It can be run using the following command:

```bash
yolo explorer
```

!!! note "Note"
    Ask AI feature works using OpenAI, so you'll be prompted to set the api key for OpenAI when you first run the GUI.
    You can set it like this - `yolo settings openai_api_key="..."`
