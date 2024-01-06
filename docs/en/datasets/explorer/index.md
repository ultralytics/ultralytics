---
comments: true
description: Discover the Ultralytics Explorer, a versatile tool and Python API for CV dataset exploration, enabling semantic search, SQL queries, and vector similarity searches.
keywords: Ultralytics Explorer, CV Dataset Tools, Semantic Search, SQL Dataset Queries, Vector Similarity, Python API, GUI Explorer, Dataset Analysis, YOLO Explorer, Data Insights
---

# Ultralytics Explorer

Ultralytics Explorer is a tool for exploring CV datasets using semantic search, SQL queries and vector similarity search. It is also a Python API for accessing the same functionality.

### Installation of optional dependencies

Explorer depends on external libraries for some of its functionality. These are automatically installed on usage. To manually install these dependencies, use the following command:

```bash
pip install ultralytics[explorer]
```

## GUI Explorer Usage

The GUI demo runs in your browser allowing you to create embeddings for your dataset and search for similar images, run SQL queries and perform semantic search. It can be run using the following command:

```bash
yolo explorer
```

### Explorer API

This is a Python API for Exploring your datasets. It also powers the GUI Explorer. You can use this to create your own exploratory notebooks or scripts to get insights into your datasets.

Learn more about the Explorer API [here](api.md).
