---
comments: true
description: Unlock advanced data exploration with Ultralytics Explorer GUI. Utilize semantic search, run SQL queries, and ask AI for natural language data insights.
keywords: Ultralytics Explorer GUI, semantic search, vector similarity, SQL queries, AI, natural language search, data exploration, machine learning, OpenAI, LLMs
---

# Explorer GUI

!!! warning "Community Note ⚠️"

    As of **`ultralytics>=8.3.10`**, Ultralytics Explorer support is deprecated. Similar (and expanded) dataset exploration features are available in [Ultralytics Platform](https://platform.ultralytics.com/).

Explorer GUI is built on the [Ultralytics Explorer API](api.md). It allows you to run semantic/vector similarity search, SQL queries, and natural language queries using the Ask AI feature powered by LLMs.

<p>
    <img width="1709" alt="Ultralytics Explorer GUI main dashboard interface" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/explorer-dashboard-screenshot-1.avif">
</p>

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/3VryynorQeo?start=306"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Ultralytics Explorer Dashboard Overview
</p>

### Installation

```bash
pip install ultralytics[explorer]
```

!!! note

    The Ask AI feature uses OpenAI, so you will be prompted to set the OpenAI API key when you first run the GUI.
    Set it with `yolo settings openai_api_key="..."`.

## Vector Semantic Similarity Search

[Semantic search](https://www.ultralytics.com/glossary/semantic-search) is a technique for finding similar images to a given image. It is based on the idea that similar images will have similar [embeddings](https://www.ultralytics.com/glossary/embeddings). In the UI, you can select one or more images and search for the images similar to them. This can be useful when you want to find images similar to a given image or a set of images that don't perform as expected.

For example, in this VOC Exploration dashboard, the user selects a few airplane images:

<p>
<img width="1710" alt="Explorer selecting airplane images for similarity search" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/explorer-dashboard-screenshot-2.avif">
</p>

After running the similarity search, you should see similar results:

<p>
<img width="1710" alt="Ultralytics Explorer semantic similarity search" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/explorer-dashboard-screenshot-3.avif">
</p>

## Ask AI

This feature lets you filter your dataset using natural language, without writing SQL. The AI-powered query generator converts your prompt into a query and returns matching results. For example, you can ask: "show me 100 images with exactly one person and 2 dogs. There can be other objects too" and it will generate the query and show you those results. Here is an example output when asked: "Show 10 images with exactly 5 persons":

<p>
<img width="1709" alt="Explorer Ask AI results for images with 5 persons" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/explorer-dashboard-screenshot-4.avif">
</p>

Note: This feature uses [Large Language Models](https://www.ultralytics.com/glossary/large-language-model-llm), so results are probabilistic and may be inaccurate.

## Run SQL queries on your CV datasets

You can run SQL queries on your dataset to filter it. It also works if you only provide the WHERE clause. For example, the following WHERE clause returns images that contain at least one person and one dog:

```sql
WHERE labels LIKE '%person%' AND labels LIKE '%dog%'
```

<p>
<img width="1707" alt="Explorer SQL query filtering images with person and dog" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/explorer-dashboard-screenshot-5.avif">
</p>

This demo was built using the Explorer API, which you can use to create your own exploratory notebooks or scripts for gaining insights into your datasets. To get started, check out the [Explorer API documentation](api.md).

## FAQ

### What is Ultralytics Explorer GUI and how do I install it?

Ultralytics Explorer GUI is a powerful interface that unlocks advanced data exploration capabilities using the [Ultralytics Explorer API](api.md). It allows you to run semantic/vector similarity search, SQL queries, and natural language queries using the Ask AI feature powered by [Large Language Models](https://www.ultralytics.com/glossary/large-language-model-llm) (LLMs).

To install the Explorer GUI, you can use pip:

```bash
pip install ultralytics[explorer]
```

Note: To use the Ask AI feature, you'll need to set the OpenAI API key: `yolo settings openai_api_key="..."`.

### How does the semantic search feature in Ultralytics Explorer GUI work?

The semantic search feature in Ultralytics Explorer GUI allows you to find images similar to a given image based on their embeddings. This technique is useful for identifying and exploring images that share visual similarities. To use this feature, select one or more images in the UI and execute a search for similar images. The result will display images that closely resemble the selected ones, facilitating efficient dataset exploration and [anomaly detection](https://www.ultralytics.com/glossary/anomaly-detection).

Learn more about semantic search and other features by visiting the [Feature Overview](#vector-semantic-similarity-search) section.

### Can I use natural language to filter datasets in Ultralytics Explorer GUI?

Yes, with the Ask AI feature powered by large language models (LLMs), you can filter your datasets using natural language queries. You don't need to be proficient in SQL. For instance, you can ask "Show me 100 images with exactly one person and 2 dogs. There can be other objects too," and the AI will generate the appropriate query under the hood to deliver the desired results.

### How do I run SQL queries on datasets using Ultralytics Explorer GUI?

Ultralytics Explorer GUI allows you to run SQL queries directly on your dataset to filter and manage data efficiently. To run a query, navigate to the SQL query section in the GUI and write your query. For example, to show images with at least one person and one dog, you could use:

```sql
WHERE labels LIKE '%person%' AND labels LIKE '%dog%'
```

You can also provide only the WHERE clause, making the querying process more flexible.

For more details, refer to the [SQL Queries Section](#run-sql-queries-on-your-cv-datasets).

### What are the benefits of using Ultralytics Explorer GUI for data exploration?

Ultralytics Explorer GUI enhances data exploration with features like semantic search, SQL querying, and natural language interactions through the Ask AI feature. These capabilities allow users to:

- Efficiently find visually similar images.
- Filter datasets using complex SQL queries.
- Utilize AI to perform natural language searches, eliminating the need for advanced SQL expertise.

These features make it a versatile tool for developers, researchers, and data scientists looking to gain deeper insights into their datasets.

Explore more about these features in the [Explorer GUI Documentation](#explorer-gui).
