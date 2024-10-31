---
comments: true
description: Unlock advanced data exploration with Ultralytics Explorer GUI. Utilize semantic search, run SQL queries, and ask AI for natural language data insights.
keywords: Ultralytics Explorer GUI, semantic search, vector similarity, SQL queries, AI, natural language search, data exploration, machine learning, OpenAI, LLMs
---

# Explorer GUI

!!! warning "Community Note ⚠️"

    As of **`ultralytics>=8.3.10`**, Ultralytics explorer support has been deprecated. But don't worry! You can now access similar and even enhanced functionality through [Ultralytics HUB](https://hub.ultralytics.com/), our intuitive no-code platform designed to streamline your workflow. With Ultralytics HUB, you can continue exploring, visualizing, and managing your data effortlessly, all without writing a single line of code. Make sure to check it out and take advantage of its powerful features!🚀

Explorer GUI is like a playground build using [Ultralytics Explorer API](api.md). It allows you to run semantic/vector similarity search, SQL queries and even search using natural language using our ask AI feature powered by LLMs.

<p>
    <img width="1709" alt="Explorer Dashboard Screenshot 1" src="https://github.com/ultralytics/docs/releases/download/0/explorer-dashboard-screenshot-1.avif">
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

    Ask AI feature works using OpenAI, so you'll be prompted to set the api key for OpenAI when you first run the GUI.
    You can set it like this - `yolo settings openai_api_key="..."`

## Vector Semantic Similarity Search

Semantic search is a technique for finding similar images to a given image. It is based on the idea that similar images will have similar [embeddings](https://www.ultralytics.com/glossary/embeddings). In the UI, you can select one of more images and search for the images similar to them. This can be useful when you want to find images similar to a given image or a set of images that don't perform as expected.

For example:
In this VOC Exploration dashboard, user selects a couple airplane images like this:

<p>
<img width="1710" alt="Explorer Dashboard Screenshot 2" src="https://github.com/ultralytics/docs/releases/download/0/explorer-dashboard-screenshot-2.avif">
</p>

On performing similarity search, you should see a similar result:

<p>
<img width="1710" alt="Explorer Dashboard Screenshot 3" src="https://github.com/ultralytics/docs/releases/download/0/explorer-dashboard-screenshot-3.avif">
</p>

## Ask AI

This allows you to write how you want to filter your dataset using natural language. You don't have to be proficient in writing SQL queries. Our AI powered query generator will automatically do that under the hood. For example - you can say - "show me 100 images with exactly one person and 2 dogs. There can be other objects too" and it'll internally generate the query and show you those results. Here's an example output when asked to "Show 10 images with exactly 5 persons" and you'll see a result like this:

<p>
<img width="1709" alt="Explorer Dashboard Screenshot 4" src="https://github.com/ultralytics/docs/releases/download/0/explorer-dashboard-screenshot-4.avif">
</p>

Note: This works using LLMs under the hood so the results are probabilistic and might get things wrong sometimes

## Run SQL queries on your CV datasets

You can run SQL queries on your dataset to filter it. It also works if you only provide the WHERE clause. Example SQL query would show only the images that have at least one 1 person and 1 dog in them:

```sql
WHERE labels LIKE '%person%' AND labels LIKE '%dog%'
```

<p>
<img width="1707" alt="Explorer Dashboard Screenshot 5" src="https://github.com/ultralytics/docs/releases/download/0/explorer-dashboard-screenshot-5.avif">
</p>

This is a Demo build using the Explorer API. You can use the API to build your own exploratory notebooks or scripts to get insights into your datasets. Learn more about the Explorer API [here](api.md).

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

See an example of a natural language query [here](#ask-ai).

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
