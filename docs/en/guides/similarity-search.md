---
title: Semantic Image Search with CLIP and FAISS
comments: true
description: Build a semantic image search engine with OpenAI CLIP, Meta FAISS, and Flask. Embed images, run natural-language queries, and serve results from a web app with the Ultralytics Python package.
keywords: CLIP, FAISS, Flask, semantic search, semantic image search, image retrieval, natural language image search, zero-shot search, vector search, embeddings, OpenAI, Meta, Ultralytics, VisualAISearch, computer vision, web app
---

# How to Build Semantic Image Search with OpenAI CLIP and Meta FAISS

This guide walks you through building a **semantic image search** engine using [OpenAI CLIP](https://openai.com/index/clip/), [Meta FAISS](https://github.com/facebookresearch/faiss), and [Flask](https://flask.palletsprojects.com/en/stable/). By combining CLIP's visual-language [embeddings](https://developers.openai.com/api/docs/guides/embeddings) with FAISS's efficient nearest-neighbor search, you can build a web interface that retrieves relevant images from natural language queries, no labels or categories required.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/zplKRlX3sLg"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How Similarity Search Works | Visual Search Using OpenAI CLIP, META FAISS and Ultralytics Package 🎉
</p>

![Flask webpage with semantic search results overview](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/flask-ui.avif)

The Ultralytics Python package wraps this entire pipeline behind two classes, so you can launch a working search app or run queries programmatically in a few lines. This guide covers [why semantic search is useful](#why-use-semantic-image-search), [how it works](#how-semantic-image-search-works), [running the web app](#run-the-semantic-search-web-app), [searching programmatically](#search-images-programmatically), and [configuring parameters](#configure-visualaisearch-parameters).

## Why Use Semantic Image Search?

Building your own semantic image search system with CLIP and FAISS provides several compelling advantages:

- **Zero-shot capabilities:** You don't need to train on your dataset. CLIP's [zero-shot learning](https://www.ultralytics.com/glossary/zero-shot-learning) lets you query any image collection with free-form natural language, saving time and resources.
- **Human-like understanding:** Unlike keyword search, CLIP understands semantic context and retrieves images from abstract, emotional, or relational queries like "a happy child in nature" or "a futuristic city skyline at night."
- **No labels or metadata:** This approach needs only raw images. CLIP generates embeddings without any manual annotation.
- **Flexible and scalable search:** FAISS delivers fast nearest-neighbor search even on large datasets, with real-time response across thousands or millions of embeddings.
- **Cross-domain applications:** Whether you're building a personal photo archive, a creative inspiration tool, a product search engine, or an art recommendation system, the same stack adapts with minimal tweaking.

## How Semantic Image Search Works

The pipeline combines three components, each handling one stage of turning images and text into ranked results:

- **CLIP** uses a vision encoder (e.g., ResNet or ViT) for images and a text encoder (Transformer-based) for language to project both into the same multimodal embedding space. This allows direct comparison between text and images using [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity).
- **FAISS** (Facebook AI Similarity Search) builds an index of the image embeddings and enables fast, scalable retrieval of the closest vectors to a given query.
- **Flask** provides a simple web interface to submit natural language queries and display semantically matched images from the index.

![OpenAI Clip image retrieval workflow](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/clip-image-retrieval.avif)

Because both images and text land in the same vector space, retrieval is zero-shot: you don't need labels or categories, just image data and a good prompt.

![Meta FAISS embedding vectors building workflow](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/faiss-indexing-workflow.avif)

## Run the Semantic Search Web App

The `SearchApp` class launches the full Flask interface. On first run it downloads a sample image set, builds the FAISS index, and serves a page where you can type a query and view ranked results.

??? note "Image Path Warning"

    If you're using your own images, make sure to provide an absolute path to the image directory. Otherwise, the images may not appear on the webpage due to Flask's file serving limitations.

=== "Python"

    ```python
    from ultralytics import solutions

    app = solutions.SearchApp(
        # data = "path/to/img/directory" # Optional, build search engine with your own images
        device="cpu"  # configure the device for processing, e.g., "cpu" or "cuda"
    )

    app.run(debug=False)  # You can also use `debug=True` argument for testing
    ```

## Search Images Programmatically

The `VisualAISearch` class performs all the backend operations without the web layer:

- Loads or builds a FAISS index from local images.
- Extracts image and text [embeddings](https://developers.openai.com/api/docs/guides/embeddings) using CLIP.
- Performs similarity search using cosine similarity.

Call the searcher with a natural language query to get back a list of matching image filenames ranked by similarity:

=== "Python"

    ```python
    from ultralytics import solutions

    searcher = solutions.VisualAISearch(
        # data = "path/to/img/directory" # Optional, build search engine with your own images
        device="cpu"  # configure the device for processing, e.g., "cpu" or "cuda"
    )

    results = searcher("a dog sitting on a bench")

    # Ranked Results:
    #     - 000000546829.jpg | Similarity: 0.3269
    #     - 000000549220.jpg | Similarity: 0.2899
    #     - 000000517069.jpg | Similarity: 0.2761
    #     - 000000029393.jpg | Similarity: 0.2742
    #     - 000000534270.jpg | Similarity: 0.2680
    ```

## Configure VisualAISearch Parameters

The table below outlines the available parameters for `VisualAISearch`:

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["data"]) }}
{% from "macros/track-args.md" import param_table %}
{{ param_table(["device"]) }}

!!! tip "Manage your data in the cloud"

    To search image collections at production scale without managing local files, you can organize and version your images in the [Ultralytics Platform](../platform/data/index.md) before indexing them with CLIP and FAISS.

## Conclusion

With CLIP, FAISS, and the Ultralytics Python package, you can stand up a zero-shot semantic image search engine in just a few lines, either as a Flask web app or as a programmatic search backend. From here, point `data` at your own image directory to index it, then explore other [Ultralytics Solutions](../solutions/index.md) to build on top of your computer vision workflows.

## FAQ

### How does CLIP understand both images and text?

[CLIP](https://github.com/openai/CLIP) (Contrastive Language Image Pretraining) is a model developed by [OpenAI](https://openai.com/) that learns to connect visual and linguistic information. It's trained on a massive dataset of images paired with natural language captions. This training allows it to map both images and text into a shared embedding space, so you can compare them directly using vector similarity.

### Why is CLIP considered so powerful for AI tasks?

What makes CLIP stand out is its ability to generalize. Instead of being trained just for specific labels or tasks, it learns from natural language itself. This allows it to handle flexible queries like "a man riding a jet ski" or "a surreal dreamscape," making it useful for everything from classification to creative semantic search, without retraining.

### What exactly does FAISS do in this project (Semantic Search)?

[FAISS](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/) (Facebook AI Similarity Search) is a toolkit that helps you search through high-dimensional vectors very efficiently. Once CLIP turns your images into embeddings, FAISS makes it fast and easy to find the closest matches to a text query, perfect for real-time image retrieval.

### Why use the [Ultralytics](https://www.ultralytics.com/) [Python package](https://github.com/ultralytics/ultralytics/) if CLIP and FAISS are from OpenAI and Meta?

While CLIP and FAISS are developed by OpenAI and Meta respectively, the [Ultralytics Python package](https://pypi.org/project/ultralytics/) wraps both into a complete semantic image search pipeline behind a few lines of code that just work:

=== "Python"

    ```python
    from ultralytics import solutions

    searcher = solutions.VisualAISearch(
        # data = "path/to/img/directory" # Optional, build search engine with your own images
        device="cpu"  # configure the device for processing, e.g., "cpu" or "cuda"
    )

    results = searcher("a dog sitting on a bench")
    ```

This high-level implementation handles:

- CLIP-based image and text embedding generation.
- FAISS index creation and management.
- Efficient semantic search with cosine similarity.
- Directory-based image loading and [visualization](https://www.ultralytics.com/glossary/data-visualization).

### Can I customize the frontend of this app?

Yes. The current setup uses Flask with a basic HTML frontend, but you can replace it with your own HTML or build a more dynamic UI with React, Vue, or another frontend framework. Flask can serve as the backend API for your custom interface.

### Is it possible to search through videos instead of static images?

Not directly. A simple workaround is to extract individual frames from your videos (e.g., one every second), treat them as standalone images, and feed those into the system. This way, the search engine can semantically index visual moments from your videos.
