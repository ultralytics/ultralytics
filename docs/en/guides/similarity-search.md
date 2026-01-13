---
comments: true
description: Build a semantic image search web app using OpenAI CLIP, Meta FAISS, and Flask. Learn how to embed images and retrieve them using natural language.
keywords: CLIP, FAISS, Flask, semantic search, image retrieval, OpenAI, Ultralytics, tutorial, computer vision, web app
---

# Semantic Image Search with OpenAI CLIP and Meta FAISS

## Introduction

This guide walks you through building a **semantic image search** engine using [OpenAI CLIP](https://openai.com/blog/clip), [Meta FAISS](https://github.com/facebookresearch/faiss), and [Flask](https://flask.palletsprojects.com/en/stable/). By combining CLIP's powerful visual-language embeddings with FAISS's efficient nearest-neighbor search, you can create a fully functional web interface where you can retrieve relevant images using natural language queries.

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

## Semantic Image Search Visual Preview

![Flask webpage with semantic search results overview](https://github.com/ultralytics/docs/releases/download/0/flask-ui.avif)

## How It Works

- **CLIP** uses a vision encoder (e.g., ResNet or ViT) for images and a text encoder (Transformer-based) for language to project both into the same multimodal embedding space. This allows for direct comparison between text and images using [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity).
- **FAISS** (Facebook AI Similarity Search) builds an index of the image embeddings and enables fast, scalable retrieval of the closest vectors to a given query.
- **Flask** provides a simple web interface to submit natural language queries and display semantically matched images from the index.

This architecture supports zero-shot search, meaning you don't need labels or categories, just image data and a good prompt.

!!! example "Semantic Image Search using Ultralytics Python package"

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

## `VisualAISearch` class

This class performs all the backend operations:

- Loads or builds a FAISS index from local images.
- Extracts image and text [embeddings](https://platform.openai.com/docs/guides/embeddings) using CLIP.
- Performs similarity search using cosine similarity.

!!! example "Similar Images Search"

    ??? note "Image Path Warning"

         If you're using your own images, make sure to provide an absolute path to the image directory. Otherwise, the images may not appear on the webpage due to Flask's file serving limitations.

    === "Python"

        ```python
        from ultralytics import solutions

        searcher = solutions.VisualAISearch(
            # data = "path/to/img/directory" # Optional, build search engine with your own images
            device="cuda"  # configure the device for processing, e.g., "cpu" or "cuda"
        )

        results = searcher("a dog sitting on a bench")

        # Ranked Results:
        #     - 000000546829.jpg | Similarity: 0.3269
        #     - 000000549220.jpg | Similarity: 0.2899
        #     - 000000517069.jpg | Similarity: 0.2761
        #     - 000000029393.jpg | Similarity: 0.2742
        #     - 000000534270.jpg | Similarity: 0.2680
        ```

## `VisualAISearch` Parameters

The table below outlines the available parameters for `VisualAISearch`:

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["data"]) }}
{% from "macros/track-args.md" import param_table %}
{{ param_table(["device"]) }}

## Advantages of Semantic Image Search with CLIP and FAISS

Building your own semantic image search system with CLIP and FAISS provides several compelling advantages:

1. **Zero-Shot Capabilities**: You don't need to train the model on your specific dataset. CLIP's zero-shot learning lets you perform search queries on any image dataset using free-form natural language, saving both time and resources.

2. **Human-Like Understanding**: Unlike keyword-based search engines, CLIP understands semantic context. It can retrieve images based on abstract, emotional, or relational queries like "a happy child in nature" or "a futuristic city skyline at night".

    ![OpenAI Clip image retrieval workflow](https://github.com/ultralytics/docs/releases/download/0/clip-image-retrieval.avif)

3. **No Need for Labels or Metadata**: Traditional image search systems require carefully labeled data. This approach only needs raw images. CLIP generates embeddings without needing any manual annotation.

4. **Flexible and Scalable Search**: FAISS enables fast nearest-neighbor search even with large-scale datasets. It's optimized for speed and memory, allowing real-time response even with thousands (or millions) of embeddings.

    ![Meta FAISS embedding vectors building workflow](https://github.com/ultralytics/docs/releases/download/0/faiss-indexing-workflow.avif)

5. **Cross-Domain Applications**: Whether you're building a personal photo archive, a creative inspiration tool, a product search engine, or even an art recommendation system, this stack adapts to diverse domains with minimal tweaking.

## FAQ

### How does CLIP understand both images and text?

[CLIP](https://github.com/openai/CLIP) (Contrastive Language Image Pretraining) is a model developed by [OpenAI](https://openai.com/) that learns to connect visual and linguistic information. It's trained on a massive dataset of images paired with natural language captions. This training allows it to map both images and text into a shared embedding space, so you can compare them directly using vector similarity.

### Why is CLIP considered so powerful for AI tasks?

What makes CLIP stand out is its ability to generalize. Instead of being trained just for specific labels or tasks, it learns from natural language itself. This allows it to handle flexible queries like “a man riding a jet ski” or “a surreal dreamscape,” making it useful for everything from classification to creative semantic search, without retraining.

### What exactly does FAISS do in this project (Semantic Search)?

[FAISS](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/) (Facebook AI Similarity Search) is a toolkit that helps you search through high-dimensional vectors very efficiently. Once CLIP turns your images into embeddings, FAISS makes it fast and easy to find the closest matches to a text query, perfect for real-time image retrieval.

### Why use the [Ultralytics](https://www.ultralytics.com/) [Python package](https://github.com/ultralytics/ultralytics/) if CLIP and FAISS are from OpenAI and Meta?

While CLIP and FAISS are developed by OpenAI and Meta respectively, the [Ultralytics Python package](https://pypi.org/project/ultralytics/) simplifies their integration into a complete semantic image search pipeline in a 2-lines workflow that just works:

!!! example "Similar Images Search"

    === "Python"

        ```python
        from ultralytics import solutions

        searcher = solutions.VisualAISearch(
            # data = "path/to/img/directory" # Optional, build search engine with your own images
            device="cuda"  # configure the device for processing, e.g., "cpu" or "cuda"
        )

        results = searcher("a dog sitting on a bench")

        # Ranked Results:
        #     - 000000546829.jpg | Similarity: 0.3269
        #     - 000000549220.jpg | Similarity: 0.2899
        #     - 000000517069.jpg | Similarity: 0.2761
        #     - 000000029393.jpg | Similarity: 0.2742
        #     - 000000534270.jpg | Similarity: 0.2680
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
