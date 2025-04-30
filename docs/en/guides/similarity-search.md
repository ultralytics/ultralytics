# Semantic Search 🧠✨

Semantic Search is a powerful solution that enables AI-based image retrieval based on natural language queries. Unlike traditional keyword or tag-based searches, this solution understands the visual and contextual meaning of a phrase—returning images that best match the query.

## 🔍 What It Does

**Semantic Search** uses the power of OpenCLIP and FAISS to search through a dataset of images based on a textual prompt. Instead of relying on filenames or metadata, it uses deep visual understanding to find the most semantically similar images.

### Example use cases:
- "Find all images of people riding bicycles"
- "Retrieve images where two people are talking"
- "Search for animals in nature"

## 🛠️ How It Works

The solution has two major components:

### 1. **OpenCLIP Embeddings**
- Uses OpenCLIP's `ViT-B-32-quickgelu` model to extract image and text embeddings.
- Text inputs are tokenized and converted into vector representations.
- Images are preprocessed and embedded using the CLIP model to capture visual semantics.

### 2. **FAISS Indexing**
- FAISS (Facebook AI Similarity Search) is used to build a fast, L2-normalized inner product index of image embeddings.
- This allows fast, scalable similarity search across thousands of images.
- The top-k results are returned based on semantic closeness to the input prompt.

## 🖼️ Live UI Demo

The solution includes a simple yet elegant web interface built using Flask and HTML:

- Input: A natural language description of the scene you're looking for.
- Output: A responsive grid of top-matching images.

Try it with prompts like:
- `"man on a skateboard"`
- `"city skyline at night"`
- `"dog jumping over a fence"`

![UI](https://raw.githubusercontent.com/ultralytics/assets/main/logo/favicon.png)

## 🧪 Setup & Run

Install the necessary dependencies (automatically checked in the script):

```bash
pip install faiss-cpu open-clip-torch flask
```

Then run the app:

```bash
python similarity_search.py
```

This will:
- Load or build the FAISS index from images in the `images/` directory.
- Launch a local Flask server at `http://127.0.0.1:5000`.

## 📦 Solution Overview

| Component | Purpose |
|----------|---------|
| `VisualAISearch` | Main logic for embedding, indexing, and search |
| `SearchApp` | Flask web server that renders search results |
| `index.html` | Frontend template for interactive UI |
| `images/` | Folder with images to be indexed |
| `faiss.index` | Saved FAISS vector index |
| `paths.npy` | Stores image paths corresponding to index entries |

## 🧠 Technical Highlights

- **FAISS Inner Product Index** with L2 normalization for cosine similarity.
- Automatically downloads sample images if none found.
- Skips invalid images gracefully with helpful logging.
- Prioritizes user experience with animated frontend and styled UI elements.

## 📈 Output Format

The search returns a ranked list of top-k image filenames and their similarity scores, which can be logged or displayed visually.

```python
[
  ("person_walking_dog.jpg", 0.78),
  ("city_walker.png", 0.74),
  ...
]
```

---

Let me know if you'd like this exported as Markdown or added to an MkDocs structure!