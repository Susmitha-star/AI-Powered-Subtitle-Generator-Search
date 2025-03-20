#  🌟 AI-Powered-Subtitle-Generator-Search



## Overview
This project enhances video subtitle search using **NLP & Machine Learning**. It allows users to find the most relevant subtitles for movies and TV shows based on **text queries** or **audio clips**. The system processes **82,498 subtitles** from OpenSubtitles.org and retrieves results efficiently using **semantic search**.

## ✨ Features
- **Keyword & Semantic Search:** Uses **TF-IDF** and **BERT embeddings** for accurate subtitle matching.
- **Audio-Based Query:** Finds relevant subtitles from a **2-minute audio clip**.
- **Fast Retrieval:** Stores subtitle embeddings in **ChromaDB** for optimized searching.
- **Efficient Processing:** Handles large datasets with **document chunking** and **vector-based search**.

## 📚 Dataset
The project works with **82,498 subtitle files** from [OpenSubtitles.org](https://www.opensubtitles.org/en). The data is stored in **eng_subtitles_database.db** with:
- `num`: Unique Subtitle ID (can be used to fetch more details).
- `name`: Subtitle file name.
- `content`: Compressed subtitle text (binary, `latin-1` encoded).

## ⚙️ Installation
### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/video-subtitle-search.git
cd video-subtitle-search
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app.py
```

## 🎯 How It Works
### Part 1: Ingesting Subtitle Data
1. Load the `eng_subtitles_database.db` file.
2. Decode and clean subtitle content.
3. Generate **TF-IDF** (for keyword search) and **BERT embeddings** (for semantic search).
4. Store embeddings in **ChromaDB**.

### Part 2: Retrieving Subtitles
1. Convert **user query** (text/audio) into vector embeddings.
2. Compute **cosine similarity** to find relevant subtitles.
3. Return top-matching subtitle files.

## 🚀 Future Improvements
- Improve accuracy with **advanced transformer models**.
- Optimize **chunking & embedding strategies**.
- Add **real-time subtitle search for live streams**.

## 📄 License
This project is open-source under the [MIT License](LICENSE).

## 📢 Contributing
Pull requests are welcome! Feel free to open issues for improvements.

---

👉 **Star the repo if you find it useful!** ⭐

#AI #MachineLearning #NLP #SearchEngine #Python #ChromaDB #DeepLearning #VideoSearch

