# youtube_liked_playlist_sorter

🎬 YouTube Liked Videos Sorter — Phase 2 (AI-Powered Categorization & Auto Playlists)

This project automatically fetches your Liked YouTube videos, applies machine learning (SentenceTransformer + UMAP + HDBSCAN) to cluster them by topic, and then creates playlists for each cluster using the YouTube Data API.

Originally keyword-based, this phase introduces unsupervised clustering using multilingual embeddings, letting the code infer categories automatically instead of relying on manually defined keywords.

# Features
- Fetches liked videos via the YouTube Data API
- Cleans and pre-processes multilingual text (English + Korean supported)
- Encodes video titles/descriptions with SentenceTransformer (distiluse-base-multilingual-cased-v2)
- Clusters videos via UMAP dimensionality reduction and HDBSCAN
- Automatically labels clusters using top TF-IDF keywords
- Creates YouTube playlists from discovered clusters
- Saves results as CSVs for transparency and debugging

# Workflow Overview
1. Fetch liked videos with fetch_liked_videos.py
2. Auto-cluster and label topics with cluster_auto.py
3. Automatically create playlists on your YouTube account with make_playlists.py

Each script builds upon the previous one — from data collection → machine learning clustering → playlist generation.

# Setup Instructions
1. Prerequisites
- Python 3.10 or higher
- Google Cloud Project with YouTube Data API v3 enabled
- A valid client_secret.json (OAuth Desktop App)

2. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# or .venv\Scripts\activate  # Windows

pip install -r requirements.txt
```
3. Run the scripts
```bash
# Step 1: Fetch liked videos
python app/fetch_liked_videos.py

# Step 2: Cluster liked videos automatically
python app/cluster_auto.py

# Step 3: Create playlists based on clusters
python app/make_playlists.py
```
Outputs
| File                                | Description                           |
| ----------------------------------- | ------------------------------------- |
| `data/liked_videos_raw.csv`         | Raw metadata of liked videos          |
| `data/liked_videos_clusters.csv`    | Clustered + labeled results           |
| `data/liked_videos_categorized.csv` | Simple keyword version (from Phase 1) |

# Authentication
- During your first run, a browser window opens for Google OAuth login.
- The script will generate a token.json — keep this private.
- If you switch YouTube accounts, delete token.json and re-authenticate.

# Tech Stack 
- Python 3.12
- SentenceTransformer for multilingual embeddings
- UMAP for dimensionality reduction
- HDBSCAN for unsupervised clustering
- scikit-learn, NumPy, pandas for data handling
- Google API Client for YouTube playlist automation

# Troubleshooting
| Problem                   | Likely Cause                    | Fix                                                                                                 |
| ------------------------- | ------------------------------- | --------------------------------------------------------------------------------------------------- |
| `insufficientPermissions` | Wrong OAuth scope               | Delete `token.json`, re-run script, and ensure `https://www.googleapis.com/auth/youtube` is enabled |
| `quotaExceeded`           | YouTube API quota limit reached | Wait 24 hours or request more quota in Google Cloud Console                                         |
| Playlist not created      | OAuth still cached              | Delete `token.json`, sign in again                                                                  |
| `FileNotFoundError`       | Missing CSV                     | Run `fetch_liked_videos.py` before clustering                                                       |

# Future Improvements
- Replace HDBSCAN + TF-IDF labeling with a neural topic modeling approach (e.g., BERTopic or LDA + transformers)
- Add a feedback-based re-training system to learn user-preferred clusters over time
- Integrate Gemini or OpenAI APIs for semantic summarization of clusters
- Add a simple dashboard (Streamlit / Flask) to preview clusters before uploading
- Implement automated playlist syncing to update clusters when new likes are added

# Summary
The YouTube Liked Videos Sorter automates the process of organizing your liked videos into thematic playlists, combining data science, AI, and YouTube API automation into a single workflow.

Future versions will make the system self-learning and adaptive, bridging AI categorization with real-time playlist management.
