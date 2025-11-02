# app/cluster_with_sklearn.py
import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ROOT = os.path.dirname(os.path.dirname(__file__))
RAW = os.path.join(ROOT, "data", "liked_videos_raw.csv")
OUT = os.path.join(ROOT, "data", "liked_videos_clusters.csv")

# 1) Load raw data
df = pd.read_csv(RAW)

# Combine title + description (handle NaN)
def safe(x): return "" if pd.isna(x) else str(x)
texts = (df["title"].map(safe) + " " + df["description"].map(safe)).tolist()

# 2) Vectorize text with TF-IDF
#   - strip URLs, emojis, punctuation-ish noise
def preproc(s: str) -> str:
    s = s.lower()
    s = re.sub(r"http\S+", " ", s)           # URLs
    s = re.sub(r"[\W_]+", " ", s)            # non-letters
    return s

vectorizer = TfidfVectorizer(
    preprocessor=preproc,
    max_features=25000,
    min_df=2,
    stop_words="english",     # good default; works ok for mixed langs too
)
X_tfidf = vectorizer.fit_transform(texts)

# 3) Optional: reduce dimensions (faster, nicer clustering)
svd = TruncatedSVD(n_components=min(300, X_tfidf.shape[1]-1))
X = make_pipeline(svd, Normalizer(copy=False)).fit_transform(X_tfidf)

# 4) Pick best k via silhouette
candidates = [5,6,7,8,9,10] if len(df) >= 80 else [3,4,5,6,7]
best_k, best_score, best_labels, best_model = None, -1, None, None
for k in candidates:
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(X)
    # silhouette needs at least 2 clusters and < n_samples
    if len(set(labels)) > 1 and len(set(labels)) < len(df):
        score = silhouette_score(X, labels)
    else:
        score = -1
    print(f"k={k} silhouette={score:.4f}")
    if score > best_score:
        best_k, best_score, best_labels, best_model = k, score, labels, km

print(f"Chosen k = {best_k} (silhouette={best_score:.4f})")

# 5) Name clusters from top terms
#   - Map SVD components back to TF-IDF features for readability
terms = np.array(vectorizer.get_feature_names_out())

# inverse transform: approximate term weights per cluster center
# (project centroids back to TF-IDF space)
centroids_reduced = best_model.cluster_centers_
centroids_tfidf = svd.inverse_transform(centroids_reduced)

def cluster_name(row, topn=6):
    idx = np.argsort(row)[::-1][:topn]
    return " ".join(terms[idx])

names = [cluster_name(centroids_tfidf[i]) for i in range(best_k)]

# 6) Save result CSV
out = df.copy()
out["cluster_id"] = best_labels
out["cluster_name"] = [names[c] for c in best_labels]
out.to_csv(OUT, index=False)
print(f"✅ wrote: {OUT}")
print("Examples:")
print(out[["title", "cluster_id", "cluster_name"]].head(10))
