import os, re, numpy as np, pandas as pd
from collections import defaultdict

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP
import hdbscan

# ---------- paths ----------
ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

RAW = os.path.join(DATA_DIR, "liked_videos_raw.csv")
OUT = os.path.join(DATA_DIR, "liked_videos_clusters.csv")

if not os.path.exists(RAW):
    raise FileNotFoundError(f"Missing {RAW}. Run: python3 app/fetch_liked_videos.py")

# ---------- helpers ----------
def safe(x): 
    return "" if pd.isna(x) else str(x)

def basic_clean(s: str) -> str:
    s = s.lower()
    s = re.sub(r"http\S+", " ", s)                # remove URLs
    s = re.sub(r"[^0-9a-z가-힣\s]", " ", s)       # keep en+kr letters/numbers/spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Minimal bilingual stopwords (extend later if needed)
KR_SW = set("은 는 이 가 을 를 에서 에 에게 와 과 도 으로 의 하고 보다 보다도 및 또 또한 그리고 그래서 그러나 하지만 입니다 합니다 되다 있다 없다 너무 정말 많이 매일 그냥 이런 저런 그런".split())
EN_SW = set("the a an and or of to for with on in is are be from at by this that it as you your i we they their our us my me".split())

def tokenize(s: str):
    toks = s.split()
    return [t for t in toks if t not in KR_SW and t not in EN_SW and len(t) > 1]

# ---------- load data ----------
df = pd.read_csv(RAW)
texts_raw = (df["title"].map(safe) + " " + df["description"].map(safe)).apply(basic_clean).tolist()
N = len(df)

# ---------- multilingual embeddings ----------
print("Encoding (multilingual)…")
model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
emb = model.encode(texts_raw, batch_size=32, show_progress_bar=True)
X = normalize(np.array(emb))

# ---------- UMAP reduction + HDBSCAN clustering ----------
# keep these safely below N
n_comp = max(2, min(15, N - 2))       # must be < N
n_nei  = max(2, min(15, N - 1))       # must be < N

print("UMAP → HDBSCAN…")
reducer = UMAP(
    n_neighbors=n_nei,
    n_components=n_comp,
    min_dist=0.0,
    metric="cosine",     # good for text embeddings
    init="random",       # avoid spectral init (no eigsh issues)
    random_state=42,
)
Xr = reducer.fit_transform(X)

# cluster size relative to N with bounds
min_cluster = max(5, int(N * 0.03))
min_cluster = min(min_cluster, max(2, N // 2))

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=min_cluster,
    min_samples=min(10, max(3, n_nei // 2)),
    metric="euclidean"
)
labels = clusterer.fit_predict(Xr)   # -1 = noise
df["cluster_id"] = labels

# Fallback if everything is noise or a single cluster
if (set(labels) == {-1}) or (len(set(labels)) <= 1):
    from sklearn.cluster import KMeans
    print("HDBSCAN resulted in only noise/single cluster; falling back to KMeans(k=3).")
    k_for_fallback = min(3, max(2, N - 1))
    km = KMeans(n_clusters=k_for_fallback, n_init="auto", random_state=42)
    labels = km.fit_predict(X)
    df["cluster_id"] = labels

# ---------- auto-name clusters (c-TF-IDF-ish) ----------
docs_by_cluster = defaultdict(list)
for i, c in enumerate(df["cluster_id"]):
    if c != -1:
        docs_by_cluster[int(c)].append(texts_raw[i])

# Fit vocabulary on full corpus so distinctiveness has meaning
tfidf = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, max_features=3000)
tfidf.fit(texts_raw)

cluster_names = {}
terms = np.array(tfidf.get_feature_names_out())

for c, docs in docs_by_cluster.items():
    big_doc = " ".join(docs)
    vec = tfidf.transform([big_doc]).toarray()[0]
    top = terms[np.argsort(vec)[::-1][:5]]
    label = " ".join(top)
    label = " / ".join([w.capitalize() for w in label.split()])
    cluster_names[c] = label if label else f"Cluster {c}"

def pretty_name(cid):
    if cid == -1:
        return "Misc"
    return cluster_names.get(int(cid), f"Cluster {cid}")

df["cluster_name"] = df["cluster_id"].apply(pretty_name).str[:148]  # keep playlist title short

# ---------- save ----------
df.to_csv(OUT, index=False)
print("Clusters:", sorted(set(df["cluster_id"])))
print("Names:")
for c in sorted(cluster_names.keys()):
    print(f"  {c}: {cluster_names[c]}")
print(f"✅ wrote: {OUT}")
