# app/make_playlists.py
import os
import time
from collections import defaultdict

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

ROOT = os.path.dirname(os.path.dirname(__file__))
CLIENT_SECRET = os.path.join(ROOT, "client_secret.json")
TOKEN_PATH = os.path.join(ROOT, "token.json")
CLUSTERS_CSV = os.path.join(ROOT, "data", "liked_videos_clusters.csv")

# WRITE scope so we can create playlists & add videos
SCOPES = ["https://www.googleapis.com/auth/youtube"]

def get_channel_id(youtube):
    resp = youtube.channels().list(part="id,snippet", mine=True).execute()
    return resp["items"][0]["id"], resp["items"][0]["snippet"]["title"]

def assert_data_matches_account(youtube):
    import json, os
    marker = os.path.join(ROOT, "data", "account_marker.json")
    cid, title = get_channel_id(youtube)

    # Create/update marker on first cluster run
    if not os.path.exists(marker):
        with open(marker, "w") as f:
            f.write(json.dumps({"channel_id": cid, "channel_title": title}, ensure_ascii=False))
        return

    prev = json.load(open(marker))
    if prev["channel_id"] != cid:
        raise RuntimeError(
            f"Data was generated for channel {prev['channel_title']} ({prev['channel_id']}) "
            f"but you are authenticated as {title} ({cid}). "
            "Delete data/*.csv and re-run fetch + cluster."
        )


def get_service():
    print(f"[auth] using client_secret: {CLIENT_SECRET}")
    if not os.path.exists(CLIENT_SECRET):
        raise FileNotFoundError("client_secret.json not found at project root.")

    creds = None
    if os.path.exists(TOKEN_PATH):
        print(f"[auth] found existing token: {TOKEN_PATH}")
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("[auth] refreshing existing token…")
            creds.refresh(Request())
        else:
            print("[auth] running OAuth flow in browser…")
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())
            print(f"[auth] wrote new token: {TOKEN_PATH}")

    # Extra check: show granted scopes
    print(f"[auth] granted scopes: {getattr(creds, 'scopes', None)}")
    return build("youtube", "v3", credentials=creds)

def list_my_playlists(youtube):
    mapping = {}
    pageToken = None
    total = 0
    while True:
        resp = youtube.playlists().list(
            part="snippet,contentDetails",
            mine=True,
            maxResults=50,
            pageToken=pageToken
        ).execute()
        items = resp.get("items", [])
        for item in items:
            title = item["snippet"]["title"].strip()
            mapping[title.lower()] = item["id"]
            total += 1
        pageToken = resp.get("nextPageToken")
        if not pageToken:
            break
    print(f"[list] you currently have {total} playlists")
    return mapping

def create_playlist(youtube, title, description="Auto-created by yt-like-sorter"):
    print(f"[create] playlist: {title!r}")
    req = youtube.playlists().insert(
        part="snippet,status",
        body={
            "snippet": {"title": title[:148], "description": description[:4900]},
            "status": {"privacyStatus": "private"}  # change if you want
        }
    )
    res = req.execute()
    pid = res["id"]
    print(f"[create] created id={pid}")
    return pid

def add_video_to_playlist(youtube, playlist_id, video_id):
    try:
        youtube.playlistItems().insert(
            part="snippet",
            body={
                "snippet": {
                    "playlistId": playlist_id,
                    "resourceId": {"kind": "youtube#video", "videoId": video_id}
                }
            }
        ).execute()
        return True
    except HttpError as e:
        print(f"[add] skip {video_id}: {e}")
        return False

def safe_name(name: str) -> str:
    name = (name or "untitled").strip()
    return " ".join(name.split())[:148]

def main():
    print(f"[start] clusters csv: {CLUSTERS_CSV}")
    if not os.path.exists(CLUSTERS_CSV):
        raise FileNotFoundError("Run cluster_with_sklearn.py or cluster_auto.py first to create liked_videos_clusters.csv")

    df = pd.read_csv(CLUSTERS_CSV)
    print(f"[csv] rows={len(df)} cols={list(df.columns)}")

    youtube = get_service()
    assert_data_matches_account(youtube)

    # sanity: required columns
    for col in ("cluster_name", "video_id"):
        if col not in df.columns:
            raise ValueError(f"CSV must contain '{col}' column.")

    # clean rows
    df["video_id"] = df["video_id"].astype(str).str.strip()
    df = df[df["video_id"].notna() & (df["video_id"] != "")]
    if df.empty:
        print("[csv] no valid video_id rows found; nothing to do.")
        return

    # group by cluster
    groups = defaultdict(list)
    for _, row in df.iterrows():
        cname = safe_name(str(row["cluster_name"]))
        vid = row["video_id"]
        groups[cname].append(vid)

    if not groups:
        print("[csv] grouped result is empty; nothing to do.")
        return

    youtube = get_service()
    existing = list_my_playlists(youtube)  # title_lower -> id
    title_to_id = dict(existing)

    print(f"[plan] will process {len(groups)} clusters")
    for cname, vids in groups.items():
        print(f"\n=== {cname} ({len(vids)} videos) ===")
        pid = title_to_id.get(cname.lower())
        if not pid:
            pid = create_playlist(youtube, cname, "Clustered automatically from liked videos")
            title_to_id[cname.lower()] = pid
        else:
            print(f"[reuse] playlist exists id={pid}")

        added = 0
        for v in vids:
            if add_video_to_playlist(youtube, pid, v):
                added += 1
            time.sleep(0.1)  # polite pacing
        print(f"[done] added {added}/{len(vids)}")

    print("\n✅ Done! Check your YouTube playlists.")

if __name__ == "__main__":
    main()
