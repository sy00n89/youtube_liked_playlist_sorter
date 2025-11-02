import os
import time
import pandas as pd
from typing import List, Dict
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

SCOPES = ['https://www.googleapis.com/auth/youtube.readonly']
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
TOKEN_PATH = os.path.join(ROOT, 'token.json')
CLIENT_SECRET_PATH = os.path.join(ROOT, 'client_secret.json')

CATEGORY_KEYWORDS = {
    "요리 / Cooking & Food": [
        "cook","recipe","kitchen","food","baking","meal prep",
        "요리","레시피","음식","간단","맛있","쿠킹"
    ],
    "음악 / Music": [
        "music","song","lyrics","cover","piano","guitar","concert",
        "뮤직","노래","가사","커버","피아노","기타","콘서트"
    ],
    "코딩 / Coding & Tech": [
        "python","coding","programming","javascript","tutorial","ai","cloud","aws","azure","gcp",
        "코딩","프로그래밍","자바스크립트","파이썬","튜토리얼","인공지능","클라우드"
    ],
    "운동 / Fitness & Health": [
        "workout","fitness","exercise","gym","diet","health",
        "운동","헬스","다이어트","식단","체중","홈트"
    ],
    "공부 / Productivity & Study": [
        "productivity","study","notion","time management","focus","pomodoro",
        "공부","집중","시간관리","노션","포모도로"
    ],
    "뉴스&주식 / News & Stocks": [
        "news","update","commentary","analysis","explained","stocks"
        "뉴스","해설","분석","시사","주식","ETF"
    ],
    "브이로그 / Lifestyle & Vlog": [
        "vlog","routine","day in the life","lifestyle",
        "브이로그","브로그","일상","루틴","라이프스타일"
    ],
}


def get_credentials():
    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None
        if not creds:
            if not os.path.exists(CLIENT_SECRET_PATH):
                raise FileNotFoundError('Missing client_secret.json in project root.')
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, 'w') as f:
            f.write(creds.to_json())
    return creds

def fetch_liked_videos(youtube) -> List[Dict]:
    all_items = []
    page_token = None
    while True:
        req = youtube.videos().list(part='snippet,contentDetails,statistics', myRating='like', maxResults=50, pageToken=page_token)
        resp = req.execute()
        items = resp.get('items', [])
        all_items.extend(items)
        page_token = resp.get('nextPageToken')
        if not page_token:
            break
        time.sleep(0.1)
    return all_items

def save_raw_csv(items: List[Dict], path: str):
    rows = []
    for it in items:
        vid = it.get('id')
        sn = it.get('snippet', {})
        stats = it.get('statistics', {})
        cd = it.get('contentDetails', {})
        rows.append({
            'video_id': vid,
            'title': sn.get('title', ''),
            'description': sn.get('description', ''),
            'channel_title': sn.get('channelTitle', ''),
            'published_at': sn.get('publishedAt', ''),
            'tags': '|'.join(sn.get('tags', [])) if sn.get('tags') else '',
            'duration': cd.get('duration', ''),
            'view_count': stats.get('viewCount', ''),
            'like_count': stats.get('likeCount', ''),
            'comment_count': stats.get('commentCount', ''),
            'url': f'https://www.youtube.com/watch?v={vid}' if vid else '',
        })
    pd.DataFrame(rows).to_csv(path, index=False)

def simple_categorize(row: Dict) -> str:
    haystack = ' '.join([
        str(row.get('title','')),
        str(row.get('description','')),
        str(row.get('channel_title','')),
        str(row.get('tags','')),
    ]).lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in haystack:
                return cat
    return 'Uncategorized'

def main():
    creds = get_credentials()
    youtube = build('youtube', 'v3', credentials=creds)
    print('Fetching your liked videos...')
    items = fetch_liked_videos(youtube)
    print(f'Fetched {len(items)} videos. Saving raw CSV...')
    raw_csv = os.path.join(DATA_DIR, 'liked_videos_raw.csv')
    save_raw_csv(items, raw_csv)
    print(f'Wrote {raw_csv}')
    print('Adding simple categories...')
    df = pd.read_csv(raw_csv)
    df['category'] = df.apply(simple_categorize, axis=1)
    out_csv = os.path.join(DATA_DIR, 'liked_videos_categorized.csv')
    df.to_csv(out_csv, index=False)
    print(f'Wrote {out_csv}')
    print('Done! Open the CSV to see your categories.')

if __name__ == '__main__':
    main()
