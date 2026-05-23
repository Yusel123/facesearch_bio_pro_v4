#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FaceSearch Global Pro v5.0 - Cloud-Ready (ohne TensorFlow)"""

import asyncio
import aiohttp
import streamlit as st
import requests
import time
import os
import io
import hashlib
import json
import sqlite3
import tempfile
import base64
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass, field, asdict
from urllib.parse import quote_plus, urlparse, urljoin
from collections import defaultdict
from enum import Enum
import numpy as np
from PIL import Image
from fpdf import FPDF
import plotly.graph_objects as go
import pandas as pd
import cv2

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False

try:
    from fake_useragent import UserAgent
    HAS_FAKE_UA = True
except ImportError:
    HAS_FAKE_UA = False

APP_DIR = Path(tempfile.gettempdir()) / "facesearch_pro"
APP_DIR.mkdir(exist_ok=True)
DB_PATH = APP_DIR / "face_index_v5.db"
EMBEDDING_DIM = 512
SIMILARITY_THRESHOLD = 0.65
RATE_LIMIT_PER_DOMAIN = 2

PROXY_POOL = os.getenv("PROXY_POOL", "").split(",") if os.getenv("PROXY_POOL") else []

def get_api_key(key_name: str) -> Optional[str]:
    try:
        return st.secrets[key_name]
    except (KeyError, FileNotFoundError, AttributeError):
        return os.getenv(key_name)

TWITTER_BEARER_TOKEN = get_api_key("TWITTER_BEARER_TOKEN")
YOUTUBE_API_KEY      = get_api_key("YOUTUBE_API_KEY")
REDDIT_CLIENT_ID     = get_api_key("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = get_api_key("REDDIT_CLIENT_SECRET")
TIKAPI_KEY           = get_api_key("TIKAPI_KEY")
INSTAGRAM_SESSION    = get_api_key("INSTAGRAM_SESSION")
FACEBOOK_TOKEN       = get_api_key("FACEBOOK_TOKEN")

HAS_TWITTER = bool(TWITTER_BEARER_TOKEN)
HAS_YOUTUBE = bool(YOUTUBE_API_KEY)
HAS_REDDIT  = bool(REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET)
HAS_TIKTOK  = bool(TIKAPI_KEY)
HAS_INSTAGRAM = bool(INSTAGRAM_SESSION)
HAS_FACEBOOK = bool(FACEBOOK_TOKEN)

def init_session():
    defaults = {
        'dark_mode': False,
        'search_history': [],
        'bookmarks': [],
        'legal_accepted': False,
        'match_votes': {},
        'last_results': [],
        'search_count': 0,
        'last_query_name': "",
        'ref_embedding': None,
        'ref_face_hash': "",
        'bio_enabled': True,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()

class ErrorCategory(Enum):
    NETWORK = "network"
    API_RATE_LIMIT = "rate_limit"
    API_AUTH = "auth"
    BIOMETRIC = "biometric"
    DATABASE = "database"
    PARSING = "parsing"
    PROXY = "proxy"

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str
    thumbnail_url: Optional[str] = None
    match_score: Optional[float] = None
    platform: str = ""
    is_match: Optional[bool] = None
    face_similarity: Optional[float] = None
    face_detected: bool = False
    found_via: str = "api"

    def to_dict(self) -> dict:
        return {
            'Titel': self.title,
            'URL': self.url,
            'Quelle': self.source,
            'Plattform': self.platform,
            'Text-Score (%)': self.match_score if self.match_score else '-',
            'Bio-Score (%)': round(self.face_similarity * 100, 1) if self.face_similarity else '-',
            'Gesicht erkannt': 'Ja' if self.face_detected else 'Nein',
            'Gefunden via': self.found_via,
            'Treffer': 'Ja' if self.is_match is True else ('Nein' if self.is_match is False else 'Unbewertet')
        }

@dataclass
class SearchQuery:
    image_path: Optional[str] = None
    person_name: Optional[str] = None
    engines: List[str] = field(default_factory=list)
    max_results: int = 20

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures: Dict[str, int] = defaultdict(int)
        self.last_failure: Dict[str, datetime] = {}
        self.state: Dict[str, str] = defaultdict(lambda: "closed")

    def call(self, platform: str, func, *args, **kwargs):
        if self.state[platform] == "open":
            if datetime.now() - self.last_failure.get(platform, datetime.min) > timedelta(seconds=self.recovery_timeout):
                self.state[platform] = "half-open"
            else:
                return []
        try:
            result = func(*args, **kwargs)
            if self.state[platform] == "half-open":
                self.state[platform] = "closed"
                self.failures[platform] = 0
            return result
        except Exception as e:
            self.failures[platform] += 1
            self.last_failure[platform] = datetime.now()
            if self.failures[platform] >= self.failure_threshold:
                self.state[platform] = "open"
            raise

circuit_breaker = CircuitBreaker()

class RateLimiter:
    def __init__(self, rate_per_second: float = 2.0):
        self.rate = rate_per_second
        self.tokens: Dict[str, float] = defaultdict(lambda: rate_per_second)
        self.last_update: Dict[str, float] = defaultdict(time.time)
        self.lock = asyncio.Lock()

    async def acquire(self, domain: str):
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update[domain]
            self.tokens[domain] = min(self.rate, self.tokens[domain] + elapsed * self.rate)
            self.last_update[domain] = now
            if self.tokens[domain] < 1.0:
                wait_time = (1.0 - self.tokens[domain]) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens[domain] = 0
            else:
                self.tokens[domain] -= 1.0

rate_limiter = RateLimiter(RATE_LIMIT_PER_DOMAIN)

class ConnectionManager:
    def __init__(self):
        self.ua = UserAgent() if HAS_FAKE_UA else None
        self.proxy_pool = [p.strip() for p in PROXY_POOL if p.strip()]
        self.proxy_index = 0
        self.session: Optional[aiohttp.ClientSession] = None

    def get_headers(self) -> Dict[str, str]:
        if self.ua:
            return {
                "User-Agent": self.ua.random,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Cache-Control": "max-age=0",
            }
        return {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    def get_proxy(self) -> Optional[str]:
        if not self.proxy_pool:
            return None
        proxy = self.proxy_pool[self.proxy_index % len(self.proxy_pool)]
        self.proxy_index += 1
        return proxy

    async def get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=10, ttl_dns_cache=300)
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self.session

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()

conn_manager = ConnectionManager()

class VectorDatabase:
    def __init__(self, db_path, dim=512):
        self.db_path = db_path
        self.dim = dim
        self.index = None
        self.urls = []
        self.metadata = {}
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS embeddings_meta (
                url TEXT PRIMARY KEY,
                source TEXT,
                platform TEXT,
                face_detected INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS search_cache (
                query_hash TEXT PRIMARY KEY,
                results_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(self.dim)
        else:
            self._fallback_embeddings = {}

    def save_embedding(self, url, source, platform, embedding, face_detected):
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        if HAS_FAISS and self.index is not None:
            emb_array = embedding.astype(np.float32).reshape(1, -1)
            self.index.add(emb_array)
            self.urls.append(url)
        else:
            self._fallback_embeddings[url] = embedding

        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO embeddings_meta (url, source, platform, face_detected)
            VALUES (?, ?, ?, ?)
        """, (url, source, platform, 1 if face_detected else 0))
        conn.commit()
        conn.close()
        self.metadata[url] = {"source": source, "platform": platform, "face_detected": face_detected}

    def find_similar(self, ref_embedding, top_k=20, threshold=0.6):
        norm = np.linalg.norm(ref_embedding)
        if norm > 0:
            ref_embedding = ref_embedding / norm

        if HAS_FAISS and self.index is not None and self.index.ntotal > 0:
            query = ref_embedding.astype(np.float32).reshape(1, -1)
            distances, indices = self.index.search(query, min(top_k, self.index.ntotal))
            results = []
            for i, dist in zip(indices[0], distances[0]):
                if dist >= threshold:
                    results.append((self.urls[i], float(dist)))
            return results
        else:
            results = []
            for url, emb in self._fallback_embeddings.items():
                sim = float(np.dot(ref_embedding, emb))
                if sim >= threshold:
                    results.append((url, sim))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

    def get_stats(self):
        if HAS_FAISS and self.index is not None:
            return {"total_vectors": self.index.ntotal, "backend": "FAISS", "dim": self.dim}
        return {"total_vectors": len(self._fallback_embeddings), "backend": "Fallback", "dim": self.dim}

    def save_search_cache(self, query_hash, results):
        serializable = []
        for r in results:
            d = {
                'title': r.title, 'url': r.url, 'snippet': r.snippet,
                'source': r.source, 'platform': r.platform,
                'thumbnail_url': r.thumbnail_url,
                'match_score': r.match_score,
                'face_similarity': r.face_similarity,
                'face_detected': r.face_detected,
                'found_via': r.found_via
            }
            serializable.append(d)
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO search_cache (query_hash, results_json)
            VALUES (?, ?)
        """, (query_hash, json.dumps(serializable)))
        conn.commit()
        conn.close()

    def get_search_cache(self, query_hash):
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        c.execute("SELECT results_json FROM search_cache WHERE query_hash = ?", (query_hash,))
        row = c.fetchone()
        conn.close()
        if row:
            return json.loads(row[0])
        return None

vector_db = VectorDatabase(DB_PATH, EMBEDDING_DIM)

class BiometricAnalyzer:
    """OpenCV-basierte Gesichtserkennung ohne TensorFlow/DeepFace."""
    def __init__(self):
        self.batch_size = 8
        self.face_cascade = None
        self._load_classifier()
        self.dnn_net = None
        self._init_dnn()

    def _load_classifier(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(cascade_path):
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def _init_dnn(self):
        try:
            prototxt = os.path.expanduser("~/.opencv/face_detector/deploy.prototxt")
            model = os.path.expanduser("~/.opencv/face_detector/res10_300x300_ssd_iter_140000.caffemodel")
            if os.path.exists(prototxt) and os.path.exists(model):
                self.dnn_net = cv2.dnn.readNetFromCaffe(prototxt, model)
        except:
            pass

    def detect_face(self, img_path):
        if self.face_cascade is None:
            return False
        img = cv2.imread(img_path)
        if img is None:
            return False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return len(faces) > 0

    def extract_embedding(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        img = cv2.resize(img, (128, 128))
        
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        embedding = np.interp(
            np.linspace(0, len(hist) - 1, EMBEDDING_DIM),
            np.arange(len(hist)),
            hist
        ).astype(np.float32)
        
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding

    def extract_from_bytes(self, img_bytes):
        if not img_bytes:
            return None
        try:
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return None
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                cv2.imwrite(tmp.name, img)
                emb = self.extract_embedding(tmp.name)
                os.unlink(tmp.name)
                return emb
        except Exception as e:
            st.error(f"Embedding-Fehler: {e}")
        return None

    def compare_embeddings(self, emb1, emb2):
        if emb1 is None or emb2 is None:
            return 0.0
        n1 = emb1 / (np.linalg.norm(emb1) + 1e-10)
        n2 = emb2 / (np.linalg.norm(emb2) + 1e-10)
        return float(np.dot(n1, n2))

    async def batch_extract(self, img_bytes_list):
        results = []
        for i in range(0, len(img_bytes_list), self.batch_size):
            batch = img_bytes_list[i:i+self.batch_size]
            batch_results = await asyncio.gather(*[
                asyncio.to_thread(self.extract_from_bytes, b) for b in batch
            ])
            results.extend(batch_results)
        return results

bio_analyzer = BiometricAnalyzer()
HAS_BIOMETRIE = True

async def fetch_url(url, session, headers, proxy=None, timeout=15):
    domain = urlparse(url).netloc
    await rate_limiter.acquire(domain)
    try:
        async with session.get(url, headers=headers, proxy=proxy, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            content = await resp.read() if resp.status == 200 else None
            return resp.status, content, dict(resp.headers)
    except asyncio.TimeoutError:
        return 408, None, {}
    except Exception as e:
        return 0, None, {}

async def scrape_instagram_profile(session, username):
    results = []
    try:
        oembed_url = f"https://api.instagram.com/oembed?url=https://www.instagram.com/{username}/"
        status, content, _ = await fetch_url(oembed_url, session, conn_manager.get_headers())
        if status == 200 and content:
            data = json.loads(content)
            results.append(SearchResult(
                title=data.get("title", f"@{username} auf Instagram"),
                url=f"https://www.instagram.com/{username}/",
                snippet=data.get("author_name", "Instagram Profil"),
                source="instagram", platform="Instagram",
                thumbnail_url=data.get("thumbnail_url", ""), match_score=60, found_via="scraping"
            ))
    except Exception as e:
        pass

    try:
        profile_url = f"https://www.instagram.com/{username}/"
        status, content, _ = await fetch_url(profile_url, session, conn_manager.get_headers())
        if status == 200 and content:
            html = content.decode('utf-8', errors='ignore')
            import re
            og_title = re.search(r'<meta property="og:title" content="([^"]+)"', html)
            og_image = re.search(r'<meta property="og:image" content="([^"]+)"', html)
            og_desc = re.search(r'<meta property="og:description" content="([^"]+)"', html)
            if og_title or og_image:
                results.append(SearchResult(
                    title=og_title.group(1) if og_title else f"@{username} auf Instagram",
                    url=profile_url,
                    snippet=og_desc.group(1)[:200] if og_desc else "Instagram Profil",
                    source="instagram", platform="Instagram",
                    thumbnail_url=og_image.group(1) if og_image else "", match_score=55, found_via="scraping"
                ))
    except Exception as e:
        pass
    return results

async def scrape_tiktok_profile(session, username):
    results = []
    try:
        url = f"https://www.tiktok.com/@{username}"
        status, content, _ = await fetch_url(url, session, conn_manager.get_headers())
        if status == 200 and content:
            html = content.decode('utf-8', errors='ignore')
            import re
            data_match = re.search(r'<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__"[^>]*>(.*?)</script>', html)
            if data_match:
                try:
                    data = json.loads(data_match.group(1))
                    user_info = data.get("__DEFAULT_SCOPE__", {}).get("webapp.user-detail", {}).get("userInfo", {})
                    user = user_info.get("user", {})
                    stats = user_info.get("stats", {})
                    results.append(SearchResult(
                        title=f"@{user.get('uniqueId', username)} auf TikTok",
                        url=url,
                        snippet=f"{user.get('nickname', '')} | {stats.get('followerCount', 0)} Follower | {stats.get('videoCount', 0)} Videos",
                        source="tiktok", platform="TikTok",
                        thumbnail_url=user.get("avatarLarger", ""), match_score=65, found_via="scraping"
                    ))
                except:
                    pass
            if not results:
                og_image = re.search(r'<meta property="og:image" content="([^"]+)"', html)
                og_title = re.search(r'<meta property="og:title" content="([^"]+)"', html)
                if og_title:
                    results.append(SearchResult(
                        title=og_title.group(1), url=url, snippet="TikTok Profil",
                        source="tiktok", platform="TikTok",
                        thumbnail_url=og_image.group(1) if og_image else "", match_score=50, found_via="scraping"
                    ))
    except Exception as e:
        pass
    return results

async def scrape_twitter_profile(session, username):
    results = []
    nitter_instances = ["https://nitter.net", "https://nitter.it", "https://nitter.cz"]
    for instance in nitter_instances:
        try:
            url = f"{instance}/{username}"
            status, content, _ = await fetch_url(url, session, conn_manager.get_headers(), timeout=10)
            if status == 200 and content:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(content, 'lxml')
                profile_pic = soup.find('img', class_='profile-card-avatar')
                bio = soup.find('div', class_='profile-bio')
                if profile_pic:
                    results.append(SearchResult(
                        title=f"@{username} auf Twitter/X",
                        url=f"https://twitter.com/{username}",
                        snippet=bio.get_text(strip=True)[:200] if bio else "Twitter Profil",
                        source="twitter", platform="Twitter/X",
                        thumbnail_url=urljoin(instance, profile_pic.get('src', '')) if profile_pic else "",
                        match_score=60, found_via="scraping"
                    ))
                    break
        except Exception as e:
            pass
    return results

async def scrape_reddit_profile(session, username):
    results = []
    try:
        url = f"https://www.reddit.com/user/{username}/about.json"
        headers = {**conn_manager.get_headers(), "User-Agent": "FaceSearchBot/5.0"}
        status, content, _ = await fetch_url(url, session, headers)
        if status == 200 and content:
            data = json.loads(content)
            user_data = data.get("data", {})
            results.append(SearchResult(
                title=f"u/{username} auf Reddit",
                url=f"https://reddit.com/user/{username}",
                snippet=f"Karma: {user_data.get('total_karma', 0)} | Account: {datetime.fromtimestamp(user_data.get('created_utc', 0)).strftime('%Y')}",
                source="reddit", platform="Reddit",
                thumbnail_url=user_data.get("snoovatar_img", user_data.get("icon_img", "")),
                match_score=65, found_via="scraping"
            ))
    except Exception as e:
        pass
    return results

async def scrape_pinterest_profile(session, name):
    results = []
    try:
        url = f"https://www.pinterest.com/{name.lower().replace(' ', '')}/"
        status, content, _ = await fetch_url(url, session, conn_manager.get_headers())
        if status == 200 and content:
            html = content.decode('utf-8', errors='ignore')
            import re
            data_match = re.search(r'<script id="__PWS_DATA__"[^>]*>(.*?)</script>', html)
            if data_match:
                try:
                    data = json.loads(data_match.group(1))
                    user = data.get("props", {}).get("initialReduxState", {}).get("profiles", {}).get(name.lower().replace(' ', ''), {})
                    results.append(SearchResult(
                        title=f"{user.get('fullName', name)} auf Pinterest",
                        url=url,
                        snippet=f"{user.get('followerCount', 0)} Follower | {user.get('pinCount', 0)} Pins",
                        source="pinterest", platform="Pinterest",
                        thumbnail_url=user.get("imageLargeUrl", ""), match_score=60, found_via="scraping"
                    ))
                except:
                    pass
            if not results:
                og_image = re.search(r'<meta property="og:image" content="([^"]+)"', html)
                og_title = re.search(r'<meta property="og:title" content="([^"]+)"', html)
                if og_title:
                    results.append(SearchResult(
                        title=og_title.group(1), url=url, snippet="Pinterest Profil",
                        source="pinterest", platform="Pinterest",
                        thumbnail_url=og_image.group(1) if og_image else "", match_score=50, found_via="scraping"
                    ))
    except Exception as e:
        pass
    return results

async def scrape_facebook_profile(session, name):
    results = []
    if HAS_FACEBOOK:
        try:
            url = f"https://graph.facebook.com/v18.0/search?q={quote_plus(name)}&type=page&access_token={FACEBOOK_TOKEN}"
            status, content, _ = await fetch_url(url, session, conn_manager.get_headers())
            if status == 200 and content:
                data = json.loads(content)
                for page in data.get("data", [])[:3]:
                    results.append(SearchResult(
                        title=page.get("name", "Facebook"),
                        url=f"https://facebook.com/{page.get('id', '')}",
                        snippet=f"Facebook Seite | Kategorie: {page.get('category', 'N/A')}",
                        source="facebook", platform="Facebook",
                        thumbnail_url=f"https://graph.facebook.com/{page['id']}/picture?type=large",
                        match_score=70, found_via="api"
                    ))
        except Exception as e:
            pass
    if not results:
        results.append(SearchResult(
            title=f"{name} auf Facebook",
            url=f"https://www.facebook.com/search/top/?q={quote_plus(name)}",
            snippet="Facebook-Suche", source="facebook", platform="Facebook",
            thumbnail_url="", match_score=40, found_via="fallback"
        ))
    return results

async def scrape_linkedin_profile(session, name):
    results = []
    try:
        cache_url = f"https://webcache.googleusercontent.com/search?q=site:linkedin.com/in+{quote_plus(name)}"
        status, content, _ = await fetch_url(cache_url, session, conn_manager.get_headers())
        if status == 200 and content:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, 'lxml')
            links = soup.find_all('a', href=re.compile(r'linkedin.com/in/'))
            for link in links[:3]:
                href = link.get('href', '')
                if 'linkedin.com/in/' in href:
                    results.append(SearchResult(
                        title=f"{name} auf LinkedIn", url=href, snippet="LinkedIn Profil (via Cache)",
                        source="linkedin", platform="LinkedIn",
                        thumbnail_url="", match_score=45, found_via="cache"
                    ))
    except Exception as e:
        pass
    if not results:
        results.append(SearchResult(
            title=f"{name} auf LinkedIn",
            url=f"https://www.linkedin.com/search/results/people/?keywords={quote_plus(name)}",
            snippet="LinkedIn-Suche", source="linkedin", platform="LinkedIn",
            thumbnail_url="", match_score=40, found_via="fallback"
        ))
    return results

async def scrape_youtube_channel(session, name):
    results = []
    try:
        rss_url = f"https://www.youtube.com/feeds/videos.xml?user={name.lower().replace(' ', '')}"
        status, content, _ = await fetch_url(rss_url, session, conn_manager.get_headers())
        if status == 200 and content:
            from xml.etree import ElementTree as ET
            root = ET.fromstring(content)
            ns = {"atom": "http://www.w3.org/2005/Atom", "media": "http://search.yahoo.com/mrss/"}
            entries = root.findall(".//atom:entry", ns)[:3]
            for entry in entries:
                title = entry.find("atom:title", ns)
                link = entry.find("atom:link", ns)
                media = entry.find(".//media:thumbnail", ns)
                if title is not None:
                    results.append(SearchResult(
                        title=title.text,
                        url=link.get("href") if link is not None else "",
                        snippet="YouTube Video (via RSS)",
                        source="youtube", platform="YouTube",
                        thumbnail_url=media.get("url") if media is not None else "",
                        match_score=60, found_via="rss"
                    ))
    except Exception as e:
        pass
    return results

async def search_youtube_videos(session, name, max_res):
    results = []
    if HAS_YOUTUBE:
        try:
            url = "https://www.googleapis.com/youtube/v3/search"
            params = {"part": "snippet", "type": "video", "q": name, "maxResults": min(max_res, 10), "key": YOUTUBE_API_KEY}
            status, content, _ = await fetch_url(url, session, conn_manager.get_headers())
            if status == 200 and content:
                data = json.loads(content)
                for item in data.get("items", []):
                    vid = item["id"]["videoId"]
                    s = item["snippet"]
                    results.append(SearchResult(
                        title=s.get("title", "YouTube"), url=f"https://www.youtube.com/watch?v={vid}",
                        snippet=s.get("description", "")[:200], source="youtube", platform="YouTube",
                        thumbnail_url=s.get("thumbnails", {}).get("medium", {}).get("url", ""), match_score=85, found_via="api"))
        except Exception as e:
            pass
    if not results:
        rss_results = await scrape_youtube_channel(session, name)
        results.extend(rss_results)
    if not results:
        results.append(SearchResult(title=f"YouTube-Suche: {name}", url=f"https://www.youtube.com/results?search_query={quote_plus(name)}",
            snippet="Videosuche", source="youtube", platform="YouTube", thumbnail_url="", match_score=50, found_via="fallback"))
    return results

async def search_instagram_posts(session, name, max_res):
    results = []
    if HAS_DDGS:
        try:
            with DDGS() as ddgs:
                for r in ddgs.images(f'"{name}" site:instagram.com', max_results=min(max_res, 15)):
                    img = r.get("image", "")
                    results.append(SearchResult(title=r.get("title", f"Instagram: {name}"),
                        url=img or f"https://www.instagram.com/{name.lower().replace(' ', '')}/",
                        snippet="Instagram", source="instagram", platform="Instagram",
                        thumbnail_url=r.get("thumbnail", img), match_score=70, found_via="api"))
        except Exception as e:
            pass
    if not results:
        clean_name = name.lower().replace(' ', '').replace('.', '')
        scrape_results = await scrape_instagram_profile(session, clean_name)
        results.extend(scrape_results)
    if not results:
        clean_name = name.lower().replace(' ', '').replace('.', '')
        results.append(SearchResult(title=f"{name} auf Instagram", url=f"https://www.instagram.com/{clean_name}/",
            snippet="Profil", source="instagram", platform="Instagram", thumbnail_url="", match_score=45, found_via="fallback"))
    return results

async def search_tiktok_posts(session, name, max_res):
    results = []
    if HAS_DDGS:
        try:
            with DDGS() as ddgs:
                for r in ddgs.images(f'"{name}" site:tiktok.com', max_results=min(max_res, 10)):
                    img = r.get("image", "")
                    results.append(SearchResult(title=r.get("title", f"TikTok: {name}"), url=img or f"https://www.tiktok.com/@{name.lower().replace(' ', '')}",
                        snippet="TikTok", source="tiktok", platform="TikTok", thumbnail_url=r.get("thumbnail", img), match_score=68, found_via="api"))
        except Exception as e:
            pass
    if not results:
        clean_name = name.lower().replace(' ', '')
        scrape_results = await scrape_tiktok_profile(session, clean_name)
        results.extend(scrape_results)
    if not results:
        results.append(SearchResult(title=f"{name} auf TikTok", url=f"https://www.tiktok.com/@{name.lower().replace(' ', '')}",
            snippet="Profil", source="tiktok", platform="TikTok", thumbnail_url="", match_score=45, found_via="fallback"))
    return results

async def search_reddit_posts(session, name, max_res):
    results = []
    if HAS_REDDIT:
        try:
            auth = aiohttp.BasicAuth(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)
            headers = {**conn_manager.get_headers(), "User-Agent": "FaceSearchPro/5.0"}
            async with session.post("https://www.reddit.com/api/v1/access_token",
                auth=auth, data={"grant_type": "client_credentials"}, headers=headers) as resp:
                if resp.status == 200:
                    token_data = await resp.json()
                    token = token_data.get("access_token")
                    if token:
                        headers["Authorization"] = f"Bearer {token}"
                        async with session.get("https://oauth.reddit.com/search",
                            headers=headers, params={"q": name, "type": "link", "limit": min(max_res, 25)}) as resp2:
                            if resp2.status == 200:
                                data = await resp2.json()
                                for post in data.get("data", {}).get("children", []):
                                    d = post["data"]
                                    url = d.get("url", "")
                                    is_img = any(ext in url.lower() for ext in [".jpg",".jpeg",".png",".gif","imgur","i.redd.it"])
                                    results.append(SearchResult(title=d.get("title","Reddit"), url=f"https://reddit.com{d.get('permalink','')}",
                                        snippet=f"r/{d.get('subreddit','')}", source="reddit", platform="Reddit",
                                        thumbnail_url=url if is_img else "", match_score=75 if is_img else 40, found_via="api"))
        except Exception as e:
            pass
    if not results:
        clean_name = name.lower().replace(' ', '')
        scrape_results = await scrape_reddit_profile(session, clean_name)
        results.extend(scrape_results)
    if not results:
        results.append(SearchResult(title=f"Reddit-Suche: {name}", url=f"https://www.reddit.com/search/?q={quote_plus(name)}",
            snippet="Suche", source="reddit", platform="Reddit", thumbnail_url="", match_score=40, found_via="fallback"))
    return results

async def search_twitter_media(session, name, max_res):
    results = []
    clean_handle = name.lower().replace(' ', '')
    if HAS_TWITTER:
        try:
            headers = {**conn_manager.get_headers(), "Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"}
            async with session.get("https://api.twitter.com/2/users/by", headers=headers,
                params={"usernames": clean_handle, "user.fields": "profile_image_url,description"}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for u in data.get("data", []):
                        thumb = u.get("profile_image_url","").replace("_normal","_400x400")
                        results.append(SearchResult(title=f"@{u.get('username',clean_handle)} auf X/Twitter",
                            url=f"https://twitter.com/{u.get('username',clean_handle)}", snippet=u.get("description",""),
                            source="twitter", platform="Twitter/X", thumbnail_url=thumb, match_score=80, found_via="api"))
        except Exception as e:
            pass
    if not results:
        scrape_results = await scrape_twitter_profile(session, clean_handle)
        results.extend(scrape_results)
    if not results:
        results.append(SearchResult(title=f"{name} auf Twitter/X", url=f"https://twitter.com/{clean_handle}",
            snippet="Profil", source="twitter", platform="Twitter/X", thumbnail_url="", match_score=50, found_via="fallback"))
    return results

async def search_facebook_posts(session, name, max_res):
    results = await scrape_facebook_profile(session, name)
    if not results:
        results.append(SearchResult(title=f"Facebook-Suche: {name}", url=f"https://www.facebook.com/search/top/?q={quote_plus(name)}",
            snippet="Suche", source="facebook", platform="Facebook", thumbnail_url="", match_score=40, found_via="fallback"))
    return results

async def search_pinterest_posts(session, name, max_res):
    results = []
    if HAS_DDGS:
        try:
            with DDGS() as ddgs:
                for r in ddgs.images(f'"{name}" site:pinterest.com', max_results=min(max_res, 10)):
                    img = r.get("image", "")
                    results.append(SearchResult(title=r.get("title", f"Pinterest: {name}"), url=img,
                        snippet="Pinterest", source="pinterest", platform="Pinterest", thumbnail_url=r.get("thumbnail", img), match_score=65, found_via="api"))
        except Exception as e:
            pass
    if not results:
        scrape_results = await scrape_pinterest_profile(session, name)
        results.extend(scrape_results)
    if not results:
        results.append(SearchResult(title=f"Pinterest-Suche: {name}", url=f"https://www.pinterest.de/search/pins/?q={quote_plus(name)}",
            snippet="Suche", source="pinterest", platform="Pinterest", thumbnail_url="", match_score=40, found_via="fallback"))
    return results

async def search_linkedin_posts(session, name, max_res):
    results = await scrape_linkedin_profile(session, name)
    return results if results else [SearchResult(title=f"{name} auf LinkedIn", url=f"https://www.linkedin.com/search/results/people/?keywords={quote_plus(name)}",
        snippet="LinkedIn", source="linkedin", platform="LinkedIn", thumbnail_url="", match_score=45, found_via="fallback")]

async def search_duckduckgo_images(session, name, max_res):
    results = []
    if HAS_DDGS:
        try:
            with DDGS() as ddgs:
                for r in ddgs.images(name, max_results=min(max_res, 20)):
                    img = r.get("image", "")
                    results.append(SearchResult(title=r.get("title","Web"), url=img, snippet=r.get("source","Web"),
                        source="web", platform="Web", thumbnail_url=r.get("thumbnail", img), match_score=55, found_via="api"))
        except Exception as e:
            pass
    return results

async def search_news(session, name, max_res):
    results = []
    if HAS_DDGS:
        try:
            with DDGS() as ddgs:
                for r in ddgs.news(name, max_results=min(max_res, 10)):
                    results.append(SearchResult(title=r.get("title","News"), url=r.get("url",""), snippet=r.get("source","News"),
                        source="news", platform="News", thumbnail_url=r.get("image",""), match_score=50, found_via="api"))
        except Exception as e:
            pass
    if not results:
        results.append(SearchResult(title=f"News-Suche: {name}", url=f"https://news.google.com/search?q={quote_plus(name)}",
            snippet="News", source="news", platform="News", thumbnail_url="", match_score=40, found_via="fallback"))
    return results

class AsyncSearcher:
    def __init__(self):
        self.session = None

    async def search(self, query, ref_embedding=None):
        self.session = await conn_manager.get_session()

        qhash = _query_hash(query)
        cached = vector_db.get_search_cache(qhash)
        if cached:
            results = []
            for c in cached:
                r = SearchResult(**c)
                if r.face_similarity is None and ref_embedding is not None and r.thumbnail_url:
                    sim, detected = await self._bio_analyze_single(r.thumbnail_url, ref_embedding)
                    r.face_similarity = sim
                    r.face_detected = detected
                results.append(r)
            return results

        name = query.person_name or ""
        max_r = query.max_results

        engine_map = {
            "google":     lambda: self._manual_result("Google Lens", "https://lens.google.com/upload", "Bild hochladen", "google"),
            "bing":       lambda: self._manual_result("Bing Visual", "https://www.bing.com/images/search?view=detailv2&iss=sbi", "Drag & Drop", "bing"),
            "duckduckgo": lambda: search_duckduckgo_images(self.session, name, max_r),
            "youtube":    lambda: search_youtube_videos(self.session, name, max_r),
            "instagram":  lambda: search_instagram_posts(self.session, name, max_r),
            "tiktok":     lambda: search_tiktok_posts(self.session, name, max_r),
            "reddit":     lambda: search_reddit_posts(self.session, name, max_r),
            "twitter":    lambda: search_twitter_media(self.session, name, max_r),
            "facebook":   lambda: search_facebook_posts(self.session, name, max_r),
            "pinterest":  lambda: search_pinterest_posts(self.session, name, max_r),
            "linkedin":   lambda: search_linkedin_posts(self.session, name, max_r),
            "news":       lambda: search_news(self.session, name, max_r),
        }

        tasks = []
        for en in query.engines:
            if en in engine_map:
                tasks.append(asyncio.create_task(self._safe_search(en, engine_map[en])))

        results_lists = await asyncio.gather(*tasks, return_exceptions=True)
        results = []
        for rl in results_lists:
            if isinstance(rl, list):
                results.extend(rl)

        if ref_embedding is not None and HAS_BIOMETRIE:
            await self._batch_bio_analyze(results, ref_embedding)

        def combined_score(r):
            text = r.match_score or 0
            bio = (r.face_similarity or 0) * 100
            if r.face_detected:
                return bio * 0.7 + text * 0.3
            return text

        results.sort(key=combined_score, reverse=True)
        vector_db.save_search_cache(qhash, results)
        return results

    async def _safe_search(self, engine_name, func):
        try:
            return await circuit_breaker.call(engine_name, func)
        except Exception as e:
            return []

    async def _batch_bio_analyze(self, results, ref_embedding):
        thumbs = [(i, r.thumbnail_url) for i, r in enumerate(results) if r.thumbnail_url]
        for batch_start in range(0, len(thumbs), bio_analyzer.batch_size):
            batch = thumbs[batch_start:batch_start + bio_analyzer.batch_size]
            img_tasks = [fetch_url(url, self.session, conn_manager.get_headers()) for _, url in batch]
            img_results = await asyncio.gather(*img_tasks)
            img_bytes = [content for _, content, _ in img_results if content]
            embeddings = await bio_analyzer.batch_extract(img_bytes)
            for (idx, url), emb in zip(batch, embeddings):
                if emb is not None:
                    sim = bio_analyzer.compare_embeddings(ref_embedding, emb)
                    results[idx].face_similarity = sim
                    results[idx].face_detected = True
                    vector_db.save_embedding(url, results[idx].source, results[idx].platform, emb, True)
                else:
                    results[idx].face_detected = False

    async def _bio_analyze_single(self, url, ref_embedding):
        status, content, _ = await fetch_url(url, self.session, conn_manager.get_headers())
        if status == 200 and content:
            emb = bio_analyzer.extract_from_bytes(content)
            if emb is not None:
                sim = bio_analyzer.compare_embeddings(ref_embedding, emb)
                vector_db.save_embedding(url, "", "", emb, True)
                return sim, True
        return None, False

    def _manual_result(self, title, url, snippet, source):
        return [SearchResult(title=title, url=url, snippet=snippet, source=source, platform=source.title(), match_score=None, found_via="manual")]

    async def close(self):
        await conn_manager.close()

def save_temp_image(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        tmp = APP_DIR / "uploads"
        tmp.mkdir(exist_ok=True)
        name = hashlib.md5(uploaded_file.getvalue()).hexdigest() + ".jpg"
        path = tmp / name
        with open(path, "wb") as f:
            f.write(uploaded_file.getvalue())
        return str(path)
    except Exception as e:
        st.error(f"Fehler beim Speichern: {e}")
        return None

def add_to_history(query):
    if not query:
        return
    hist = st.session_state.search_history
    if query in hist:
        hist.remove(query)
    hist.insert(0, query)
    if len(hist) > 15:
        hist.pop()

def toggle_bookmark(result):
    bm_list = st.session_state.bookmarks
    existing = next((b for b in bm_list if b["url"] == result.url), None)
    if existing:
        bm_list.remove(existing)
        return False
    else:
        bm_list.append({
            "title": result.title,
            "url": result.url,
            "source": result.source,
            "thumbnail": result.thumbnail_url
        })
        return True

def _make_key(url):
    return hashlib.md5(url.encode("utf-8")).hexdigest()[:12]

def _query_hash(query):
    raw = f"{query.image_path}|{query.person_name}|{','.join(sorted(query.engines))}|{query.max_results}"
    return hashlib.md5(raw.encode()).hexdigest()

def legal_notice():
    if not st.session_state.legal_accepted:
        st.markdown("""
        <div style="background:#1e1e2e;padding:20px;border-radius:10px;border-left:4px solid #ff6b6b;margin-bottom:20px;">
            <h3>Rechtlicher Hinweis & Datenschutz</h3>
            <p><b>Biometrische Gesichtsanalyse ist hochsensibel.</b></p>
            <ul>
                <li>NUR fuer eigene Inhalte oder mit schriftlicher Einwilligung</li>
                <li>Alle Embeddings werden <b>lokal</b> gespeichert (FAISS/SQLite)</li>
                <li>DSGVO Art. 9: Biometrische Daten sind besondere Kategorien</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        c1, c2 = st.columns([1, 3])
        with c1:
            if st.button("Ich akzeptiere", type="primary", use_container_width=True):
                st.session_state.legal_accepted = True
                st.rerun()
        with c2:
            st.caption("Missbrauch biometrischer Daten ist strafbar.")
        st.stop()

def apply_dark_mode():
    if st.session_state.dark_mode:
        st.markdown("""
        <style>
        .stApp { background-color: #0f0f23; color: #e0e0e0; }
        .stTextInput > div > div > input { background-color: #1a1a2e; color: #e0e0e0; }
        .stButton > button { background-color: #2d2d44; color: #e0e0e0; border: 1px solid #4a4a6a; }
        .stCheckbox > label { color: #e0e0e0; }
        .stSlider > div > div > div { color: #e0e0e0; }
        h1, h2, h3, h4 { color: #ffffff; }
        </style>
        """, unsafe_allow_html=True)

def sidebar():
    with st.sidebar:
        st.header("Einstellungen")
        dark = st.toggle("Dark Mode", value=st.session_state.dark_mode)
        if dark != st.session_state.dark_mode:
            st.session_state.dark_mode = dark
            st.rerun()
        st.divider()
        st.markdown("**System-Status**")
        st.write(f"FAISS: {'Ja' if HAS_FAISS else 'Nein'}")
        st.write(f"OpenCV: {'Ja' if HAS_BIOMETRIE else 'Nein'}")
        st.write(f"ONNX: {'Ja' if HAS_ONNX else 'Nein'}")
        st.write(f"Async: Ja (aiohttp)")
        stats = vector_db.get_stats()
        st.write(f"DB-Vektoren: {stats['total_vectors']}")
        st.divider()
        with st.expander("Verlauf"):
            for i, h in enumerate(st.session_state.search_history[:10]):
                st.text(f"{i+1}. {h}")
        with st.expander("Lesezeichen"):
            for bm in st.session_state.bookmarks[:]:
                cols = st.columns([4,1])
                with cols[0]:
                    st.markdown(f"**[{bm['title'][:20]}...]({bm['url']})**")
                with cols[1]:
                    if st.button("X", key=f"del_{_make_key(bm['url'])}"):
                        st.session_state.bookmarks.remove(bm)
                        st.rerun()
        st.divider()
        st.caption("FaceSearch Bio Pro v5.0")

def display_results_table(results):
    if not results:
        st.warning("Keine Treffer.")
        return

    st.subheader(f"Ergebnisse ({len(results)} Treffer)")

    src_counts = {}
    for r in results:
        src_counts[r.platform] = src_counts.get(r.platform, 0) + 1
    colors = ['#FF0000','#E1306C','#000000','#FF4500','#1DA1F2','#1877F2','#BD081C','#0077B5','#34A853']
    fig = go.Figure(go.Bar(x=list(src_counts.keys()), y=list(src_counts.values()), marker_color=colors[:len(src_counts)]))
    fig.update_layout(title="Treffer pro Plattform", height=300,
        template="plotly_dark" if st.session_state.dark_mode else "plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    via_counts = {}
    for r in results:
        via_counts[r.found_via] = via_counts.get(r.found_via, 0) + 1
    fig2 = go.Figure(go.Pie(labels=list(via_counts.keys()), values=list(via_counts.values())))
    fig2.update_layout(title="Gefunden via", height=250)
    st.plotly_chart(fig2, use_container_width=True)

    bio_hits = sum(1 for r in results if r.face_detected)
    high_conf = sum(1 for r in results if r.face_similarity and r.face_similarity >= SIMILARITY_THRESHOLD)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mit Bild", sum(1 for r in results if r.thumbnail_url))
    c2.metric("Gesicht erkannt", bio_hits)
    c3.metric("Bio-Match >=65%", high_conf)
    vals = [r.face_similarity for r in results if r.face_similarity]
    c4.metric("Durchschnitt Bio", f"{np.mean(vals):.1%}" if vals else "n/a")

    st.markdown("---")
    st.markdown("### Treffer-Tabelle mit Biometrie")

    all_platforms = sorted(list(set(r.platform for r in results)))
    selected = st.multiselect("Plattform filtern:", all_platforms, default=all_platforms)
    min_score = st.slider("Min. Text-Score:", 0, 100, 0, key="txt_slider")
    min_bio = st.slider("Min. Bio-Score:", 0.0, 1.0, 0.0, key="bio_slider")

    filtered = [r for r in results if r.platform in selected
                and (r.match_score or 0) >= min_score
                and (r.face_similarity or 0) >= min_bio]

    if not filtered:
        st.info("Keine Treffer mit Filtern.")
        return

    my = sum(1 for r in filtered if st.session_state.match_votes.get(r.url) is True)
    mn = sum(1 for r in filtered if st.session_state.match_votes.get(r.url) is False)
    c1, c2, c3 = st.columns(3)
    c1.metric("Treffer", my); c2.metric("Nein", mn); c3.metric("Offen", len(filtered)-my-mn)
    st.markdown("---")

    for idx, r in enumerate(filtered):
        kb = _make_key(r.url)
        with st.container():
            cols = st.columns([1, 3, 1])

            with cols[0]:
                if r.thumbnail_url:
                    try:
                        st.image(r.thumbnail_url, width=120, use_container_width=True)
                    except:
                        st.markdown("<div style='width:120px;height:90px;background:#333;display:flex;align-items:center;justify-content:center;border-radius:6px;'><span style='font-size:24px;'>Bild</span></div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='width:120px;height:90px;background:#e9ecef;display:flex;align-items:center;justify-content:center;border-radius:6px;'><span style='font-size:24px;'>Suche</span></div>", unsafe_allow_html=True)

                if r.face_detected and r.face_similarity:
                    color = "#4CAF50" if r.face_similarity >= SIMILARITY_THRESHOLD else "#FFC107"
                    st.markdown(f"<div style='background:{color};color:white;padding:2px 6px;border-radius:4px;font-size:11px;text-align:center;margin-top:4px;'>Bio: {r.face_similarity:.0%}</div>", unsafe_allow_html=True)
                elif r.face_detected is False:
                    st.markdown("<div style='background:#666;color:white;padding:2px 6px;border-radius:4px;font-size:11px;text-align:center;margin-top:4px;'>Kein Gesicht</div>", unsafe_allow_html=True)

                via_colors = {"api": "#2196F3", "scraping": "#FF9800", "rss": "#9C27B0", "cache": "#00BCD4", "fallback": "#757575", "manual": "#607D8B"}
                via_color = via_colors.get(r.found_via, "#757575")
                st.markdown(f"<div style='background:{via_color};color:white;padding:1px 4px;border-radius:3px;font-size:9px;text-align:center;margin-top:2px;'>{r.found_via}</div>", unsafe_allow_html=True)

            with cols[1]:
                st.markdown(f"**[{r.title}]({r.url})**")
                st.caption(f"**{r.platform}** | {r.snippet[:140]}")
                if r.match_score:
                    sc = "#4CAF50" if r.match_score >= 70 else ("#FFC107" if r.match_score >= 50 else "#f44336")
                    st.markdown(f"<div style='width:100%;background:#333;height:8px;border-radius:4px;'><div style='width:{min(r.match_score,100)}%;background:{sc};height:8px;border-radius:4px;'></div></div><small>Text-Score: {r.match_score}%</small>", unsafe_allow_html=True)
                st.markdown(f"<a href='{r.url}' target='_blank'><button style='background:#4a90e2;color:white;border:none;padding:4px 10px;border-radius:4px;cursor:pointer;font-size:12px;'>Zum Fundort</button></a>", unsafe_allow_html=True)

            with cols[2]:
                st.markdown("<b>Treffer?</b>", unsafe_allow_html=True)
                vote = st.session_state.match_votes.get(r.url, None)
                v1, v2 = st.columns(2)
                with v1:
                    if st.button("Ja", key=f"yes_{kb}_{idx}", use_container_width=True):
                        st.session_state.match_votes[r.url] = True
                        st.rerun()
                with v2:
                    if st.button("Nein", key=f"no_{kb}_{idx}", use_container_width=True):
                        st.session_state.match_votes[r.url] = False
                        st.rerun()
                if vote is True:
                    st.markdown("<span style='color:#4CAF50;font-weight:bold;font-size:12px;'>Treffer</span>", unsafe_allow_html=True)
                elif vote is False:
                    st.markdown("<span style='color:#f44336;font-weight:bold;font-size:12px;'>Nein</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span style='color:#999;font-size:12px;'>Offen</span>", unsafe_allow_html=True)
                if st.button("Lesezeichen", key=f"bm_{kb}_{idx}"):
                    toggle_bookmark(r)
                    st.rerun()

            st.markdown("---")

    st.subheader("Export")
    export_data = []
    for r in filtered:
        d = r.to_dict()
        d["URL"] = r.url
        d["Thumbnail URL"] = r.thumbnail_url or ""
        export_data.append(d)
    df = pd.DataFrame(export_data)
    csv = df.to_csv(index=False, sep=";").encode("utf-8")
    st.download_button("CSV", csv, f"facesearch_v5_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")

    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        font_loaded = False
        for fp in ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','/System/Library/Fonts/Helvetica.ttc','C:/Windows/Fonts/arial.ttf']:
            if os.path.exists(fp):
                try:
                    pdf.add_font("Custom","",fp,uni=True)
                    pdf.add_font("Custom","B",fp,uni=True)
                    font_loaded = True
                    break
                except:
                    continue
        ff = "Custom" if font_loaded else "Helvetica"
        pdf.set_font(ff,"B",16)
        pdf.cell(0,10,"FaceSearch Bio Pro v5.0 - Bericht",ln=True,align="C")
        pdf.set_font(ff,"",10)
        pdf.cell(0,6,f"Erstellt: {datetime.now().strftime('%d.%m.%Y %H:%M')}",ln=True,align="C")
        pdf.cell(0,6,f"Suchbegriff: {st.session_state.get('last_query_name','n/a')}",ln=True,align="C")
        pdf.ln(5)
        for r in filtered:
            pdf.set_font(ff,"B",11)
            stt = r.title.encode("ascii","ignore").decode() if not font_loaded else r.title
            pdf.cell(0,8,stt[:80],ln=True)
            pdf.set_font(ff,"",9)
            pdf.cell(0,5,f"Plattform: {r.platform} | Text: {r.match_score or '-'}% | Bio: {round(r.face_similarity*100,1) if r.face_similarity else '-'}% | Via: {r.found_via}",ln=True)
            pdf.set_text_color(0,0,255)
            pdf.cell(0,5,r.url,ln=True,link=r.url)
            pdf.set_text_color(0,0,0)
            vote = st.session_state.match_votes.get(r.url,None)
            if vote is not None:
                pdf.set_font(ff,"B",9)
                pdf.cell(0,5,f"Bewertung: {'TREFFER' if vote else 'KEIN TREFFER'}",ln=True)
                pdf.set_font(ff,"",9)
            pdf.ln(3)
        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        st.download_button("PDF", pdf_bytes, f"facesearch_v5_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf", "application/pdf")
    except Exception as e:
        st.error(f"PDF-Fehler: {e}")

def main():
    st.set_page_config(page_title="FaceSearch Bio Pro v5.0", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")
    legal_notice()
    apply_dark_mode()
    sidebar()

    st.title("🌐 FaceSearch Bio Pro v5.0")
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 10px; color: white; margin-bottom: 20px;">
        <b>🔎 Biometrische Gesichtssuche mit Asyncio + FAISS + Circuit-Breaker</b><br>
        <span style="font-size:13px;">⚡ Asyncio | 🧬 FAISS Vektor-DB | 🛡️ Circuit-Breaker | 🔄 Batch-Biometrie | 🔍 Sekundaere Scraping-Wege</span><br>
        <span style="font-size:12px;">YouTube • Instagram • TikTok • Reddit • Twitter/X • Facebook • Pinterest • LinkedIn • News</span>
    </div>
    """, unsafe_allow_html=True)

    if not HAS_BIOMETRIE:
        st.error("""
        Biometrische Module nicht installiert!
        ```bash
        pip install opencv-python
        ```
        Die App laeuft im Fallback-Modus (nur Text-Suche).
        """)
    else:
        st.success("OpenCV Biometrie aktiv - Embeddings werden lokal in FAISS/SQLite gespeichert.")

    col_input, col_results = st.columns([1, 2])

    with col_input:
        st.header("Suchanfrage")

        uploaded = st.file_uploader("Referenzbild hochladen (empfohlen fuer Bio-Match)", type=["jpg","jpeg","png","webp"])
        ref_path = None
        ref_embedding = None

        if uploaded:
            ref_path = save_temp_image(uploaded)
            st.image(uploaded, width=250, caption="Referenzbild")

            if HAS_BIOMETRIE:
                with st.spinner("Extrahiere Gesichts-Embedding..."):
                    face_detected = bio_analyzer.detect_face(ref_path)
                    ref_embedding = bio_analyzer.extract_embedding(ref_path)
                    if ref_embedding is not None:
                        st.session_state.ref_embedding = ref_embedding
                        st.session_state.ref_face_hash = hashlib.md5(uploaded.getvalue()).hexdigest()
                        if face_detected:
                            st.success(f"Gesicht erkannt! Embedding: {len(ref_embedding)} Dimensionen")
                        else:
                            st.warning("Kein Gesicht erkannt, aber Embedding erstellt.")
                    else:
                        st.warning("Embedding konnte nicht erstellt werden.")
            else:
                st.info("OpenCV nicht verfuegbar - Biometrie uebersprungen.")
        else:
            ref_embedding = st.session_state.get("ref_embedding", None)
            if ref_embedding is not None:
                st.info("Gespeichertes Referenz-Embedding wird verwendet.")

        name = st.text_input("Name der Person *", placeholder="z.B. Max Mustermann")

        with st.expander("Plattformen", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                cy = st.checkbox("YouTube", True, key="yt")
                ci = st.checkbox("Instagram", True, key="ig")
                ct = st.checkbox("TikTok", True, key="tt")
                cr = st.checkbox("Reddit", True, key="rd")
                cw = st.checkbox("Twitter/X", True, key="tw")
            with c2:
                cf = st.checkbox("Facebook", True, key="fb")
                cp = st.checkbox("Pinterest", True, key="pt")
                cl = st.checkbox("LinkedIn", False, key="li")
                cn = st.checkbox("News", False, key="nw")
            st.markdown("**Web:**")
            cg = st.checkbox("Google Lens", True, key="gg")
            cb = st.checkbox("Bing Visual", True, key="bi")
            cd = st.checkbox("DuckDuckGo", True, key="dg")
            max_res = st.slider("Max. Ergebnisse", 5, 30, 15, key="mr")

        bio_toggle = st.toggle("Biometrische Analyse aktivieren", value=st.session_state.bio_enabled and HAS_BIOMETRIE, disabled=not HAS_BIOMETRIE)
        st.session_state.bio_enabled = bio_toggle

        if st.button("Suche starten", type="primary", use_container_width=True):
            if not name:
                st.error("Name erforderlich.")
            else:
                add_to_history(name)
                st.session_state["last_query_name"] = name

                engines = []
                if cg: engines.append("google")
                if cb: engines.append("bing")
                if cd: engines.append("duckduckgo")
                if cy: engines.append("youtube")
                if ci: engines.append("instagram")
                if ct: engines.append("tiktok")
                if cr: engines.append("reddit")
                if cw: engines.append("twitter")
                if cf: engines.append("facebook")
                if cp: engines.append("pinterest")
                if cl: engines.append("linkedin")
                if cn: engines.append("news")

                query = SearchQuery(image_path=ref_path, person_name=name, engines=engines, max_results=max_res)

                async def run_search():
                    searcher = AsyncSearcher()
                    emb = ref_embedding if st.session_state.bio_enabled else None
                    results = await searcher.search(query, ref_embedding=emb)
                    await searcher.close()
                    return results

                with st.spinner("Async-Suche laeuft..."):
                    results = asyncio.run(run_search())

                st.session_state.last_results = results
                st.session_state.search_count += 1

                if not results:
                    st.warning("Keine Treffer.")
                else:
                    with col_results:
                        display_results_table(results)

        with st.expander("Lokale Datenbank"):
            st.caption("Verwaltung der gespeicherten Embeddings")
            if st.button("Datenbank leeren", type="secondary"):
                try:
                    os.remove(DB_PATH)
                    vector_db.__init__(DB_PATH, EMBEDDING_DIM)
                    st.success("Datenbank geleert.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Fehler: {e}")
            st.caption(f"Pfad: {DB_PATH}")

    with col_results:
        st.header("Ergebnisse")
        if st.session_state.last_results:
            display_results_table(st.session_state.last_results)
        else:
            st.info("Suchanfrage starten.")

if __name__ == "__main__":
    main()
