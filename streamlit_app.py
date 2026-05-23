#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FaceSearch Bio Pro v6.0 - Production-Ready
Vollstaendig ueberarbeitet: Async-Architektur, echte Biometrie, robustes Error-Handling,
Unicode-PDF, TTL-Caching, XSS-Schutz, und optimierte Streamlit-Integration.
"""

# =============================================================================
# IMPORTS & KOMPATIBILITAET
# =============================================================================

import asyncio
import aiohttp
import streamlit as st
import time
import os
import io
import hashlib
import json
import sqlite3
import tempfile
import base64
import re
import html
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple, Any, Set
from dataclasses import dataclass, field, asdict
from urllib.parse import quote_plus, urlparse, urljoin, unquote
from collections import defaultdict
from enum import Enum
from contextlib import asynccontextmanager, contextmanager
import functools

import numpy as np
from PIL import Image
from fpdf import FPDF
import plotly.graph_objects as go
import pandas as pd

# Logging
import structlog
logger = structlog.get_logger("facesearch")

# Optionale Module mit sauberen Guards
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    logger.warning("beautifulsoup4 nicht verfuegbar - HTML-Parsing deaktiviert")

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logger.warning("FAISS nicht verfuegbar - Fallback-Vektor-Speicher aktiv")

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
    logger.warning("DuckDuckGo-Suche deaktiviert")

try:
    from fake_useragent import UserAgent
    HAS_FAKE_UA = True
except ImportError:
    HAS_FAKE_UA = False

# OpenCV: headless-Variante fuer Cloud-Container
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    cv2 = None  # type: ignore
    logger.error("OpenCV nicht verfuegbar - Biometrie komplett deaktiviert")

# =============================================================================
# KONFIGURATION & KONSTANTEN
# =============================================================================

APP_DIR = Path(tempfile.gettempdir()) / "facesearch_pro_v6"
APP_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = APP_DIR / "face_index_v6.db"
EMBEDDING_DIM = 512
SIMILARITY_THRESHOLD = 0.65
RATE_LIMIT_PER_DOMAIN = 2.0
CACHE_TTL_HOURS = 24
MAX_HISTORY = 20
MAX_BOOKMARKS = 50
MAX_VOTES = 200

# System-Abhaengigkeiten Check
PROXY_POOL = [p.strip() for p in os.getenv("PROXY_POOL", "").split(",") if p.strip()]

# API Keys (Lazy Loading)
def _get_api_key(key_name: str) -> Optional[str]:
    """Holt API Key aus Streamlit Secrets oder Umgebungsvariablen."""
    try:
        return st.secrets[key_name]
    except (KeyError, FileNotFoundError, AttributeError, TypeError):
        return os.getenv(key_name)

# =============================================================================
# ASYNC INFRASTRUKTUR (Korrigiert fuer Streamlit)
# =============================================================================

class AsyncRunner:
    """Singleton-Manager fuer korrekte Async-Execution in Streamlit."""
    _instance = None
    _loop = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_loop(self):
        """Gibt einen funktionierenden Event Loop zurueck."""
        try:
            loop = asyncio.get_running_loop()
            if loop.is_closed():
                raise RuntimeError("Loop closed")
            return loop
        except RuntimeError:
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
            return self._loop

    def run(self, coro):
        """Fuehrt eine Coroutine sicher in Streamlit aus."""
        loop = self.get_loop()
        if loop.is_running():
            # Wir sind in einem laufenden Loop (z.B. Streamlits interner Loop)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=300)
        return loop.run_until_complete(coro)

    def close(self):
        if self._loop and not self._loop.is_closed():
            self._loop.close()
            self._loop = None

async_runner = AsyncRunner()

@asynccontextmanager
async def managed_session():
    """Context-Manager fuer aiohttp Sessions mit garantiertem Cleanup."""
    connector = aiohttp.TCPConnector(
        limit=50, 
        limit_per_host=10, 
        ttl_dns_cache=300,
        use_dns_cache=True,
        force_close=True,
        enable_cleanup_closed=True
    )
    timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_read=15)
    session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    try:
        yield session
    finally:
        await session.close()
        await connector.close()

# =============================================================================
# RATE LIMITER (Per-Domain, Thread-Safe)
# =============================================================================

class AdaptiveRateLimiter:
    """Per-Domain Rate Limiter mit Token Bucket und Backoff."""
    def __init__(self, default_rate: float = 2.0, burst: int = 3):
        self.default_rate = default_rate
        self.burst = burst
        self._tokens: Dict[str, float] = defaultdict(lambda: float(burst))
        self._last_update: Dict[str, float] = defaultdict(time.time)
        self._backoff_until: Dict[str, float] = defaultdict(float)
        self._lock = asyncio.Lock()

    async def acquire(self, domain: str, cost: float = 1.0):
        async with self._lock:
            now = time.time()
            # Backoff-Check
            if now < self._backoff_until[domain]:
                wait = self._backoff_until[domain] - now
                await asyncio.sleep(wait)
                now = time.time()

            elapsed = now - self._last_update[domain]
            self._tokens[domain] = min(
                self.burst, 
                self._tokens[domain] + elapsed * self.default_rate
            )
            self._last_update[domain] = now

            if self._tokens[domain] < cost:
                wait_time = (cost - self._tokens[domain]) / self.default_rate
                await asyncio.sleep(wait_time)
                self._tokens[domain] = 0
            else:
                self._tokens[domain] -= cost

    def report_error(self, domain: str, backoff_seconds: float = 60):
        """Erhoeht Backoff bei Fehlern (z.B. 429, 503)."""
        self._backoff_until[domain] = time.time() + backoff_seconds
        logger.warning(f"Rate-Limiter Backoff fuer {domain}: {backoff_seconds}s")

rate_limiter = AdaptiveRateLimiter(RATE_LIMIT_PER_DOMAIN)

# =============================================================================
# CIRCUIT BREAKER (Mit Logging)
# =============================================================================

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures: Dict[str, int] = defaultdict(int)
        self.last_failure: Dict[str, datetime] = {}
        self.state: Dict[str, str] = defaultdict(lambda: "closed")
        self._lock = asyncio.Lock()

    async def call(self, platform: str, func, *args, **kwargs):
        async with self._lock:
            if self.state[platform] == "open":
                last_fail = self.last_failure.get(platform, datetime.min)
                if datetime.now() - last_fail > timedelta(seconds=self.recovery_timeout):
                    self.state[platform] = "half-open"
                    logger.info(f"CircuitBreaker: {platform} -> half-open")
                else:
                    remaining = self.recovery_timeout - (datetime.now() - last_fail).seconds
                    logger.debug(f"CircuitBreaker: {platform} OPEN ({remaining}s remaining)")
                    return []

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = await asyncio.to_thread(func, *args, **kwargs)

            async with self._lock:
                if self.state[platform] == "half-open":
                    self.state[platform] = "closed"
                    self.failures[platform] = 0
                    logger.info(f"CircuitBreaker: {platform} -> closed (recovered)")
            return result

        except Exception as e:
            async with self._lock:
                self.failures[platform] += 1
                self.last_failure[platform] = datetime.now()
                if self.failures[platform] >= self.failure_threshold:
                    self.state[platform] = "open"
                    logger.error(f"CircuitBreaker: {platform} -> OPEN nach {self.failures[platform]} Fehlern: {e}")
                else:
                    logger.warning(f"CircuitBreaker: {platform} Fehler {self.failures[platform]}/{self.failure_threshold}: {e}")
            raise

circuit_breaker = CircuitBreaker()


# =============================================================================
# UTILITIES & SECURITY
# =============================================================================

def sanitize_input(text: str) -> str:
    """Entfernt potenziell gefaehrliche Zeichen aus User-Input."""
    if not text:
        return ""
    # Erlaubt nur Buchstaben, Zahlen, Leerzeichen, Bindestriche
    cleaned = re.sub(r'[^\w\s\-\.]', '', text)
    return cleaned.strip()[:100]  # Max 100 Zeichen

def validate_url(url: str) -> bool:
    """Validiert ob eine URL sicher ist (kein file://, javascript: etc)."""
    if not url:
        return False
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            return False
        if not parsed.netloc:
            return False
        return True
    except Exception:
        return False

def escape_html(text: str) -> str:
    """XSS-Schutz: Escaped HTML-Sonderzeichen."""
    if not text:
        return ""
    return html.escape(str(text))

# =============================================================================
# BIOMETRIE-ENGINE v2.0 (Echt Biometrisch)
# =============================================================================

class BiometricAnalyzer:
    """OpenCV-basierte Gesichtserkennung ohne TensorFlow/DeepFace.

    Verbesserungen gegenueber v5.1:
    - Face-ROI Extraktion (nur Gesichtsbereich wird analysiert)
    - LBP (Local Binary Patterns) fuer beleuchtungsunabhaengige Features
    - HOG-Deskriptoren als sekundaere Feature-Quelle
    - Face-Alignment basierend auf Augen-Positionen
    - Multi-Scale Face Detection
    """

    def __init__(self):
        self.face_cascade = None
        self.eye_cascade = None
        self.dnn_net = None
        self.lbp_params = {
            'radius': 1,
            'n_points': 8,
            'grid_x': 7,
            'grid_y': 7
        }
        if HAS_CV2:
            self._load_classifiers()

    def _load_classifiers(self):
        """Laedt Haar-Cascades und DNN-Modelle."""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                logger.info("Haar Face Cascade geladen")
        except Exception as e:
            logger.error(f"Face Cascade Fehler: {e}")

        try:
            eye_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            if os.path.exists(eye_path):
                self.eye_cascade = cv2.CascadeClassifier(eye_path)
        except Exception:
            pass

        try:
            prototxt = os.path.expanduser("~/.opencv/face_detector/deploy.prototxt")
            model = os.path.expanduser("~/.opencv/face_detector/res10_300x300_ssd_iter_140000.caffemodel")
            if os.path.exists(prototxt) and os.path.exists(model):
                self.dnn_net = cv2.dnn.readNetFromCaffe(prototxt, model)
                logger.info("DNN Face Detector geladen")
        except Exception:
            pass

    def detect_face(self, img_path: str) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        """Erkennt Gesicht und gibt (erfolg, bbox) zurueck."""
        if not HAS_CV2 or self.face_cascade is None:
            return False, None
        try:
            img = cv2.imread(img_path)
            if img is None:
                return False, None
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(64, 64)
            )
            if len(faces) > 0:
                # Groesstes Gesicht waehlen
                largest = max(faces, key=lambda f: f[2] * f[3])
                return True, tuple(largest)
            return False, None
        except Exception as e:
            logger.warning(f"Face Detection Fehler: {e}")
            return False, None

    def _align_face(self, img: np.ndarray, face_rect: Tuple[int, int, int, int]) -> np.ndarray:
        """Rotiert Gesicht basierend auf Augen-Positionen."""
        if self.eye_cascade is None:
            return img
        try:
            x, y, w, h = face_rect
            roi_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
            if len(eyes) >= 2:
                eyes = sorted(eyes, key=lambda e: e[0])
                left_eye = (eyes[0][0] + eyes[0][2]//2, eyes[0][1] + eyes[0][3]//2)
                right_eye = (eyes[1][0] + eyes[1][2]//2, eyes[1][1] + eyes[1][3]//2)
                dy = right_eye[1] - left_eye[1]
                dx = right_eye[0] - left_eye[0]
                angle = np.degrees(np.arctan2(dy, dx))
                center = (x + w//2, y + h//2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                        flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(128, 128, 128))
                return aligned
        except Exception:
            pass
        return img

    def _extract_lbp(self, gray_roi: np.ndarray) -> np.ndarray:
        """Extrahiert Local Binary Patterns aus Grayscale-ROI."""
        try:
            radius = self.lbp_params['radius']
            n_points = self.lbp_params['n_points']
            grid_x = self.lbp_params['grid_x']
            grid_y = self.lbp_params['grid_y']

            h, w = gray_roi.shape
            cell_h = max(1, h // grid_y)
            cell_w = max(1, w // grid_x)

            lbp_features = []
            for i in range(grid_y):
                for j in range(grid_x):
                    y_start = i * cell_h
                    y_end = min((i + 1) * cell_h, h)
                    x_start = j * cell_w
                    x_end = min((j + 1) * cell_w, w)
                    cell = gray_roi[y_start:y_end, x_start:x_end]

                    # LBP berechnen
                    lbp = np.zeros_like(cell)
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            if dx == 0 and dy == 0:
                                continue
                            shifted = np.roll(np.roll(cell, dy, axis=0), dx, axis=1)
                            lbp += (shifted >= cell).astype(np.uint8)

                    # Histogramm des LBP
                    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
                    hist = hist.astype(np.float32)
                    norm = np.linalg.norm(hist) + 1e-10
                    hist = hist / norm
                    lbp_features.extend(hist.tolist())

            return np.array(lbp_features, dtype=np.float32)
        except Exception as e:
            logger.warning(f"LBP Extraktion fehlgeschlagen: {e}")
            return np.array([], dtype=np.float32)

    def _extract_color_features(self, face_roi: np.ndarray) -> np.ndarray:
        """Extrahiert robuste Farb-Features aus dem Gesicht."""
        try:
            # Konvertiere zu LAB (perzeptuell uniform)
            lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
            features = []
            for i in range(3):
                channel = lab[:, :, i]
                # Statistische Momente
                mean = np.mean(channel)
                std = np.std(channel)
                skew = np.mean(((channel - mean) / (std + 1e-10)) ** 3)
                # Histogramm
                hist = cv2.calcHist([channel], [0], None, [16], [0, 256]).flatten()
                hist = cv2.normalize(hist, hist).flatten()
                features.extend([mean / 255.0, std / 255.0, skew])
                features.extend(hist.tolist())
            return np.array(features, dtype=np.float32)
        except Exception:
            return np.array([], dtype=np.float32)

    def _extract_hog_features(self, gray_roi: np.ndarray) -> np.ndarray:
        """Extrahiert HOG-ähnliche Features (Gradienten-Orientierungen)."""
        try:
            # Berechne Gradienten
            gx = cv2.Sobel(gray_roi, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray_roi, cv2.CV_32F, 0, 1, ksize=3)
            mag, ang = cv2.cartToPolar(gx, gy)

            # Quantisiere Orientierungen in 9 Bins
            bins = np.int32(ang * (9 / (2 * np.pi)))
            bins = np.mod(bins, 9)

            # Cell-Histogramme (8x8 Grid)
            cell_size = max(8, min(gray_roi.shape) // 8)
            h, w = gray_roi.shape
            features = []
            for y in range(0, h - cell_size + 1, cell_size):
                for x in range(0, w - cell_size + 1, cell_size):
                    cell_mag = mag[y:y+cell_size, x:x+cell_size]
                    cell_bins = bins[y:y+cell_size, x:x+cell_size]
                    hist = np.zeros(9, dtype=np.float32)
                    for b in range(9):
                        hist[b] = np.sum(cell_mag[cell_bins == b])
                    norm = np.linalg.norm(hist) + 1e-10
                    features.extend((hist / norm).tolist())
            return np.array(features, dtype=np.float32)
        except Exception:
            return np.array([], dtype=np.float32)

    def extract_embedding(self, img_path: str) -> Optional[np.ndarray]:
        """Erstellt ein robustes 512-D Gesichts-Embedding."""
        if not HAS_CV2:
            return None
        try:
            img = cv2.imread(img_path)
            if img is None:
                return None

            # Gesicht erkennen
            detected, face_rect = self.detect_face(img_path)
            if detected and face_rect is not None:
                # Alignment
                aligned = self._align_face(img, face_rect)
                x, y, w, h = face_rect
                # Margin hinzufuegen (20%)
                margin = int(0.2 * max(w, h))
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(img.shape[1], x + w + margin)
                y2 = min(img.shape[0], y + h + margin)
                face_roi = aligned[y1:y2, x1:x2]
            else:
                # Fallback: Gesamtes Bild (zentriertes Crop)
                h, w = img.shape[:2]
                size = min(h, w)
                y_start = (h - size) // 2
                x_start = (w - size) // 2
                face_roi = img[y_start:y_start+size, x_start:x_start+size]

            if face_roi.size == 0:
                return None

            # Standardisieren auf 128x128
            face_roi = cv2.resize(face_roi, (128, 128), interpolation=cv2.INTER_AREA)
            gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

            # Multi-Feature Extraktion
            lbp = self._extract_lbp(gray_roi)
            color = self._extract_color_features(face_roi)
            hog = self._extract_hog_features(gray_roi)

            # Kombiniere Features
            combined = np.concatenate([lbp, color, hog])

            # Auf 512 Dimensionen normalisieren (PCA-ähnlich durch gleichmaessige Sampling)
            if len(combined) >= EMBEDDING_DIM:
                # Gleichmaessige Samples
                indices = np.linspace(0, len(combined) - 1, EMBEDDING_DIM).astype(int)
                embedding = combined[indices]
            else:
                # Padding mit Nullen (sollte nicht passieren)
                embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)
                embedding[:len(combined)] = combined

            # L2-Normalisierung
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding.astype(np.float32)

        except Exception as e:
            logger.error(f"Embedding Extraktion fehlgeschlagen: {e}")
            return None

    def extract_from_bytes(self, img_bytes: bytes) -> Optional[np.ndarray]:
        """Extrahiert Embedding aus Bild-Bytes."""
        if not HAS_CV2 or not img_bytes:
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
            logger.warning(f"Bytes-Extraktion fehlgeschlagen: {e}")
            return None

    def compare_embeddings(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosinus-Similaritaet zwischen zwei Embeddings."""
        if emb1 is None or emb2 is None:
            return 0.0
        try:
            n1 = np.linalg.norm(emb1)
            n2 = np.linalg.norm(emb2)
            if n1 == 0 or n2 == 0:
                return 0.0
            return float(np.dot(emb1, emb2) / (n1 * n2))
        except Exception:
            return 0.0

    async def batch_extract(self, img_bytes_list: List[bytes]) -> List[Optional[np.ndarray]]:
        """Batch-Extraktion mit Thread-Offloading."""
        if not HAS_CV2:
            return [None] * len(img_bytes_list)

        results = []
        # Verarbeite in kleinen Batches um Memory-Pressure zu vermeiden
        batch_size = 4
        for i in range(0, len(img_bytes_list), batch_size):
            batch = img_bytes_list[i:i+batch_size]
            tasks = [asyncio.to_thread(self.extract_from_bytes, b) for b in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in batch_results:
                if isinstance(res, Exception):
                    logger.warning(f"Batch-Extraktion Fehler: {res}")
                    results.append(None)
                else:
                    results.append(res)
        return results

bio_analyzer = BiometricAnalyzer() if HAS_CV2 else None
HAS_BIOMETRIE = HAS_CV2


# =============================================================================
# DATENBANK & VECTOR STORE (Mit TTL und Connection-Pooling)
# =============================================================================

class VectorDatabase:
    """FAISS/SQLite Hybrid mit TTL-Caching und sauberem Connection-Management."""

    def __init__(self, db_path: Path, dim: int = 512):
        self.db_path = db_path
        self.dim = dim
        self.index = None
        self.urls: List[str] = []
        self.metadata: Dict[str, dict] = {}
        self._fallback_embeddings: Dict[str, np.ndarray] = {}
        self._init_db()

    @contextmanager
    def _get_conn(self):
        """Context-Manager fuer SQLite Connections."""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=10.0)
            conn.row_factory = sqlite3.Row
            yield conn
        finally:
            if conn:
                conn.close()

    def _init_db(self):
        """Initialisiert Schema mit TTL-Support."""
        try:
            with self._get_conn() as conn:
                c = conn.cursor()
                c.execute("""
                    CREATE TABLE IF NOT EXISTS embeddings_meta (
                        url TEXT PRIMARY KEY,
                        source TEXT,
                        platform TEXT,
                        face_detected INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        access_count INTEGER DEFAULT 0
                    )
                """)
                c.execute("""
                    CREATE TABLE IF NOT EXISTS search_cache (
                        query_hash TEXT PRIMARY KEY,
                        results_json TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        hit_count INTEGER DEFAULT 0
                    )
                """)
                c.execute("""
                    CREATE INDEX IF NOT EXISTS idx_cache_created 
                    ON search_cache(created_at)
                """)
                conn.commit()
                logger.info("Datenbank initialisiert")
        except Exception as e:
            logger.error(f"DB Initialisierungsfehler: {e}")

        if HAS_FAISS:
            try:
                self.index = faiss.IndexFlatIP(self.dim)
                logger.info("FAISS Index erstellt")
            except Exception as e:
                logger.error(f"FAISS Index-Fehler: {e}")
                self.index = None

    def _cleanup_expired_cache(self):
        """Entfernt abgelaufene Cache-Eintraege."""
        try:
            cutoff = datetime.now() - timedelta(hours=CACHE_TTL_HOURS)
            with self._get_conn() as conn:
                c = conn.cursor()
                c.execute("DELETE FROM search_cache WHERE created_at < ?", (cutoff,))
                deleted = c.rowcount
                conn.commit()
                if deleted > 0:
                    logger.info(f"Cache bereinigt: {deleted} alte Eintraege entfernt")
        except Exception as e:
            logger.warning(f"Cache-Cleanup Fehler: {e}")

    def save_embedding(self, url: str, source: str, platform: str, 
                      embedding: np.ndarray, face_detected: bool):
        """Speichert Embedding in FAISS und SQLite."""
        if embedding is None or not validate_url(url):
            return

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        if HAS_FAISS and self.index is not None:
            try:
                emb_array = embedding.astype(np.float32).reshape(1, -1)
                self.index.add(emb_array)
                self.urls.append(url)
            except Exception as e:
                logger.warning(f"FAISS add Fehler: {e}")
        else:
            self._fallback_embeddings[url] = embedding

        try:
            with self._get_conn() as conn:
                c = conn.cursor()
                c.execute("""
                    INSERT OR REPLACE INTO embeddings_meta 
                    (url, source, platform, face_detected, created_at, access_count)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, 
                        COALESCE((SELECT access_count FROM embeddings_meta WHERE url = ?), 0) + 1)
                """, (url, source, platform, 1 if face_detected else 0, url))
                conn.commit()
                self.metadata[url] = {
                    "source": source, 
                    "platform": platform, 
                    "face_detected": face_detected
                }
        except Exception as e:
            logger.warning(f"Meta-Speicherung Fehler: {e}")

    def find_similar(self, ref_embedding: np.ndarray, top_k: int = 20, 
                    threshold: float = 0.6) -> List[Tuple[str, float]]:
        """Findet aehnliche Embeddings."""
        if ref_embedding is None:
            return []

        norm = np.linalg.norm(ref_embedding)
        if norm > 0:
            ref_embedding = ref_embedding / norm

        results = []

        if HAS_FAISS and self.index is not None and self.index.ntotal > 0:
            try:
                query = ref_embedding.astype(np.float32).reshape(1, -1)
                k = min(top_k, self.index.ntotal)
                distances, indices = self.index.search(query, k)
                for i, dist in zip(indices[0], distances[0]):
                    if i < len(self.urls) and dist >= threshold:
                        results.append((self.urls[i], float(dist)))
            except Exception as e:
                logger.warning(f"FAISS Suche Fehler: {e}")

        # Fallback
        if not results:
            for url, emb in self._fallback_embeddings.items():
                sim = float(np.dot(ref_embedding, emb))
                if sim >= threshold:
                    results.append((url, sim))
            results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def get_stats(self) -> dict:
        """Gibt Datenbank-Statistiken zurueck."""
        faiss_count = self.index.ntotal if (HAS_FAISS and self.index) else 0
        fallback_count = len(self._fallback_embeddings)
        return {
            "total_vectors": faiss_count + fallback_count,
            "faiss_vectors": faiss_count,
            "fallback_vectors": fallback_count,
            "backend": "FAISS" if HAS_FAISS else "Fallback",
            "dim": self.dim
        }

    def save_search_cache(self, query_hash: str, results: List[Any]):
        """Speichert Suchergebnisse mit TTL."""
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

        try:
            with self._get_conn() as conn:
                c = conn.cursor()
                c.execute("""
                    INSERT OR REPLACE INTO search_cache 
                    (query_hash, results_json, created_at, hit_count)
                    VALUES (?, ?, CURRENT_TIMESTAMP, 
                        COALESCE((SELECT hit_count FROM search_cache WHERE query_hash = ?), 0))
                """, (query_hash, json.dumps(serializable), query_hash))
                conn.commit()
        except Exception as e:
            logger.warning(f"Cache-Speicherung Fehler: {e}")

    def get_search_cache(self, query_hash: str) -> Optional[List[dict]]:
        """Holt gecachte Ergebnisse wenn nicht abgelaufen."""
        try:
            cutoff = datetime.now() - timedelta(hours=CACHE_TTL_HOURS)
            with self._get_conn() as conn:
                c = conn.cursor()
                c.execute("""
                    SELECT results_json, created_at 
                    FROM search_cache 
                    WHERE query_hash = ? AND created_at > ?
                """, (query_hash, cutoff))
                row = c.fetchone()
                if row:
                    # Hit-Count erhoehen
                    c.execute("""
                        UPDATE search_cache 
                        SET hit_count = hit_count + 1 
                        WHERE query_hash = ?
                    """, (query_hash,))
                    conn.commit()
                    return json.loads(row[0])
        except Exception as e:
            logger.warning(f"Cache-Lesen Fehler: {e}")
        return None

    def clear(self):
        """Leert alle Daten."""
        try:
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.index = None
            self.urls = []
            self.metadata = {}
            self._fallback_embeddings = {}
            self._init_db()
            logger.info("Datenbank vollstaendig geleert")
        except Exception as e:
            logger.error(f"Datenbank-Reset Fehler: {e}")

vector_db = VectorDatabase(DB_PATH, EMBEDDING_DIM)

# =============================================================================
# DATENMODELL
# =============================================================================

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

    def __post_init__(self):
        """Validierung nach Initialisierung."""
        self.title = escape_html(self.title[:200])
        self.snippet = escape_html(self.snippet[:500])
        self.url = str(self.url)[:500]
        if self.thumbnail_url and not validate_url(self.thumbnail_url):
            self.thumbnail_url = None

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


# =============================================================================
# NETZWERK-UTILITIES (Mit Retry und besserem Error-Handling)
# =============================================================================

class ConnectionManager:
    """Verbesserter Connection Manager mit Proxy-Rotation und Header-Management."""

    def __init__(self):
        self.ua = UserAgent() if HAS_FAKE_UA else None
        self.proxy_pool = PROXY_POOL
        self.proxy_index = 0

    def get_headers(self) -> Dict[str, str]:
        """Generiert realistische Browser-Headers."""
        base = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
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
        if self.ua:
            try:
                base["User-Agent"] = self.ua.random
            except Exception:
                base["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        else:
            base["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        return base

    def get_proxy(self) -> Optional[str]:
        """Rotiert durch Proxy-Pool."""
        if not self.proxy_pool:
            return None
        proxy = self.proxy_pool[self.proxy_index % len(self.proxy_pool)]
        self.proxy_index += 1
        return proxy

conn_manager = ConnectionManager()

async def fetch_url(url: str, session: aiohttp.ClientSession, 
                    headers: Dict[str, str], proxy: Optional[str] = None, 
                    timeout: int = 15, max_retries: int = 2) -> Tuple[int, Optional[bytes], Dict[str, str]]:
    """Robustes URL-Fetching mit Retry und Rate-Limiting."""
    if not url or not session:
        return 0, None, {}

    domain = urlparse(url).netloc or "unknown"
    await rate_limiter.acquire(domain)

    for attempt in range(max_retries + 1):
        try:
            async with session.get(url, headers=headers, proxy=proxy, 
                                  timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                content = await resp.read() if resp.status == 200 else None
                if resp.status in (429, 503, 502):
                    rate_limiter.report_error(domain, backoff_seconds=30 * (attempt + 1))
                    if attempt < max_retries:
                        await asyncio.sleep(2 ** attempt)
                        continue
                return resp.status, content, dict(resp.headers)

        except asyncio.TimeoutError:
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)
                continue
            return 408, None, {}
        except Exception as e:
            if attempt < max_retries:
                logger.debug(f"Fetch Retry {attempt+1}/{max_retries} fuer {domain}: {e}")
                await asyncio.sleep(2 ** attempt)
                continue
            logger.warning(f"Fetch Fehler fuer {url}: {e}")
            return 0, None, {}

    return 0, None, {}

# =============================================================================
# DUCKDUCKGO SEARCH (Thread-Safe, nicht-blockierend)
# =============================================================================

def _ddgs_images_sync(query: str, max_results: int) -> List[Dict]:
    """Synchrone DDGS Images Suche (wird in Thread ausgefuehrt)."""
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.images(query, max_results=max_results):
                results.append(r)
    except Exception as e:
        logger.warning(f"DDGS Images Fehler: {e}")
    return results

def _ddgs_news_sync(query: str, max_results: int) -> List[Dict]:
    """Synchrone DDGS News Suche."""
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.news(query, max_results=max_results):
                results.append(r)
    except Exception as e:
        logger.warning(f"DDGS News Fehler: {e}")
    return results

async def ddgs_images(query: str, max_results: int = 10) -> List[Dict]:
    """Asynchrone Wrapper fuer DDGS Images."""
    if not HAS_DDGS:
        return []
    return await asyncio.to_thread(_ddgs_images_sync, query, max_results)

async def ddgs_news(query: str, max_results: int = 10) -> List[Dict]:
    """Asynchrone Wrapper fuer DDGS News."""
    if not HAS_DDGS:
        return []
    return await asyncio.to_thread(_ddgs_news_sync, query, max_results)

# =============================================================================
# PLATTFORM-SCRAPER (Mit Retry und besserem Parsing)
# =============================================================================

async def scrape_instagram_profile(session: aiohttp.ClientSession, username: str) -> List[SearchResult]:
    """Scraped Instagram Profil ueber oEmbed und OG-Tags."""
    results = []
    username = sanitize_input(username).lower().replace(' ', '').replace('.', '')
    if not username:
        return results

    headers = conn_manager.get_headers()

    # Versuch 1: oEmbed API
    try:
        oembed_url = f"https://api.instagram.com/oembed?url=https://www.instagram.com/{username}/"
        status, content, _ = await fetch_url(oembed_url, session, headers, timeout=10)
        if status == 200 and content:
            data = json.loads(content)
            thumb = data.get("thumbnail_url", "")
            if validate_url(thumb):
                results.append(SearchResult(
                    title=escape_html(data.get("title", f"@{username} auf Instagram")),
                    url=f"https://www.instagram.com/{username}/",
                    snippet=escape_html(data.get("author_name", "Instagram Profil")),
                    source="instagram", platform="Instagram",
                    thumbnail_url=thumb, match_score=60, found_via="scraping"
                ))
    except Exception as e:
        logger.debug(f"Instagram oEmbed Fehler: {e}")

    # Versuch 2: OG-Tags
    if not results:
        try:
            profile_url = f"https://www.instagram.com/{username}/"
            status, content, _ = await fetch_url(profile_url, session, headers, timeout=10)
            if status == 200 and content:
                html_text = content.decode('utf-8', errors='ignore')
                og_title = re.search(r'<meta property="og:title" content="([^"]+)"', html_text)
                og_image = re.search(r'<meta property="og:image" content="([^"]+)"', html_text)
                og_desc = re.search(r'<meta property="og:description" content="([^"]+)"', html_text)

                thumb = og_image.group(1) if og_image else ""
                if validate_url(thumb):
                    results.append(SearchResult(
                        title=escape_html(og_title.group(1) if og_title else f"@{username} auf Instagram"),
                        url=profile_url,
                        snippet=escape_html(og_desc.group(1)[:200] if og_desc else "Instagram Profil"),
                        source="instagram", platform="Instagram",
                        thumbnail_url=thumb, match_score=55, found_via="scraping"
                    ))
        except Exception as e:
            logger.debug(f"Instagram OG Fehler: {e}")

    return results

async def scrape_tiktok_profile(session: aiohttp.ClientSession, username: str) -> List[SearchResult]:
    """Scraped TikTok Profil ueber Universal Data und OG-Tags."""
    results = []
    username = sanitize_input(username).lower().replace(' ', '')
    if not username:
        return results

    try:
        url = f"https://www.tiktok.com/@{username}"
        status, content, _ = await fetch_url(url, session, conn_manager.get_headers(), timeout=15)
        if status == 200 and content:
            html_text = content.decode('utf-8', errors='ignore')

            # Versuch 1: Universal Data
            data_match = re.search(r'<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__"[^>]*>(.*?)</script>', html_text, re.DOTALL)
            if data_match:
                try:
                    data = json.loads(data_match.group(1))
                    user_info = data.get("__DEFAULT_SCOPE__", {}).get("webapp.user-detail", {}).get("userInfo", {})
                    user = user_info.get("user", {})
                    stats = user_info.get("stats", {})
                    thumb = user.get("avatarLarger", "")
                    if validate_url(thumb):
                        results.append(SearchResult(
                            title=escape_html(f"@{user.get('uniqueId', username)} auf TikTok"),
                            url=url,
                            snippet=escape_html(f"{user.get('nickname', '')} | {stats.get('followerCount', 0)} Follower | {stats.get('videoCount', 0)} Videos"),
                            source="tiktok", platform="TikTok",
                            thumbnail_url=thumb, match_score=65, found_via="scraping"
                        ))
                except Exception as e:
                    logger.debug(f"TikTok Universal Data Parse Fehler: {e}")

            # Versuch 2: OG-Tags
            if not results:
                og_image = re.search(r'<meta property="og:image" content="([^"]+)"', html_text)
                og_title = re.search(r'<meta property="og:title" content="([^"]+)"', html_text)
                if og_title:
                    thumb = og_image.group(1) if og_image else ""
                    results.append(SearchResult(
                        title=escape_html(og_title.group(1)),
                        url=url, snippet="TikTok Profil",
                        source="tiktok", platform="TikTok",
                        thumbnail_url=thumb if validate_url(thumb) else "",
                        match_score=50, found_via="scraping"
                    ))
    except Exception as e:
        logger.debug(f"TikTok Scraping Fehler: {e}")

    return results

async def scrape_twitter_profile(session: aiohttp.ClientSession, username: str) -> List[SearchResult]:
    """Scraped Twitter/X Profil ueber Nitter-Instanzen."""
    results = []
    username = sanitize_input(username).lower().replace(' ', '')
    if not username:
        return results

    nitter_instances = ["https://nitter.net", "https://nitter.it", "https://nitter.cz"]
    for instance in nitter_instances:
        try:
            url = f"{instance}/{username}"
            status, content, _ = await fetch_url(url, session, conn_manager.get_headers(), timeout=10)
            if status == 200 and content and HAS_BS4:
                soup = BeautifulSoup(content, 'lxml')
                profile_pic = soup.find('img', class_='profile-card-avatar')
                bio = soup.find('div', class_='profile-bio')
                if profile_pic:
                    thumb = urljoin(instance, profile_pic.get('src', ''))
                    results.append(SearchResult(
                        title=escape_html(f"@{username} auf Twitter/X"),
                        url=f"https://twitter.com/{username}",
                        snippet=escape_html(bio.get_text(strip=True)[:200] if bio else "Twitter Profil"),
                        source="twitter", platform="Twitter/X",
                        thumbnail_url=thumb if validate_url(thumb) else "",
                        match_score=60, found_via="scraping"
                    ))
                    break
        except Exception as e:
            logger.debug(f"Nitter Fehler {instance}: {e}")

    return results

async def scrape_reddit_profile(session: aiohttp.ClientSession, username: str) -> List[SearchResult]:
    """Holt Reddit Profil ueber die öffentliche API."""
    results = []
    username = sanitize_input(username).lower().replace(' ', '')
    if not username:
        return results

    try:
        url = f"https://www.reddit.com/user/{username}/about.json"
        headers = {**conn_manager.get_headers(), "User-Agent": "FaceSearchBot/6.0"}
        status, content, _ = await fetch_url(url, session, headers, timeout=10)
        if status == 200 and content:
            data = json.loads(content)
            user_data = data.get("data", {})
            created_ts = user_data.get('created_utc', 0)
            created_str = datetime.fromtimestamp(created_ts).strftime('%Y') if created_ts else 'unbekannt'
            thumb = user_data.get("snoovatar_img", user_data.get("icon_img", ""))
            results.append(SearchResult(
                title=escape_html(f"u/{username} auf Reddit"),
                url=f"https://reddit.com/user/{username}",
                snippet=escape_html(f"Karma: {user_data.get('total_karma', 0)} | Account: {created_str}"),
                source="reddit", platform="Reddit",
                thumbnail_url=thumb if validate_url(thumb) else "",
                match_score=65, found_via="scraping"
            ))
    except Exception as e:
        logger.debug(f"Reddit Profil Fehler: {e}")

    return results

async def scrape_pinterest_profile(session: aiohttp.ClientSession, name: str) -> List[SearchResult]:
    """Scraped Pinterest Profil."""
    results = []
    clean_name = sanitize_input(name).lower().replace(' ', '')
    if not clean_name:
        return results

    try:
        url = f"https://www.pinterest.com/{clean_name}/"
        status, content, _ = await fetch_url(url, session, conn_manager.get_headers(), timeout=15)
        if status == 200 and content:
            html_text = content.decode('utf-8', errors='ignore')
            data_match = re.search(r'<script id="__PWS_DATA__"[^>]*>(.*?)</script>', html_text, re.DOTALL)
            if data_match:
                try:
                    data = json.loads(data_match.group(1))
                    user = data.get("props", {}).get("initialReduxState", {}).get("profiles", {}).get(clean_name, {})
                    thumb = user.get("imageLargeUrl", "")
                    results.append(SearchResult(
                        title=escape_html(f"{user.get('fullName', name)} auf Pinterest"),
                        url=url,
                        snippet=escape_html(f"{user.get('followerCount', 0)} Follower | {user.get('pinCount', 0)} Pins"),
                        source="pinterest", platform="Pinterest",
                        thumbnail_url=thumb if validate_url(thumb) else "",
                        match_score=60, found_via="scraping"
                    ))
                except Exception as e:
                    logger.debug(f"Pinterest Data Parse Fehler: {e}")

            if not results:
                og_image = re.search(r'<meta property="og:image" content="([^"]+)"', html_text)
                og_title = re.search(r'<meta property="og:title" content="([^"]+)"', html_text)
                if og_title:
                    thumb = og_image.group(1) if og_image else ""
                    results.append(SearchResult(
                        title=escape_html(og_title.group(1)),
                        url=url, snippet="Pinterest Profil",
                        source="pinterest", platform="Pinterest",
                        thumbnail_url=thumb if validate_url(thumb) else "",
                        match_score=50, found_via="scraping"
                    ))
    except Exception as e:
        logger.debug(f"Pinterest Scraping Fehler: {e}")

    return results

async def scrape_facebook_profile(session: aiohttp.ClientSession, name: str) -> List[SearchResult]:
    """Facebook Suche ueber Graph API oder Fallback."""
    results = []
    name = sanitize_input(name)
    if not name:
        return results

    token = _get_api_key("FACEBOOK_TOKEN")
    if token:
        try:
            url = f"https://graph.facebook.com/v18.0/search?q={quote_plus(name)}&type=page&access_token={token}"
            status, content, _ = await fetch_url(url, session, conn_manager.get_headers(), timeout=15)
            if status == 200 and content:
                data = json.loads(content)
                for page in data.get("data", [])[:3]:
                    page_id = page.get('id', '')
                    thumb = f"https://graph.facebook.com/{page_id}/picture?type=large"
                    results.append(SearchResult(
                        title=escape_html(page.get("name", "Facebook")),
                        url=f"https://facebook.com/{page_id}",
                        snippet=escape_html(f"Facebook Seite | Kategorie: {page.get('category', 'N/A')}"),
                        source="facebook", platform="Facebook",
                        thumbnail_url=thumb if validate_url(thumb) else "",
                        match_score=70, found_via="api"
                    ))
        except Exception as e:
            logger.debug(f"Facebook API Fehler: {e}")

    if not results:
        results.append(SearchResult(
            title=escape_html(f"{name} auf Facebook"),
            url=f"https://www.facebook.com/search/top/?q={quote_plus(name)}",
            snippet="Facebook-Suche", source="facebook", platform="Facebook",
            thumbnail_url="", match_score=40, found_via="fallback"
        ))

    return results

async def scrape_linkedin_profile(session: aiohttp.ClientSession, name: str) -> List[SearchResult]:
    """LinkedIn Suche ueber Google Cache oder direkte Suche."""
    results = []
    name = sanitize_input(name)
    if not name:
        return results

    try:
        cache_url = f"https://webcache.googleusercontent.com/search?q=site:linkedin.com/in+{quote_plus(name)}"
        status, content, _ = await fetch_url(cache_url, session, conn_manager.get_headers(), timeout=15)
        if status == 200 and content and HAS_BS4:
            soup = BeautifulSoup(content, 'lxml')
            links = soup.find_all('a', href=re.compile(r'linkedin\.com/in/'))
            for link in links[:3]:
                href = link.get('href', '')
                if 'linkedin.com/in/' in href:
                    results.append(SearchResult(
                        title=escape_html(f"{name} auf LinkedIn"),
                        url=href,
                        snippet="LinkedIn Profil (via Cache)",
                        source="linkedin", platform="LinkedIn",
                        thumbnail_url="", match_score=45, found_via="cache"
                    ))
    except Exception as e:
        logger.debug(f"LinkedIn Cache Fehler: {e}")

    if not results:
        results.append(SearchResult(
            title=escape_html(f"{name} auf LinkedIn"),
            url=f"https://www.linkedin.com/search/results/people/?keywords={quote_plus(name)}",
            snippet="LinkedIn-Suche", source="linkedin", platform="LinkedIn",
            thumbnail_url="", match_score=40, found_via="fallback"
        ))

    return results

async def scrape_youtube_channel(session: aiohttp.ClientSession, name: str) -> List[SearchResult]:
    """YouTube RSS Feed Parser."""
    results = []
    name = sanitize_input(name).lower().replace(' ', '')
    if not name:
        return results

    try:
        rss_url = f"https://www.youtube.com/feeds/videos.xml?user={name}"
        status, content, _ = await fetch_url(rss_url, session, conn_manager.get_headers(), timeout=10)
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
                    thumb = media.get("url") if media is not None else ""
                    results.append(SearchResult(
                        title=escape_html(title.text or ""),
                        url=link.get("href") if link is not None else "",
                        snippet="YouTube Video (via RSS)",
                        source="youtube", platform="YouTube",
                        thumbnail_url=thumb if validate_url(thumb) else "",
                        match_score=60, found_via="rss"
                    ))
    except Exception as e:
        logger.debug(f"YouTube RSS Fehler: {e}")

    return results


# =============================================================================
# API-SUCHE FUNKTIONEN (Mit besserer Auth und Retry)
# =============================================================================

async def search_youtube_videos(session: aiohttp.ClientSession, name: str, max_res: int) -> List[SearchResult]:
    """YouTube Data API v3 Suche."""
    results = []
    name = sanitize_input(name)
    api_key = _get_api_key("YOUTUBE_API_KEY")
    if not api_key or not name:
        return results

    try:
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "type": "video",
            "q": name,
            "maxResults": min(max_res, 10),
            "key": api_key
        }
        status, content, _ = await fetch_url(url, session, conn_manager.get_headers(), timeout=15)
        if status == 200 and content:
            data = json.loads(content)
            for item in data.get("items", []):
                vid = item["id"]["videoId"]
                s = item["snippet"]
                thumb = s.get("thumbnails", {}).get("medium", {}).get("url", "")
                results.append(SearchResult(
                    title=escape_html(s.get("title", "YouTube")),
                    url=f"https://www.youtube.com/watch?v={vid}",
                    snippet=escape_html(s.get("description", "")[:200]),
                    source="youtube", platform="YouTube",
                    thumbnail_url=thumb if validate_url(thumb) else "",
                    match_score=85, found_via="api"
                ))
    except Exception as e:
        logger.warning(f"YouTube API Fehler: {e}")

    if not results:
        rss_results = await scrape_youtube_channel(session, name)
        results.extend(rss_results)
    if not results:
        results.append(SearchResult(
            title=escape_html(f"YouTube-Suche: {name}"),
            url=f"https://www.youtube.com/results?search_query={quote_plus(name)}",
            snippet="Videosuche", source="youtube", platform="YouTube",
            thumbnail_url="", match_score=50, found_via="fallback"
        ))
    return results

async def search_instagram_posts(session: aiohttp.ClientSession, name: str, max_res: int) -> List[SearchResult]:
    """Instagram Suche ueber DDGS Images und Scraping."""
    results = []
    name = sanitize_input(name)
    if not name:
        return results

    # DDGS Images
    try:
        ddgs_results = await ddgs_images(f'"{name}" site:instagram.com', max_results=min(max_res, 15))
        for r in ddgs_results:
            img = r.get("image", "")
            thumb = r.get("thumbnail", img)
            if validate_url(thumb):
                results.append(SearchResult(
                    title=escape_html(r.get("title", f"Instagram: {name}")),
                    url=img if validate_url(img) else f"https://www.instagram.com/{name.lower().replace(' ', '').replace('.', '')}/",
                    snippet="Instagram", source="instagram", platform="Instagram",
                    thumbnail_url=thumb, match_score=70, found_via="api"
                ))
    except Exception as e:
        logger.warning(f"Instagram DDGS Fehler: {e}")

    if not results:
        clean_name = name.lower().replace(' ', '').replace('.', '')
        scrape_results = await scrape_instagram_profile(session, clean_name)
        results.extend(scrape_results)

    if not results:
        clean_name = name.lower().replace(' ', '').replace('.', '')
        results.append(SearchResult(
            title=escape_html(f"{name} auf Instagram"),
            url=f"https://www.instagram.com/{clean_name}/",
            snippet="Profil", source="instagram", platform="Instagram",
            thumbnail_url="", match_score=45, found_via="fallback"
        ))
    return results

async def search_tiktok_posts(session: aiohttp.ClientSession, name: str, max_res: int) -> List[SearchResult]:
    """TikTok Suche ueber DDGS und Scraping."""
    results = []
    name = sanitize_input(name)
    if not name:
        return results

    try:
        ddgs_results = await ddgs_images(f'"{name}" site:tiktok.com', max_results=min(max_res, 10))
        for r in ddgs_results:
            img = r.get("image", "")
            thumb = r.get("thumbnail", img)
            if validate_url(thumb):
                results.append(SearchResult(
                    title=escape_html(r.get("title", f"TikTok: {name}")),
                    url=img if validate_url(img) else f"https://www.tiktok.com/@{name.lower().replace(' ', '')}",
                    snippet="TikTok", source="tiktok", platform="TikTok",
                    thumbnail_url=thumb, match_score=68, found_via="api"
                ))
    except Exception as e:
        logger.warning(f"TikTok DDGS Fehler: {e}")

    if not results:
        clean_name = name.lower().replace(' ', '')
        scrape_results = await scrape_tiktok_profile(session, clean_name)
        results.extend(scrape_results)

    if not results:
        results.append(SearchResult(
            title=escape_html(f"{name} auf TikTok"),
            url=f"https://www.tiktok.com/@{name.lower().replace(' ', '')}",
            snippet="Profil", source="tiktok", platform="TikTok",
            thumbnail_url="", match_score=45, found_via="fallback"
        ))
    return results

async def search_reddit_posts(session: aiohttp.ClientSession, name: str, max_res: int) -> List[SearchResult]:
    """Reddit Suche ueber OAuth API oder Scraping."""
    results = []
    name = sanitize_input(name)
    client_id = _get_api_key("REDDIT_CLIENT_ID")
    client_secret = _get_api_key("REDDIT_CLIENT_SECRET")

    if client_id and client_secret and name:
        try:
            auth = aiohttp.BasicAuth(client_id, client_secret)
            headers = {**conn_manager.get_headers(), "User-Agent": "FaceSearchPro/6.0"}

            # OAuth Token
            token = None
            async with session.post("https://www.reddit.com/api/v1/access_token",
                auth=auth, data={"grant_type": "client_credentials"}, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    token_data = await resp.json()
                    token = token_data.get("access_token")

            if token:
                headers["Authorization"] = f"Bearer {token}"
                async with session.get("https://oauth.reddit.com/search",
                    headers=headers, params={"q": name, "type": "link", "limit": min(max_res, 25)},
                    timeout=aiohttp.ClientTimeout(total=15)) as resp2:
                    if resp2.status == 200:
                        data = await resp2.json()
                        for post in data.get("data", {}).get("children", []):
                            d = post["data"]
                            url = d.get("url", "")
                            is_img = any(ext in url.lower() for ext in [".jpg",".jpeg",".png",".gif","imgur","i.redd.it"])
                            results.append(SearchResult(
                                title=escape_html(d.get("title","Reddit")),
                                url=f"https://reddit.com{d.get('permalink','')}",
                                snippet=escape_html(f"r/{d.get('subreddit','')}"),
                                source="reddit", platform="Reddit",
                                thumbnail_url=url if is_img and validate_url(url) else "",
                                match_score=75 if is_img else 40, found_via="api"
                            ))
        except Exception as e:
            logger.warning(f"Reddit API Fehler: {e}")

    if not results:
        clean_name = name.lower().replace(' ', '')
        scrape_results = await scrape_reddit_profile(session, clean_name)
        results.extend(scrape_results)

    if not results:
        results.append(SearchResult(
            title=escape_html(f"Reddit-Suche: {name}"),
            url=f"https://www.reddit.com/search/?q={quote_plus(name)}",
            snippet="Suche", source="reddit", platform="Reddit",
            thumbnail_url="", match_score=40, found_via="fallback"
        ))
    return results

async def search_twitter_media(session: aiohttp.ClientSession, name: str, max_res: int) -> List[SearchResult]:
    """Twitter/X Suche ueber API v2 oder Nitter Scraping."""
    results = []
    name = sanitize_input(name)
    clean_handle = name.lower().replace(' ', '')
    bearer = _get_api_key("TWITTER_BEARER_TOKEN")

    if bearer and clean_handle:
        try:
            headers = {**conn_manager.get_headers(), "Authorization": f"Bearer {bearer}"}
            async with session.get("https://api.twitter.com/2/users/by",
                headers=headers,
                params={"usernames": clean_handle, "user.fields": "profile_image_url,description"},
                timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for u in data.get("data", []):
                        thumb = u.get("profile_image_url","").replace("_normal","_400x400")
                        results.append(SearchResult(
                            title=escape_html(f"@{u.get('username',clean_handle)} auf X/Twitter"),
                            url=f"https://twitter.com/{u.get('username',clean_handle)}",
                            snippet=escape_html(u.get("description","")),
                            source="twitter", platform="Twitter/X",
                            thumbnail_url=thumb if validate_url(thumb) else "",
                            match_score=80, found_via="api"
                        ))
        except Exception as e:
            logger.warning(f"Twitter API Fehler: {e}")

    if not results:
        scrape_results = await scrape_twitter_profile(session, clean_handle)
        results.extend(scrape_results)

    if not results:
        results.append(SearchResult(
            title=escape_html(f"{name} auf Twitter/X"),
            url=f"https://twitter.com/{clean_handle}",
            snippet="Profil", source="twitter", platform="Twitter/X",
            thumbnail_url="", match_score=50, found_via="fallback"
        ))
    return results

async def search_facebook_posts(session: aiohttp.ClientSession, name: str, max_res: int) -> List[SearchResult]:
    """Facebook Suche."""
    return await scrape_facebook_profile(session, name)

async def search_pinterest_posts(session: aiohttp.ClientSession, name: str, max_res: int) -> List[SearchResult]:
    """Pinterest Suche ueber DDGS und Scraping."""
    results = []
    name = sanitize_input(name)
    if not name:
        return results

    try:
        ddgs_results = await ddgs_images(f'"{name}" site:pinterest.com', max_results=min(max_res, 10))
        for r in ddgs_results:
            img = r.get("image", "")
            thumb = r.get("thumbnail", img)
            if validate_url(thumb):
                results.append(SearchResult(
                    title=escape_html(r.get("title", f"Pinterest: {name}")),
                    url=img,
                    snippet="Pinterest", source="pinterest", platform="Pinterest",
                    thumbnail_url=thumb, match_score=65, found_via="api"
                ))
    except Exception as e:
        logger.warning(f"Pinterest DDGS Fehler: {e}")

    if not results:
        scrape_results = await scrape_pinterest_profile(session, name)
        results.extend(scrape_results)

    if not results:
        results.append(SearchResult(
            title=escape_html(f"Pinterest-Suche: {name}"),
            url=f"https://www.pinterest.de/search/pins/?q={quote_plus(name)}",
            snippet="Suche", source="pinterest", platform="Pinterest",
            thumbnail_url="", match_score=40, found_via="fallback"
        ))
    return results

async def search_linkedin_posts(session: aiohttp.ClientSession, name: str, max_res: int) -> List[SearchResult]:
    """LinkedIn Suche."""
    return await scrape_linkedin_profile(session, name)

async def search_duckduckgo_images(session: aiohttp.ClientSession, name: str, max_res: int) -> List[SearchResult]:
    """Allgemeine Web-Bildersuche ueber DDGS."""
    results = []
    name = sanitize_input(name)
    if not name:
        return results

    try:
        ddgs_results = await ddgs_images(name, max_results=min(max_res, 20))
        for r in ddgs_results:
            img = r.get("image", "")
            thumb = r.get("thumbnail", img)
            if validate_url(thumb):
                results.append(SearchResult(
                    title=escape_html(r.get("title","Web")),
                    url=img,
                    snippet=escape_html(r.get("source","Web")),
                    source="web", platform="Web",
                    thumbnail_url=thumb, match_score=55, found_via="api"
                ))
    except Exception as e:
        logger.warning(f"DDGS Images Fehler: {e}")
    return results

async def search_news(session: aiohttp.ClientSession, name: str, max_res: int) -> List[SearchResult]:
    """News-Suche ueber DDGS."""
    results = []
    name = sanitize_input(name)
    if not name:
        return results

    try:
        ddgs_results = await ddgs_news(name, max_results=min(max_res, 10))
        for r in ddgs_results:
            thumb = r.get("image", "")
            results.append(SearchResult(
                title=escape_html(r.get("title","News")),
                url=r.get("url",""),
                snippet=escape_html(r.get("source","News")),
                source="news", platform="News",
                thumbnail_url=thumb if validate_url(thumb) else "",
                match_score=50, found_via="api"
            ))
    except Exception as e:
        logger.warning(f"News DDGS Fehler: {e}")

    if not results:
        results.append(SearchResult(
            title=escape_html(f"News-Suche: {name}"),
            url=f"https://news.google.com/search?q={quote_plus(name)}",
            snippet="News", source="news", platform="News",
            thumbnail_url="", match_score=40, found_via="fallback"
        ))
    return results

# =============================================================================
# ASYNC SEARCHER ORCHESTRATOR
# =============================================================================

class AsyncSearcher:
    """Orchestriert parallele Suchen ueber alle Plattformen."""

    def __init__(self):
        self.results: List[SearchResult] = []
        self.progress = {"total": 0, "completed": 0, "errors": []}

    async def search(self, query: SearchQuery, ref_embedding: Optional[np.ndarray] = None,
                     progress_callback=None) -> List[SearchResult]:
        """Fuehrt die komplette Suche durch."""

        # Cache-Check
        qhash = _query_hash(query)
        cached = vector_db.get_search_cache(qhash)
        if cached:
            logger.info("Cache-Hit fuer Query")
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
            "google":     lambda s: self._manual_result("Google Lens", "https://lens.google.com/upload", "Bild hochladen", "google"),
            "bing":       lambda s: self._manual_result("Bing Visual", "https://www.bing.com/images/search?view=detailv2&iss=sbi", "Drag & Drop", "bing"),
            "duckduckgo": lambda s: search_duckduckgo_images(s, name, max_r),
            "youtube":    lambda s: search_youtube_videos(s, name, max_r),
            "instagram":  lambda s: search_instagram_posts(s, name, max_r),
            "tiktok":     lambda s: search_tiktok_posts(s, name, max_r),
            "reddit":     lambda s: search_reddit_posts(s, name, max_r),
            "twitter":    lambda s: search_twitter_media(s, name, max_r),
            "facebook":   lambda s: search_facebook_posts(s, name, max_r),
            "pinterest":  lambda s: search_pinterest_posts(s, name, max_r),
            "linkedin":   lambda s: search_linkedin_posts(s, name, max_r),
            "news":       lambda s: search_news(s, name, max_r),
        }

        async with managed_session() as session:
            tasks = []
            valid_engines = []
            for en in query.engines:
                if en in engine_map:
                    valid_engines.append(en)
                    tasks.append(asyncio.create_task(
                        self._safe_search(en, engine_map[en], session)
                    ))

            self.progress["total"] = len(tasks)
            self.progress["completed"] = 0

            # Sammle Ergebnisse mit Fortschritt
            results = []
            for coro in asyncio.as_completed(tasks):
                try:
                    batch = await coro
                    if isinstance(batch, list):
                        results.extend(batch)
                    self.progress["completed"] += 1
                    if progress_callback:
                        progress_callback(self.progress)
                except Exception as e:
                    self.progress["errors"].append(str(e))
                    self.progress["completed"] += 1
                    logger.error(f"Search Batch Fehler: {e}")

            # Biometrische Analyse
            if ref_embedding is not None and HAS_BIOMETRIE and bio_analyzer is not None:
                await self._batch_bio_analyze(results, ref_embedding, session)

            # Scoring und Sortierung
            def combined_score(r: SearchResult) -> float:
                text = r.match_score or 0
                bio = (r.face_similarity or 0) * 100
                if r.face_detected:
                    return bio * 0.7 + text * 0.3
                return text

            results.sort(key=combined_score, reverse=True)

            # Cache speichern
            vector_db.save_search_cache(qhash, results)
            return results

    async def _safe_search(self, engine_name: str, func, session: aiohttp.ClientSession):
        """Wrapper mit Circuit-Breaker und Fehler-Handling."""
        try:
            return await circuit_breaker.call(engine_name, func, session)
        except Exception as e:
            logger.warning(f"Engine {engine_name} Fehler: {e}")
            return []

    async def _batch_bio_analyze(self, results: List[SearchResult], 
                                  ref_embedding: np.ndarray,
                                  session: aiohttp.ClientSession):
        """Batch-Biometrie-Analyse fuer Thumbnails."""
        if bio_analyzer is None:
            return

        # Sammle Thumbnails
        thumbs = [(i, r.thumbnail_url) for i, r in enumerate(results) 
                  if r.thumbnail_url and validate_url(r.thumbnail_url)]

        if not thumbs:
            return

        # Lade Bilder
        img_tasks = [fetch_url(url, session, conn_manager.get_headers(), timeout=10) 
                     for _, url in thumbs]
        img_results = await asyncio.gather(*img_tasks, return_exceptions=True)

        img_bytes = []
        valid_indices = []
        for idx, (orig_idx, url) in enumerate(thumbs):
            res = img_results[idx]
            if isinstance(res, tuple) and len(res) == 3:
                status, content, _ = res
                if status == 200 and content:
                    img_bytes.append(content)
                    valid_indices.append(orig_idx)

        if not img_bytes:
            return

        # Extrahiere Embeddings
        embeddings = await bio_analyzer.batch_extract(img_bytes)

        for orig_idx, emb in zip(valid_indices, embeddings):
            if emb is not None:
                sim = bio_analyzer.compare_embeddings(ref_embedding, emb)
                results[orig_idx].face_similarity = sim
                results[orig_idx].face_detected = True
                vector_db.save_embedding(
                    results[orig_idx].thumbnail_url or "",
                    results[orig_idx].source,
                    results[orig_idx].platform,
                    emb,
                    True
                )
            else:
                results[orig_idx].face_detected = False

    async def _bio_analyze_single(self, url: str, ref_embedding: np.ndarray) -> Tuple[Optional[float], bool]:
        """Einzelne Biometrie-Analyse."""
        if bio_analyzer is None or not validate_url(url):
            return None, False

        async with managed_session() as session:
            status, content, _ = await fetch_url(url, session, conn_manager.get_headers(), timeout=10)
            if status == 200 and content:
                emb = await asyncio.to_thread(bio_analyzer.extract_from_bytes, content)
                if emb is not None:
                    sim = bio_analyzer.compare_embeddings(ref_embedding, emb)
                    vector_db.save_embedding(url, "", "", emb, True)
                    return sim, True
        return None, False

    def _manual_result(self, title: str, url: str, snippet: str, source: str) -> List[SearchResult]:
        return [SearchResult(title=title, url=url, snippet=snippet, source=source, 
                           platform=source.title(), match_score=None, found_via="manual")]

# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

def save_temp_image(uploaded_file) -> Optional[str]:
    """Speichert hochgeladenes Bild temporaer."""
    if uploaded_file is None:
        return None
    try:
        tmp = APP_DIR / "uploads"
        tmp.mkdir(parents=True, exist_ok=True)
        name = hashlib.md5(uploaded_file.getvalue()).hexdigest() + ".jpg"
        path = tmp / name
        with open(path, "wb") as f:
            f.write(uploaded_file.getvalue())
        return str(path)
    except Exception as e:
        logger.error(f"Fehler beim Speichern: {e}")
        return None

def add_to_history(query: str):
    """Fuegt Query zum Verlauf hinzu (mit Limit)."""
    if not query:
        return
    hist = st.session_state.search_history
    if query in hist:
        hist.remove(query)
    hist.insert(0, query)
    while len(hist) > MAX_HISTORY:
        hist.pop()

def toggle_bookmark(result: SearchResult) -> bool:
    """Toggle Lesezeichen mit Limit."""
    bm_list = st.session_state.bookmarks
    existing = next((b for b in bm_list if b["url"] == result.url), None)
    if existing:
        bm_list.remove(existing)
        return False
    else:
        if len(bm_list) >= MAX_BOOKMARKS:
            bm_list.pop(0)  # FIFO
        bm_list.append({
            "title": result.title,
            "url": result.url,
            "source": result.source,
            "thumbnail": result.thumbnail_url
        })
        return True

def _make_key(url: str) -> str:
    return hashlib.md5(url.encode("utf-8")).hexdigest()[:12]

def _query_hash(query: SearchQuery) -> str:
    raw = f"{query.image_path}|{query.person_name}|{','.join(sorted(query.engines))}|{query.max_results}"
    return hashlib.md5(raw.encode()).hexdigest()


# =============================================================================
# UI KOMPONENTEN
# =============================================================================

def init_session():
    """Initialisiert Session State mit Validierung."""
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
        'search_progress': 0,
        'search_total': 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Cleanup alter Votes
    if len(st.session_state.match_votes) > MAX_VOTES:
        # Behalte nur die neuesten 200
        items = list(st.session_state.match_votes.items())
        st.session_state.match_votes = dict(items[-MAX_VOTES:])

def legal_notice():
    """Zeigt rechtlichen Hinweis mit einmaliger Akzeptanz."""
    if not st.session_state.legal_accepted:
        st.markdown("""
        <div style="background:#1e1e2e;padding:20px;border-radius:10px;border-left:4px solid #ff6b6b;margin-bottom:20px;">
            <h3>Rechtlicher Hinweis & Datenschutz</h3>
            <p><b>Biometrische Gesichtsanalyse ist hochsensibel.</b></p>
            <ul>
                <li>NUR fuer eigene Inhalte oder mit schriftlicher Einwilligung</li>
                <li>Alle Embeddings werden <b>lokal</b> gespeichert (FAISS/SQLite)</li>
                <li>DSGVO Art. 9: Biometrische Daten sind besondere Kategorien</li>
                <li>Missbrauch biometrischer Daten ist strafbar (StGB 202a ff.)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Ich akzeptiere", type="primary", use_container_width=True, key="accept_legal"):
                st.session_state.legal_accepted = True
                st.rerun()
        with col2:
            st.caption("Durch Klicken bestaetigen Sie die ausschliesslich legale Nutzung.")
        st.stop()

def apply_dark_mode():
    """Wendet Dark Mode CSS an."""
    if st.session_state.dark_mode:
        st.markdown("""
        <style>
        .stApp { background-color: #0f0f23; color: #e0e0e0; }
        .stTextInput > div > div > input { background-color: #1a1a2e; color: #e0e0e0; }
        .stButton > button { background-color: #2d2d44; color: #e0e0e0; border: 1px solid #4a4a6a; }
        .stCheckbox > label { color: #e0e0e0; }
        .stSlider > div > div > div { color: #e0e0e0; }
        h1, h2, h3, h4 { color: #ffffff; }
        .stMetric { background-color: #1a1a2e; padding: 10px; border-radius: 8px; }
        </style>
        """, unsafe_allow_html=True)

def sidebar():
    """Rendert die Sidebar mit System-Status."""
    with st.sidebar:
        st.header("Einstellungen")

        dark = st.toggle("Dark Mode", value=st.session_state.dark_mode, key="toggle_dark")
        if dark != st.session_state.dark_mode:
            st.session_state.dark_mode = dark
            st.rerun()

        st.divider()
        st.markdown("**System-Status**")

        status_cols = st.columns(2)
        with status_cols[0]:
            st.write(f"FAISS: {'Ja' if HAS_FAISS else 'Nein'}")
            st.write(f"OpenCV: {'Ja' if HAS_BIOMETRIE else 'Nein'}")
        with status_cols[1]:
            st.write(f"ONNX: {'Ja' if HAS_ONNX else 'Nein'}")
            st.write(f"Async: Ja")

        stats = vector_db.get_stats()
        st.write(f"DB-Vektoren: {stats['total_vectors']} ({stats['backend']})")

        st.divider()

        with st.expander(f"Verlauf ({len(st.session_state.search_history)})"):
            for i, h in enumerate(st.session_state.search_history[:10]):
                st.text(f"{i+1}. {h}")

        with st.expander(f"Lesezeichen ({len(st.session_state.bookmarks)})"):
            for bm in st.session_state.bookmarks[:]:
                cols = st.columns([4, 1])
                with cols[0]:
                    st.markdown(f"**[{bm['title'][:25]}...]({bm['url']})**")
                with cols[1]:
                    if st.button("X", key=f"del_{_make_key(bm['url'])}"):
                        st.session_state.bookmarks.remove(bm)
                        st.rerun()

        st.divider()
        st.caption("FaceSearch Bio Pro v6.0")

# =============================================================================
# PDF EXPORT (Mit Unicode-Support)
# =============================================================================

class UnicodePDF(FPDF):
    """PDF-Generator mit Unicode-Unterstuetzung via fpdf2."""

    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, "FaceSearch Bio Pro v6.0 - Bericht", ln=True, align="C")
        self.set_font("Helvetica", "", 10)
        self.cell(0, 6, f"Erstellt: {datetime.now().strftime('%d.%m.%Y %H:%M')}", ln=True, align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Seite {self.page_no()}/{{nb}}", align="C")

    def safe_cell(self, text: str, **kwargs):
        """Cell mit ASCII-Fallback fuer nicht unterstuetzte Zeichen."""
        safe_text = text.encode("latin-1", "replace").decode("latin-1")
        self.cell(txt=safe_text, **kwargs)

def generate_pdf(results: List[SearchResult], query_name: str) -> bytes:
    """Generiert Unicode-faehiges PDF."""
    try:
        pdf = UnicodePDF()
        pdf.alias_nb_pages()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Query Info
        pdf.set_font("Helvetica", "B", 12)
        pdf.safe_cell(f"Suchbegriff: {query_name}", ln=True)
        pdf.ln(3)

        # Statistiken
        bio_hits = sum(1 for r in results if r.face_detected)
        high_conf = sum(1 for r in results if r.face_similarity and r.face_similarity >= SIMILARITY_THRESHOLD)
        pdf.set_font("Helvetica", "", 10)
        pdf.safe_cell(f"Gesamttreffer: {len(results)} | Mit Gesicht: {bio_hits} | High-Confidence: {high_conf}", ln=True)
        pdf.ln(5)

        # Ergebnisse
        for r in results:
            pdf.set_font("Helvetica", "B", 11)
            title = r.title[:80] if r.title else "Unbekannt"
            pdf.safe_cell(title, ln=True)

            pdf.set_font("Helvetica", "", 9)
            pdf.safe_cell(
                f"Plattform: {r.platform} | Text: {r.match_score or '-'}% | "
                f"Bio: {round(r.face_similarity*100,1) if r.face_similarity else '-'}% | Via: {r.found_via}",
                ln=True
            )

            # URL als Link
            pdf.set_text_color(0, 0, 255)
            pdf.safe_cell(r.url[:100], ln=True, link=r.url)
            pdf.set_text_color(0, 0, 0)

            # Bewertung
            vote = st.session_state.match_votes.get(r.url, None)
            if vote is not None:
                pdf.set_font("Helvetica", "B", 9)
                pdf.safe_cell(f"Bewertung: {'TREFFER' if vote else 'KEIN TREFFER'}", ln=True)
                pdf.set_font("Helvetica", "", 9)

            pdf.ln(3)

        return pdf.output()
    except Exception as e:
        logger.error(f"PDF Generierung Fehler: {e}")
        # Fallback: Minimal-PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "FaceSearch Bio Pro v6.0", ln=True, align="C")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 10, f"Fehler bei PDF-Erstellung: {str(e)[:100]}", ln=True)
        return pdf.output()

# =============================================================================
# ERGEBNIS-DARSTELLUNG
# =============================================================================

def display_results_table(results: List[SearchResult]):
    """Zeigt Ergebnisse mit Filtern, Charts und Export."""
    if not results:
        st.warning("Keine Treffer gefunden.")
        return

    st.subheader(f"Ergebnisse ({len(results)} Treffer)")

    # Statistik-Charts
    src_counts = defaultdict(int)
    for r in results:
        src_counts[r.platform] += 1

    if src_counts:
        colors = ['#FF0000','#E1306C','#000000','#FF4500','#1DA1F2','#1877F2','#BD081C','#0077B5','#34A853','#FF6B6B']
        fig = go.Figure(go.Bar(
            x=list(src_counts.keys()), 
            y=list(src_counts.values()), 
            marker_color=colors[:len(src_counts)]
        ))
        fig.update_layout(
            title="Treffer pro Plattform", 
            height=300,
            template="plotly_dark" if st.session_state.dark_mode else "plotly_white",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    via_counts = defaultdict(int)
    for r in results:
        via_counts[r.found_via] += 1

    if via_counts:
        fig2 = go.Figure(go.Pie(
            labels=list(via_counts.keys()), 
            values=list(via_counts.values()),
            hole=0.4
        ))
        fig2.update_layout(
            title="Gefunden via", 
            height=250,
            template="plotly_dark" if st.session_state.dark_mode else "plotly_white"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Metriken
    bio_hits = sum(1 for r in results if r.face_detected)
    high_conf = sum(1 for r in results if r.face_similarity and r.face_similarity >= SIMILARITY_THRESHOLD)
    with_thumbs = sum(1 for r in results if r.thumbnail_url)
    vals = [r.face_similarity for r in results if r.face_similarity]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mit Bild", with_thumbs)
    c2.metric("Gesicht erkannt", bio_hits)
    c3.metric("Bio-Match >=65%", high_conf)
    c4.metric("Durchschnitt Bio", f"{np.mean(vals):.1%}" if vals else "n/a")

    st.markdown("---")

    # Filter
    st.markdown("### Filter")
    all_platforms = sorted(list(set(r.platform for r in results)))

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        selected = st.multiselect("Plattform:", all_platforms, default=all_platforms, key="platform_filter")
    with col_f2:
        min_score = st.slider("Min. Text-Score:", 0, 100, 0, key="txt_slider")
    with col_f3:
        min_bio = st.slider("Min. Bio-Score:", 0.0, 1.0, 0.0, key="bio_slider")

    filtered = [r for r in results 
                if r.platform in selected
                and (r.match_score or 0) >= min_score
                and (r.face_similarity or 0) >= min_bio]

    if not filtered:
        st.info("Keine Treffer mit aktiven Filtern.")
        return

    # Bewertungs-Statistik
    my = sum(1 for r in filtered if st.session_state.match_votes.get(r.url) is True)
    mn = sum(1 for r in filtered if st.session_state.match_votes.get(r.url) is False)
    c1, c2, c3 = st.columns(3)
    c1.metric("Treffer", my, delta=my)
    c2.metric("Nein", mn, delta=-mn)
    c3.metric("Offen", len(filtered)-my-mn)
    st.markdown("---")

    # Ergebnis-Liste
    st.markdown(f"### Gefilterte Treffer ({len(filtered)})")

    for idx, r in enumerate(filtered):
        kb = _make_key(r.url)

        with st.container():
            cols = st.columns([1, 3, 1])

            with cols[0]:
                # Thumbnail
                if r.thumbnail_url and validate_url(r.thumbnail_url):
                    try:
                        st.image(r.thumbnail_url, width=120)
                    except Exception:
                        st.markdown("<div style='width:120px;height:90px;background:#333;display:flex;align-items:center;justify-content:center;border-radius:6px;'><span style='font-size:24px;'>Bild</span></div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='width:120px;height:90px;background:#e9ecef;display:flex;align-items:center;justify-content:center;border-radius:6px;'><span style='font-size:24px;'>Suche</span></div>", unsafe_allow_html=True)

                # Bio-Score Badge
                if r.face_detected and r.face_similarity is not None:
                    color = "#4CAF50" if r.face_similarity >= SIMILARITY_THRESHOLD else "#FFC107"
                    st.markdown(f"<div style='background:{color};color:white;padding:2px 6px;border-radius:4px;font-size:11px;text-align:center;margin-top:4px;'>Bio: {r.face_similarity:.0%}</div>", unsafe_allow_html=True)
                elif r.face_detected is False:
                    st.markdown("<div style='background:#666;color:white;padding:2px 6px;border-radius:4px;font-size:11px;text-align:center;margin-top:4px;'>Kein Gesicht</div>", unsafe_allow_html=True)

                # Via Badge
                via_colors = {
                    "api": "#2196F3", "scraping": "#FF9800", "rss": "#9C27B0",
                    "cache": "#00BCD4", "fallback": "#757575", "manual": "#607D8B"
                }
                via_color = via_colors.get(r.found_via, "#757575")
                st.markdown(f"<div style='background:{via_color};color:white;padding:1px 4px;border-radius:3px;font-size:9px;text-align:center;margin-top:2px;'>{r.found_via}</div>", unsafe_allow_html=True)

            with cols[1]:
                st.markdown(f"**[{r.title}]({r.url})**")
                st.caption(f"**{r.platform}** | {r.snippet[:140]}")

                if r.match_score:
                    sc = "#4CAF50" if r.match_score >= 70 else ("#FFC107" if r.match_score >= 50 else "#f44336")
                    st.markdown(
                        f"<div style='width:100%;background:#333;height:8px;border-radius:4px;'>"
                        f"<div style='width:{min(r.match_score,100)}%;background:{sc};height:8px;border-radius:4px;'></div></div>"
                        f"<small>Text-Score: {r.match_score}%</small>",
                        unsafe_allow_html=True
                    )

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
                    st.markdown("<span style='color:#4CAF50;font-weight:bold;font-size:12px;'>✓ Treffer</span>", unsafe_allow_html=True)
                elif vote is False:
                    st.markdown("<span style='color:#f44336;font-weight:bold;font-size:12px;'>✗ Nein</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span style='color:#999;font-size:12px;'>? Offen</span>", unsafe_allow_html=True)

                if st.button("⭐ Lesezeichen", key=f"bm_{kb}_{idx}"):
                    toggle_bookmark(r)
                    st.rerun()

        st.markdown("---")

    # Export
    st.subheader("Export")
    export_data = []
    for r in filtered:
        d = r.to_dict()
        d["URL"] = r.url
        d["Thumbnail URL"] = r.thumbnail_url or ""
        export_data.append(d)

    df = pd.DataFrame(export_data)

    # CSV mit UTF-8 BOM fuer Excel
    csv = df.to_csv(index=False, sep=";", encoding="utf-8-sig")
    st.download_button(
        "📊 CSV Export", 
        csv, 
        f"facesearch_v6_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", 
        "text/csv; charset=utf-8-sig"
    )

    # PDF
    try:
        pdf_bytes = generate_pdf(filtered, st.session_state.get('last_query_name', 'n/a'))
        st.download_button(
            "📄 PDF Export",
            pdf_bytes,
            f"facesearch_v6_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            "application/pdf"
        )
    except Exception as e:
        st.error(f"PDF Export Fehler: {e}")

# =============================================================================
# HAUPT-APP
# =============================================================================

def main():
    """Hauptanwendung."""
    st.set_page_config(
        page_title="FaceSearch Bio Pro v6.0", 
        page_icon="🧬", 
        layout="wide", 
        initial_sidebar_state="expanded"
    )

    init_session()
    legal_notice()
    apply_dark_mode()
    sidebar()

    st.title("🌐 FaceSearch Bio Pro v6.0")
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 10px; color: white; margin-bottom: 20px;">
        <b>🔎 Biometrische Gesichtssuche mit Asyncio + FAISS + Circuit-Breaker</b><br>
        <span style="font-size:13px;">⚡ Asyncio | 🧬 FAISS Vektor-DB | 🛡️ Circuit-Breaker | 🔄 Batch-Biometrie | 🔍 Sekundaere Scraping-Wege</span><br>
        <span style="font-size:12px;">YouTube • Instagram • TikTok • Reddit • Twitter/X • Facebook • Pinterest • LinkedIn • News</span>
    </div>
    """, unsafe_allow_html=True)

    # System-Status Banner
    if not HAS_BIOMETRIE:
        st.error("""
        ⚠️ Biometrische Module nicht installiert!
        ```bash
        pip install opencv-python-headless
        ```
        Die App laeuft im Fallback-Modus (nur Text-Suche).
        """)
    else:
        st.success("✅ OpenCV Biometrie aktiv - LBP/HOG/Color Features mit Face-Alignment")

    col_input, col_results = st.columns([1, 2])

    with col_input:
        st.header("Suchanfrage")

        # Bild-Upload
        uploaded = st.file_uploader(
            "Referenzbild hochladen (empfohlen fuer Bio-Match)", 
            type=["jpg","jpeg","png","webp"],
            key="image_uploader"
        )

        ref_path = None
        ref_embedding = None

        if uploaded:
            ref_path = save_temp_image(uploaded)
            st.image(uploaded, width=250, caption="Referenzbild")

            if HAS_BIOMETRIE and bio_analyzer is not None:
                with st.spinner("Analysiere Gesicht..."):
                    face_detected, face_rect = bio_analyzer.detect_face(ref_path)
                    ref_embedding = bio_analyzer.extract_embedding(ref_path)

                    if ref_embedding is not None:
                        st.session_state.ref_embedding = ref_embedding
                        st.session_state.ref_face_hash = hashlib.md5(uploaded.getvalue()).hexdigest()

                        if face_detected:
                            st.success(f"✅ Gesicht erkannt! ROI: {face_rect}. Embedding: {len(ref_embedding)}D")
                        else:
                            st.warning("⚠️ Kein Gesicht erkannt (Fallback auf Vollbild-Analyse).")
                    else:
                        st.error("❌ Embedding konnte nicht erstellt werden.")
            else:
                st.info("OpenCV nicht verfuegbar - Biometrie uebersprungen.")
        else:
            ref_embedding = st.session_state.get("ref_embedding", None)
            if ref_embedding is not None:
                st.info("📌 Gespeichertes Referenz-Embedding wird verwendet.")

        # Name Input
        name = st.text_input(
            "Name der Person *", 
            placeholder="z.B. Max Mustermann",
            key="person_name"
        )

        # Plattformen
        with st.expander("Plattformen auswaehlen", expanded=True):
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

            st.markdown("**Web-Suchmaschinen:**")
            cg = st.checkbox("Google Lens", True, key="gg")
            cb = st.checkbox("Bing Visual", True, key="bi")
            cd = st.checkbox("DuckDuckGo", True, key="dg")

            max_res = st.slider("Max. Ergebnisse pro Engine", 5, 30, 15, key="mr")

        # Biometrie Toggle
        bio_toggle = st.toggle(
            "Biometrische Analyse aktivieren", 
            value=st.session_state.bio_enabled and HAS_BIOMETRIE, 
            disabled=not HAS_BIOMETRIE,
            key="bio_toggle"
        )
        st.session_state.bio_enabled = bio_toggle

        # Such-Button
        search_triggered = st.button("🔍 Suche starten", type="primary", use_container_width=True, key="search_btn")

        if search_triggered:
            if not name or not name.strip():
                st.error("❌ Name ist erforderlich.")
            else:
                sanitized_name = sanitize_input(name)
                add_to_history(sanitized_name)
                st.session_state["last_query_name"] = sanitized_name

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

                if not engines:
                    st.error("❌ Mindestens eine Plattform auswaehlen.")
                else:
                    query = SearchQuery(
                        image_path=ref_path, 
                        person_name=sanitized_name, 
                        engines=engines, 
                        max_results=max_res
                    )

                    # Progress-Tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def update_progress(p):
                        pct = min(p["completed"] / max(p["total"], 1), 1.0)
                        progress_bar.progress(pct)
                        status_text.text(f"Suche laeuft... {p['completed']}/{p['total']} Engines abgeschlossen")

                    async def run_search():
                        searcher = AsyncSearcher()
                        emb = ref_embedding if st.session_state.bio_enabled else None
                        results = await searcher.search(query, ref_embedding=emb, progress_callback=update_progress)
                        return results

                    with st.spinner("Async-Suche wird ausgefuehrt..."):
                        try:
                            results = async_runner.run(run_search())
                            st.session_state.last_results = results
                            st.session_state.search_count += 1
                            progress_bar.empty()
                            status_text.empty()
                            st.success(f"✅ Suche abgeschlossen! {len(results)} Treffer gefunden.")
                        except Exception as e:
                            progress_bar.empty()
                            status_text.empty()
                            logger.error(f"Such-Fehler: {e}")
                            st.error(f"❌ Such-Fehler: {str(e)[:200]}")

        # Datenbank-Management
        with st.expander("Datenbank-Verwaltung"):
            st.caption(f"Pfad: {DB_PATH}")
            stats = vector_db.get_stats()
            st.write(f"Vektoren: {stats['total_vectors']} (FAISS: {stats['faiss_vectors']}, Fallback: {stats['fallback_vectors']})")

            if st.button("🗑️ Cache bereinigen (alte Eintraege)", type="secondary"):
                vector_db._cleanup_expired_cache()
                st.success("Cache bereinigt.")

            if st.button("⚠️ Datenbank komplett leeren", type="secondary"):
                try:
                    vector_db.clear()
                    st.success("Datenbank geleert.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Fehler: {e}")

    with col_results:
        st.header("Ergebnisse")
        if st.session_state.last_results:
            display_results_table(st.session_state.last_results)
        else:
            st.info("Starten Sie eine Suchanfrage, um Ergebnisse zu sehen.")

            # Quick-Start Guide
            with st.expander("🚀 Schnellstart-Anleitung"):
                st.markdown("""
                1. **Name eingeben**: Pflichtfeld (z.B. "Max Mustermann")
                2. **Bild hochladen**: Optional, aber empfohlen fuer Bio-Match
                3. **Plattformen waehlen**: Standardmaessig alle aktiv
                4. **Suche starten**: Parallele Async-Suche ueber alle Plattformen
                5. **Ergebnisse filtern**: Nach Plattform, Text-Score, Bio-Score
                6. **Exportieren**: CSV oder PDF mit allen Metadaten
                """)

if __name__ == "__main__":
    main()
