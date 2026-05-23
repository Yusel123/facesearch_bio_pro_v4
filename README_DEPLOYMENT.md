# FaceSearch Bio Pro v6.0 - Deployment Guide

## Was ist neu in v6.0?

### Architektonische Überarbeitung
- **AsyncRunner**: Korrektes Event-Loop-Management fuer Streamlit (kein nest_asyncio mehr noetig)
- **Managed Sessions**: Context-Manager fuer aiohttp mit garantiertem Cleanup
- **AdaptiveRateLimiter**: Per-Domain Token Bucket mit Backoff-Strategie
- **CircuitBreaker**: Thread-safe mit detailliertem Logging

### Biometrie-Engine v2.0 (Echt biometrisch)
- **Face-ROI Extraktion**: Analysiert nur den Gesichtsbereich, nicht das ganze Bild
- **Face-Alignment**: Rotationskorrektur basierend auf Augen-Positionen
- **LBP Features**: Local Binary Patterns (beleuchtungsunabhaengig)
- **HOG Deskriptoren**: Gradienten-Orientierungen fuer Form-Merkmale
- **Color Features**: LAB-Farbraum mit statistischen Momenten
- **Multi-Feature Fusion**: Kombination aller Kanaele zu robustem 512-D Embedding

### Sicherheit
- **XSS-Schutz**: `escape_html()` fuer alle dynamischen Inhalte
- **URL-Validierung**: Blockt file://, javascript:, etc.
- **Input-Sanitization**: Erlaubt nur alphanumerische Zeichen + Leerzeichen/Bindestriche
- **Session-State Limits**: Verhindert unendliches Wachstum von Votes/History

### Datenbank & Caching
- **TTL-Caching**: Automatische Invalidierung nach 24h
- **Connection Pooling**: SQLite mit Context-Managern
- **Cache-Hit Tracking**: Statistiken fuer Cache-Effizienz
- **Cleanup Jobs**: Automatische Bereinigung alter Eintraege

### UI/UX
- **Progress-Tracking**: Live-Fortschrittsbalken waehrend der Suche
- **Bessere Metriken**: Detaillierte Statistiken pro Plattform
- **Unicode-PDF**: Korrekte Umlaute mit fpdf2
- **CSV mit BOM**: UTF-8-SIG fuer Excel-Kompatibilitaet

## Dateistruktur

```
/
├── streamlit_app.py      # Hauptanwendung (v6.0, ~105KB)
├── requirements.txt      # Python-Abhaengigkeiten
├── packages.txt          # System-Abhaengigkeiten (ESSENTIELL)
└── .streamlit/
    └── config.toml       # Streamlit-Konfiguration
```

## Deployment-Schritte

1. Alle Dateien in GitHub Repository pushen
2. Auf Streamlit Cloud deployen
3. App startet ohne libGL.so.1-Fehler (dank packages.txt)

## Bekannte Einschraenkungen

- YouTube/Twitter/Reddit APIs benoetigen API-Keys (Secrets)
- Biometrie funktioniert nur mit `opencv-python-headless`
- TensorFlow/DeepFace nicht verfuegbar (Python 3.14 Inkompatibilitaet)
- LinkedIn-Scraping ist durch Bot-Schutz eingeschraenkt
