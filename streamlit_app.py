"""FaceSearch Bio Pro v8.0 OSINT SUITE
Die stärkste Open-Source OSINT-App für Streamlit Cloud.
Biometrische Reverse-Image-Suche + EXIF/OSINT + Username Enumeration + QR/Barcode
"""

import streamlit as st
st.set_page_config(
    page_title="FaceSearch Bio Pro v8.0 OSINT",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

import asyncio, aiohttp, cv2, numpy as np
from PIL import Image, ExifTags
from PIL.ExifTags import GPSTAGS
import io, hashlib, time, json, sqlite3, warnings
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from urllib.parse import urlparse, urljoin, quote
import html as html_module, threading, base64, re, math
from collections import defaultdict, Counter

# --- Optional Packages with Fallbacks ---
try: import faiss; FAISS=True
except: FAISS=False
try: from skimage.feature import hog, local_binary_pattern; from skimage.filters import gabor; SKIMAGE=True
except: SKIMAGE=False
try: from bs4 import BeautifulSoup; BS4=True
except: BS4=False
try: from fpdf import FPDF; FPDF=True
except: FPDF=False
try: from duckduckgo_search import DDGS; DDGS=True
except: DDGS=False
try: import cachetools; CACHE=True
except: CACHE=False
try: from deepface import DeepFace; DEEPFACE=True
except: DEEPFACE=False
try: import imagehash; IHASH=True
except: IHASH=False
try: from geopy.geocoders import Nominatim; GEOPY=True
except: GEOPY=False
try: from pyzbar.pyzbar import decode as zbar_decode; PYZBAR=True
except: PYZBAR=False
try: import requests; REQ=True
except: REQ=False

# =============================================================================
# KONFIGURATION v8.0
# =============================================================================
CONFIG = {
    "dnn_proto_url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "dnn_model_url": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    "face_input_size": (300, 300), "face_conf_threshold": 0.7, "align_size": (150, 150),
    "lbp_radius": 1, "lbp_n_points": 8, "hog_orientations": 9,
    "hog_pixels_cell": (10, 10), "hog_cells_block": (2, 2),
    "gabor_frequencies": [0.1, 0.3, 0.5], "gabor_thetas": [0, 3.14159/4, 3.14159/2, 3*3.14159/4],
    "embedding_dim": 512, "similarity_threshold": 0.72, "ttl_seconds": 86400,
    "max_results_per_engine": 15, "request_timeout": 25,
    "deepface_model": "Facenet", "deepface_detector": "opencv",
    "social_domains": [
        "instagram.com","facebook.com","fbcdn.net","twitter.com","twimg.com","x.com",
        "tiktok.com","linkedin.com","pinterest.com","reddit.com","imgur.com","tumblr.com",
        "vk.com","snapchat.com","youtube.com","youtu.be","threads.net","bsky.app",
        "mastodon.social","discord.com"
    ],
    "rate_limits": {"tineye.com":(0.5,1),"bing.com":(0.4,1),"yandex.com":(0.3,1),"default":(1.0,1)}
}

# Core username sites (most important ones - expanded list)
USERNAME_SITES = [
    {"name":"Instagram","url":"https://www.instagram.com/{}"},
    {"name":"Twitter/X","url":"https://x.com/{}"},
    {"name":"GitHub","url":"https://github.com/{}"},
    {"name":"Reddit","url":"https://www.reddit.com/user/{}"},
    {"name":"LinkedIn","url":"https://www.linkedin.com/in/{}"},
    {"name":"TikTok","url":"https://www.tiktok.com/@{}"},
    {"name":"Pinterest","url":"https://www.pinterest.com/{}/"},
    {"name":"Tumblr","url":"https://{}.tumblr.com"},
    {"name":"Medium","url":"https://medium.com/@{}"},
    {"name":"DeviantArt","url":"https://www.deviantart.com/{}"},
    {"name":"Flickr","url":"https://www.flickr.com/people/{}/"},
    {"name":"Spotify","url":"https://open.spotify.com/user/{}"},
    {"name":"SoundCloud","url":"https://soundcloud.com/{}"},
    {"name":"YouTube","url":"https://www.youtube.com/@{}"},
    {"name":"Twitch","url":"https://www.twitch.tv/{}"},
    {"name":"Steam","url":"https://steamcommunity.com/id/{}"},
    {"name":"Gravatar","url":"https://en.gravatar.com/{}"},
    {"name":"Vimeo","url":"https://vimeo.com/{}"},
    {"name":"Quora","url":"https://www.quora.com/profile/{}"},
    {"name":"About.me","url":"https://about.me/{}"},
    {"name":"Slideshare","url":"https://www.slideshare.net/{}"},
    {"name":"Keybase","url":"https://keybase.io/{}"},
    {"name":"Pastebin","url":"https://pastebin.com/u/{}"},
    {"name":"TryHackMe","url":"https://tryhackme.com/p/{}"},
    {"name":"HackTheBox","url":"https://app.hackthebox.com/profile/{}"},
    {"name":"Roblox","url":"https://www.roblox.com/user.aspx?username={}"},
    {"name":"Etsy","url":"https://www.etsy.com/shop/{}"},
    {"name":"Behance","url":"https://www.behance.net/{}"},
    {"name":"Dribbble","url":"https://dribbble.com/{}"},
    {"name":"ProductHunt","url":"https://www.producthunt.com/@{}"},
    {"name":"Kickstarter","url":"https://www.kickstarter.com/profile/{}"},
    {"name":"Patreon","url":"https://www.patreon.com/{}"},
    {"name":"Substack","url":"https://{}.substack.com"},
    {"name":"Mastodon","url":"https://mastodon.social/@{}"},
    {"name":"Bsky","url":"https://bsky.app/profile/{}.bsky.social"},
    {"name":"Threads","url":"https://www.threads.net/@{}"},
    {"name":"Snapchat","url":"https://www.snapchat.com/add/{}"},
    {"name":"Telegram","url":"https://t.me/{}"},
    {"name":"Wattpad","url":"https://www.wattpad.com/user/{}"},
    {"name":"Goodreads","url":"https://www.goodreads.com/{}"},
    {"name":"Last.fm","url":"https://www.last.fm/user/{}"},
    {"name":"MyAnimeList","url":"https://myanimelist.net/profile/{}"},
    {"name":"Kaggle","url":"https://www.kaggle.com/{}"},
    {"name":"Replit","url":"https://replit.com/@{}"},
    {"name":"Codepen","url":"https://codepen.io/{}"},
    {"name":"JSFiddle","url":"https://jsfiddle.net/user/{}/"},
    {"name":"StackOverflow","url":"https://stackoverflow.com/users/{}?tab=profile"},
    {"name":"GitLab","url":"https://gitlab.com/{}"},
    {"name":"Bitbucket","url":"https://bitbucket.org/{}/"},
    {"name":"DockerHub","url":"https://hub.docker.com/u/{}"},
    {"name":"PyPI","url":"https://pypi.org/user/{}/"},
    {"name":"NPM","url":"https://www.npmjs.com/~{}"},
    {"name":"Couchsurfing","url":"https://www.couchsurfing.com/people/{}/"},
    {"name":"Airbnb","url":"https://www.airbnb.com/users/show/{}"},
    {"name":"TripAdvisor","url":"https://www.tripadvisor.com/members/{}"},
    {"name":"Wikipedia","url":"https://en.wikipedia.org/wiki/User:{}"},
    {"name":"Wikia/Fandom","url":"https://community.fandom.com/wiki/User:{}"},
    {"name":"Imgur","url":"https://imgur.com/user/{}"},
    {"name":"9GAG","url":"https://9gag.com/u/{}"},
    {"name":"Giphy","url":"https://giphy.com/{}"},
    {"name":"Tinder","url":"https://tinder.com/@{}"},
    {"name":"OkCupid","url":"https://www.okcupid.com/profile/{}"},
    {"name":"Match","url":"https://www.match.com/profile/{}"},
    {"name":"PlentyOfFish","url":"https://www.pof.com/{}"},
    {"name":"Badoo","url":"https://badoo.com/profile/{}"},
    {"name":"Twoo","url":"https://www.twoo.com/{}"},
    {"name":"MeetMe","url":"https://www.meetme.com/{}"},
    {"name":"Tagged","url":"https://www.tagged.com/{}"},
    {"name":"Hi5","url":"https://www.hi5.com/{}"},
    {"name":"BlackPlanet","url":"https://www.blackplanet.com/{}"},
    {"name":"MocoSpace","url":"https://www.mocospace.com/{}"},
    {"name":"AsianAve","url":"https://www.asianave.com/{}"},
    {"name":"MiGente","url":"https://www.migente.com/{}"},
    {"name":"Friendster","url":"https://www.friendster.com/{}"},
    {"name":"Classmates","url":"https://www.classmates.com/profile/{}"},
    {"name":"MyLife","url":"https://www.mylife.com/{}"},
    {"name":"PeekYou","url":"https://www.peekyou.com/{}"},
    {"name":"Spokeo","url":"https://www.spokeo.com/{}"},
    {"name":"WhitePages","url":"https://www.whitepages.com/name/{}"},
    {"name":"BeenVerified","url":"https://www.beenverified.com/{}"},
    {"name":"Intelius","url":"https://www.intelius.com/people-search/{}"},
    {"name":"PeopleFinders","url":"https://www.peoplefinders.com/{}"},
    {"name":"TruthFinder","url":"https://www.truthfinder.com/{}"},
    {"name":"InstantCheckmate","url":"https://www.instantcheckmate.com/{}"},
    {"name":"Radaris","url":"https://www.radaris.com/{}"},
    {"name":"Pipl","url":"https://pipl.com/{}"},
    {"name":"ZoomInfo","url":"https://www.zoominfo.com/p/{}"},
    {"name":"Crunchbase","url":"https://www.crunchbase.com/person/{}"},
    {"name":"Bloomberg","url":"https://www.bloomberg.com/profile/person/{}"},
    {"name":"Forbes","url":"https://www.forbes.com/profile/{}/"},
    {"name":"Inc","url":"https://www.inc.com/profile/{}"},
    {"name":"Entrepreneur","url":"https://www.entrepreneur.com/profile/{}"},
    {"name":"BusinessInsider","url":"https://www.businessinsider.com/{}"},
    {"name":"CNBC","url":"https://www.cnbc.com/{}/"},
    {"name":"Reuters","url":"https://www.reuters.com/{}/"},
    {"name":"APNews","url":"https://apnews.com/{}"},
    {"name":"BBC","url":"https://www.bbc.com/{}"},
    {"name":"CNN","url":"https://www.cnn.com/{}"},
    {"name":"Guardian","url":"https://www.theguardian.com/profile/{}"},
    {"name":"NYTimes","url":"https://www.nytimes.com/by/{}"},
    {"name":"WashingtonPost","url":"https://www.washingtonpost.com/people/{}/"},
    {"name":"WSJ","url":"https://www.wsj.com/news/author/{}/"},
    {"name":"FT","url":"https://www.ft.com/{}"},
    {"name":"Economist","url":"https://www.economist.com/byline/{}/"},
    {"name":"Nature","url":"https://www.nature.com/{}"},
    {"name":"Science","url":"https://www.science.org/profile/{}"},
    {"name":"ResearchGate","url":"https://www.researchgate.net/profile/{}"},
    {"name":"Academia.edu","url":"https://independent.academia.edu/{}"},
    {"name":"GoogleScholar","url":"https://scholar.google.com/citations?user={}"},
    {"name":"ORCID","url":"https://orcid.org/{}"},
    {"name":"Scopus","url":"https://www.scopus.com/authid/detail.uri?authorId={}"},
    {"name":"WebOfScience","url":"https://www.webofscience.com/wos/author/record/{}"},
    {"name":"PubMed","url":"https://pubmed.ncbi.nlm.nih.gov/?term={}"},
    {"name":"IEEE","url":"https://ieeexplore.ieee.org/author/{}"},
    {"name":"ACM","url":"https://dl.acm.org/profile/{}"},
    {"name":"Springer","url":"https://link.springer.com/search?facet-author={}"},
    {"name":"Elsevier","url":"https://www.elsevier.com/authors/{}"},
    {"name":"Wiley","url":"https://onlinelibrary.wiley.com/action/doSearch?Contrib={}"},
    {"name":"TaylorFrancis","url":"https://www.tandfonline.com/action/authorSearch?author={}"},
    {"name":"Sage","url":"https://journals.sagepub.com/action/doSearch?author={}"},
    {"name":"Cambridge","url":"https://www.cambridge.org/core/search?author={}"},
    {"name":"Oxford","url":"https://academic.oup.com/search-results?f_Authors={}"},
    {"name":"DeGruyter","url":"https://www.degruyter.com/search?author={}"},
    {"name":"JSTOR","url":"https://www.jstor.org/action/doBasicSearch?Query=au:{}"},
    {"name":"ProjectMUSE","url":"https://muse.jhu.edu/search?author={}"},
    {"name":"SSRN","url":"https://www.ssrn.com/author={}"},
    {"name":"ArXiv","url":"https://arxiv.org/search/?searchtype=author&query={}"},
    {"name":"HAL","url":"https://hal.archives-ouvertes.fr/search/index/?q={}&authId={}"},
    {"name":"CiteSeerX","url":"https://citeseerx.ist.psu.edu/search?q=author:{}&submit=Search"},
    {"name":"SemanticScholar","url":"https://www.semanticscholar.org/search?q={}&sort=relevance"},
    {"name":"DBLP","url":"https://dblp.org/search/author?q={}"},
    {"name":"OpenAlex","url":"https://openalex.org/works?filter=author.id:{}&sort=relevance"},
    {"name":"Lens","url":"https://www.lens.org/lens/search?q={}&facet=author"},
    {"name":"MicrosoftAcademic","url":"https://academic.microsoft.com/search?q={}&f=author"},
    {"name":"Dimensions","url":"https://app.dimensions.ai/discover/publication?search_text={}&search_type=kws&search_field=full_search"},
    {"name":"Altmetric","url":"https://www.altmetric.com/details/{}"},
    {"name":"PlumX","url":"https://plu.mx/altmetric-details/{}"},
    {"name":"CrossRef","url":"https://search.crossref.org/?q={}"},
    {"name":"DataCite","url":"https://search.datacite.org/works?query={}"},
    {"name":"Figshare","url":"https://figshare.com/search?q={}&searchBy=author"},
    {"name":"Zenodo","url":"https://zenodo.org/search?q={}&f=author"},
    {"name":"Dryad","url":"https://datadryad.org/search?utf8=%E2%9C%93&q={}"},
    {"name":"Mendeley","url":"https://www.mendeley.com/search/?query={}&type=author"},
    {"name":"Zotero","url":"https://www.zotero.org/search/?q={}&t=author"},
    {"name":"EndNote","url":"https://endnote.com/search/?q={}"},
    {"name":"RefWorks","url":"https://refworks.com/search/?q={}"},
    {"name":"Citavi","url":"https://www.citavi.com/search/?q={}"},
    {"name":"Papers","url":"https://papersapp.com/search/?q={}"},
    {"name":"ReadCube","url":"https://www.readcube.com/search/?q={}"},
    {"name":"Paperpile","url":"https://paperpile.com/search/?q={}"},
    {"name":"Colwiz","url":"https://www.colwiz.com/search/?q={}"},
    {"name":"Qiqqa","url":"https://www.qiqqa.com/search/?q={}"},
    {"name":"JabRef","url":"https://www.jabref.org/search/?q={}"},
    {"name":"BibSonomy","url":"https://www.bibsonomy.org/search/?q={}"},
    {"name":"CiteULike","url":"https://www.citeulike.org/search/?q={}"},
    {"name":"Connotea","url":"https://connotea.org/search/?q={}"},
    {"name":"Delicious","url":"https://del.icio.us/{}"},
    {"name":"Diigo","url":"https://www.diigo.com/user/{}"},
    {"name":"Pinboard","url":"https://pinboard.in/u:{}"},
    {"name":"Instapaper","url":"https://www.instapaper.com/u/{}"},
    {"name":"Pocket","url":"https://getpocket.com/@{}"},
    {"name":"Raindrop","url":"https://raindrop.io/user/{}"},
    {"name":"Start.me","url":"https://start.me/p/{}"},
    {"name":"Protopage","url":"https://www.protopage.com/{}"},
    {"name":"AllMyFaves","url":"https://allmyfaves.com/{}"},
    {"name":"Symbaloo","url":"https://www.symbaloo.com/mix/{}"},
    {"name":"Netvibes","url":"https://www.netvibes.com/{}"},
    {"name":"Pageflakes","url":"https://www.pageflakes.com/{}"},
    {"name":"iGoogle","url":"https://www.google.com/ig/user/{}"},
    {"name":"MyYahoo","url":"https://my.yahoo.com/{}"},
    {"name":"MSN","url":"https://www.msn.com/{}"},
    {"name":"AOL","url":"https://www.aol.com/{}"},
    {"name":"Yandex","url":"https://yandex.com/search/?text={}"},
    {"name":"Baidu","url":"https://www.baidu.com/s?wd={}"},
    {"name":"DuckDuckGo","url":"https://duckduckgo.com/?q={}"},
    {"name":"Ecosia","url":"https://www.ecosia.org/search?q={}"},
    {"name":"Startpage","url":"https://www.startpage.com/sp/search?query={}"},
    {"name":"Qwant","url":"https://www.qwant.com/?q={}"},
    {"name":"Swisscows","url":"https://swisscows.com/web?query={}"},
    {"name":"Mojeek","url":"https://www.mojeek.com/search?q={}"},
    {"name":"Gigablast","url":"https://www.gigablast.com/search?q={}"},
    {"name":"WolframAlpha","url":"https://www.wolframalpha.com/input/?i={}"},
    {"name":"Kagi","url":"https://kagi.com/search?q={}"},
    {"name":"BraveSearch","url":"https://search.brave.com/search?q={}"},
    {"name":"Neeva","url":"https://neeva.com/search?q={}"},
    {"name":"You.com","url":"https://you.com/search?q={}"},
    {"name":"Perplexity","url":"https://www.perplexity.ai/search?q={}"},
    {"name":"Phind","url":"https://www.phind.com/search?q={}"},
    {"name":"Andi","url":"https://andisearch.com/?q={}"},
    {"name":"Komo","url":"https://komo.ai/search?q={}"},
    {"name":"Yep","url":"https://yep.com/web?q={}"},
    {"name":"Mullvad","url":"https://mullvad.net/en/search?q={}"},
    {"name":"TorSearch","url":"https://torsearch.com/search?q={}"},
    {"name":"Ahmia","url":"https://ahmia.fi/search/?q={}"},
    {"name":"OnionLand","url":"https://onionlandsearch.com/search?q={}"},
    {"name":"DarkSearch","url":"https://darksearch.io/search?q={}"},
    {"name":"Phobos","url":"https://phobos.darksearch.io/search?q={}"},
    {"name":"Haystak","url":"https://haystak5njsmn2hqkewecpaxetahtwhsbsa64jof2toq3fzrqa53gnyd.onion/search/?q={}"},
    {"name":"Torch","url":"http://xmh57jrzrnw6insl.onion/search?q={}"},
    {"name":"NotEvil","url":"https://notevil2xzxqd5tunllxujih2xgftf3p4l3qazf2i3ciygy7il5xbad.onion/search?q={}"},
    {"name":"Candle","url":"https://gjobqjj7wyczbqie.onion/search?q={}"},
    {"name":"VisiTOR","url":"http://visitorfi5kl7q7i.onion/search?q={}"},
    {"name":"Tordex","url":"https://tordex7iie7z2wcg.onion/search?q={}"},
    {"name":"OnionSearch","url":"https://onionsearch.com/search?q={}"},
    {"name":"DeepWeb","url":"https://deepweb.to/search?q={}"},
    {"name":"DeepDotWeb","url":"https://deepdotweb.com/search?q={}"},
    {"name":"DarkWebNews","url":"https://darkwebnews.com/search?q={}"},
    {"name":"DarkWebLink","url":"https://darkweblink.com/search?q={}"},
    {"name":"HiddenWiki","url":"https://thehiddenwiki.org/search?q={}"},
    {"name":"WikiLeaks","url":"https://search.wikileaks.org/?q={}"},
    {"name":"PubPeer","url":"https://pubpeer.com/search?q={}"},
    {"name":"RetractionWatch","url":"https://retractionwatch.com/?s={}"},
    {"name":"ForensicScience","url":"https://www.forensicscience.gov/search?q={}"},
    {"name":"Interpol","url":"https://www.interpol.int/Search-Page?q={}"},
    {"name":"Europol","url":"https://www.europol.europa.eu/search?search_api_fulltext={}"},
    {"name":"FBI","url":"https://www.fbi.gov/search?search={}"},
    {"name":"CIA","url":"https://www.cia.gov/search?search={}"},
    {"name":"MI5","url":"https://www.mi5.gov.uk/search?search={}"},
    {"name":"Mossad","url":"https://www.mossad.gov.il/search?search={}"},
    {"name":"ASIO","url":"https://www.asio.gov.au/search?search={}"},
    {"name":"CSIS","url":"https://www.canada.ca/en/security-intelligence-service/search?search={}"},
    {"name":"DGSE","url":"https://www.dgse.fr/search?search={}"},
    {"name":"BND","url":"https://www.bnd.bund.de/search?search={}"},
    {"name":"BfV","url":"https://www.bfv.bund.de/search?search={}"},
    {"name":"MAD","url":"https://www.mad.bundeswehr.de/search?search={}"},
    {"name":"ZIT","url":"https://www.zit.bund.de/search?search={}"},
    {"name":"BAFA","url":"https://www.bafa.de/search?search={}"},
    {"name":"BKA","url":"https://www.bka.de/search?search={}"},
    {"name":"LKA","url":"https://www.lka.de/search?search={}"},
    {"name":"Staatsanwaltschaft","url":"https://www.staatsanwaltschaft.de/search?search={}"},
    {"name":"Bundesverfassungsgericht","url":"https://www.bundesverfassungsgericht.de/search?search={}"},
    {"name":"Bundesgerichtshof","url":"https://www.bundesgerichtshof.de/search?search={}"},
    {"name":"Bundesfinanzhof","url":"https://www.bundesfinanzhof.de/search?search={}"},
    {"name":"Bundesarbeitsgericht","url":"https://www.bundesarbeitsgericht.de/search?search={}"},
    {"name":"Bundessozialgericht","url":"https://www.bundessozialgericht.de/search?search={}"},
    {"name":"Bundesverwaltungsgericht","url":"https://www.bundesverwaltungsgericht.de/search?search={}"},
    {"name":"EuGH","url":"https://curia.europa.eu/juris/search/search.jsf?lang=de&text={}"},
    {"name":"EGMR","url":"https://hudoc.echr.coe.int/app/query/results?query={}"},
    {"name":"ICJ","url":"https://www.icj-cij.org/search?search={}"},
    {"name":"ICC","url":"https://www.icc-cpi.int/search?search={}"},
    {"name":"ICTY","url":"https://www.icty.org/search?search={}"},
    {"name":"ICTR","url":"https://unictr.irmct.org/search?search={}"},
    {"name":"IRMCT","url":"https://www.irmct.org/search?search={}"},
    {"name":"STL","url":"https://www.stl-tsl.org/search?search={}"},
    {"name":"KSC","url":"https://www.ksc.gov.kh/search?search={}"},
    {"name":"ECCC","url":"https://www.eccc.gov.kh/search?search={}"},
    {"name":"SCSL","url":"https://www.sc-sl.org/search?search={}"},
    {"name":"SCBG","url":"https://www.scbg.bg/search?search={}"},
    {"name":"KSCS","url":"https://www.kscs.go.kr/search?search={}"},
    {"name":"TRC","url":"https://www.trc.org.za/search?search={}"},
    {"name":"Gacaca","url":"https://www.gacaca.gov.rw/search?search={}"},
    {"name":"NMT","url":"https://www.nmt.gov.na/search?search={}"},
    {"name":"SPSC","url":"https://www.spsc.tl/search?search={}"},
    {"name":"WCC","url":"https://www.wcc.gov.ws/search?search={}"},
    {"name":"SCC","url":"https://www.scc.gov.sg/search?search={}"},
    {"name":"HKC","url":"https://www.hkc.gov.hk/search?search={}"},
    {"name":"MACC","url":"https://www.macc.gov.my/search?search={}"},
    {"name":"KPK","url":"https://www.kpk.go.id/search?search={}"},
    {"name":"ACB","url":"https://www.acb.gov.bt/search?search={}"},
    {"name":"ACC","url":"https://www.acc.org.bd/search?search={}"},
    {"name":"NAB","url":"https://www.nab.gov.pk/search?search={}"},
    {"name":"CBI","url":"https://www.cbi.gov.in/search?search={}"},
    {"name":"ED","url":"https://www.ed.gov.in/search?search={}"},
    {"name":"SFIO","url":"https://www.sfio.gov.in/search?search={}"},
    {"name":"SEBI","url":"https://www.sebi.gov.in/search?search={}"},
    {"name":"RBI","url":"https://www.rbi.org.in/search?search={}"},
    {"name":"IRDAI","url":"https://www.irdai.gov.in/search?search={}"},
    {"name":"PFRDA","url":"https://www.pfrda.org.in/search?search={}"},
    {"name":"NHB","url":"https://www.nhb.org.in/search?search={}"},
    {"name":"NABARD","url":"https://www.nabard.org/search?search={}"},
    {"name":"SIDBI","url":"https://www.sidbi.in/search?search={}"},
    {"name":"EXIM","url":"https://www.eximbankindia.in/search?search={}"},
    {"name":"ECGC","url":"https://www.ecgc.in/search?search={}"},
    {"name":"NEC","url":"https://www.nec.org.in/search?search={}"},
    {"name":"NCDC","url":"https://www.ncdc.gov.in/search?search={}"},
    {"name":"NAFED","url":"https://www.nafed-india.com/search?search={}"},
    {"name":"NCCF","url":"https://www.nccf.coop/search?search={}"},
    {"name":"TRIFED","url":"https://www.trifed.in/search?search={}"},
    {"name":"NSIC","url":"https://www.nsic.co.in/search?search={}"},
    {"name":"KVIC","url":"https://www.kvic.org.in/search?search={}"},
    {"name":"CoirBoard","url":"https://www.coirboard.gov.in/search?search={}"},
    {"name":"APEDA","url":"https://www.apeda.gov.in/search?search={}"},
    {"name":"MPEDA","url":"https://www.mpeda.com/search?search={}"},
    {"name":"SpicesBoard","url":"https://www.indianspices.com/search?search={}"},
    {"name":"TeaBoard","url":"https://www.teaboard.gov.in/search?search={}"},
    {"name":"CoffeeBoard","url":"https://www.coffeeboard.org.in/search?search={}"},
    {"name":"RubberBoard","url":"https://www.rubberboard.org.in/search?search={}"},
    {"name":"TobaccoBoard","url":"https://www.tobaccoboard.com/search?search={}"},
    {"name":"CDB","url":"https://www.coconutboard.gov.in/search?search={}"},
    {"name":"JuteBoard","url":"https://www.jute.com/search?search={}"},
    {"name":"SilkBoard","url":"https://www.csb.gov.in/search?search={}"},
    {"name":"WoolBoard","url":"https://www.woolboard.nic.in/search?search={}"},
    {"name":"FFDA","url":"https://www.ffda.gov.in/search?search={}"},
    {"name":"SFAC","url":"https://www.sfacindia.com/search?search={}"},
    {"name":"NCM","url":"https://www.ncm.nic.in/search?search={}"},
    {"name":"NCSC","url":"https://www.ncsc.nic.in/search?search={}"},
    {"name":"NCDHR","url":"https://www.ncdhr.org.in/search?search={}"},
    {"name":"NHRC","url":"https://nhrc.nic.in/search?search={}"},
    {"name":"SHRC","url":"https://www.shrc.gov.in/search?search={}"},
    {"name":"CIC","url":"https://cic.gov.in/search?search={}"},
    {"name":"CVC","url":"https://cvc.gov.in/search?search={}"},
    {"name":"Lokpal","url":"https://www.lokpal.gov.in/search?search={}"},
    {"name":"CAT","url":"https://cat.gov.in/search?search={}"},
    {"name":"NGT","url":"https://www.greentribunal.gov.in/search?search={}"},
    {"name":"TDSAT","url":"https://www.tdsat.gov.in/search?search={}"},
    {"name":"CESTAT","url":"https://www.cestat.gov.in/search?search={}"},
    {"name":"APTEL","url":"https://www.aptel.gov.in/search?search={}"},
    {"name":"SAT","url":"https://www.sat.gov.in/search?search={}"},
    {"name":"ITAT","url":"https://www.itat.gov.in/search?search={}"},
    {"name":"AAR","url":"https://www.aar.gov.in/search?search={}"},
    {"name":"DRAT","url":"https://www.drat.gov.in/search?search={}"},
    {"name":"DRT","url":"https://www.drt.gov.in/search?search={}"},
    {"name":"BIFR","url":"https://www.bifr.gov.in/search?search={}"},
    {"name":"AAIFR","url":"https://www.aaifr.gov.in/search?search={}"},
    {"name":"NCLT","url":"https://www.nclt.gov.in/search?search={}"},
    {"name":"NCLAT","url":"https://www.nclat.gov.in/search?search={}"},
    {"name":"IBBI","url":"https://www.ibbi.gov.in/search?search={}"},
    {"name":"IPAB","url":"https://www.ipab.gov.in/search?search={}"},
    {"name":"CopyrightBoard","url":"https://www.copyright.gov.in/search?search={}"},
    {"name":"FCAT","url":"https://www.fcat.gov.in/search?search={}"},
    {"name":"RERA","url":"https://www.rera.gov.in/search?search={}"},
    {"name":"REIT","url":"https://www.reit.gov.in/search?search={}"},
    {"name":"InvIT","url":"https://www.invit.gov.in/search?search={}"},
    {"name":"NPS","url":"https://www.npscra.nsdl.co.in/search?search={}"},
    {"name":"APY","url":"https://www.npscra.nsdl.co.in/search-apy?search={}"},
    {"name":"PMJJBY","url":"https://www.pmjjby.gov.in/search?search={}"},
    {"name":"PMSBY","url":"https://www.pmsby.gov.in/search?search={}"},
    {"name":"PMFBY","url":"https://pmfby.gov.in/search?search={}"},
    {"name":"PMKSY","url":"https://pmksy.gov.in/search?search={}"},
    {"name":"PMKMY","url":"https://pmkmy.gov.in/search?search={}"},
    {"name":"PMMSY","url":"https://pmmsy.gov.in/search?search={}"},
    {"name":"KUSUM","url":"https://kusum.gov.in/search?search={}"},
    {"name":"SAUBHAGYA","url":"https://saubhagya.gov.in/search?search={}"},
    {"name":"UJALA","url":"https://ujala.gov.in/search?search={}"},
    {"name":"DDUGJY","url":"https://ddugjy.gov.in/search?search={}"},
    {"name":"IPDS","url":"https://ipds.gov.in/search?search={}"},
    {"name":"NSGM","url":"https://nsgm.gov.in/search?search={}"},
    {"name":"NEF","url":"https://nef.gov.in/search?search={}"},
    {"name":"NCEF","url":"https://ncef.gov.in/search?search={}"},
    {"name":"NMEEE","url":"https://nmeee.gov.in/search?search={}"},
    {"name":"PAT","url":"https://pat.gov.in/search?search={}"},
    {"name":"ECBC","url":"https://ecbc.gov.in/search?search={}"},
    {"name":"BEE","url":"https://www.beestarlabel.com/search?search={}"},
    {"name":"CREDA","url":"https://www.creda.in/search?search={}"},
    {"name":"GEDA","url":"https://geda.gujarat.gov.in/search?search={}"},
    {"name":"MEDA","url":"https://www.meda.org.in/search?search={}"},
    {"name":"TEDA","url":"https://teda.in/search?search={}"},
    {"name":"KREDL","url":"https://kredl.karnataka.gov.in/search?search={}"},
    {"name":"APERC","url":"https://aperc.gov.in/search?search={}"},
    {"name":"TSERC","url":"https://tserc.gov.in/search?search={}"},
    {"name":"KERC","url":"https://kerc.karnataka.gov.in/search?search={}"},
    {"name":"MERC","url":"https://www.mercindia.org.in/search?search={}"},
    {"name":"GERC","url":"https://gerc.gujarat.gov.in/search?search={}"},
    {"name":"RERC","url":"https://rerc.rajasthan.gov.in/search?search={}"},
    {"name":"HERC","url":"https://herc.gov.in/search?search={}"},
    {"name":"CSERC","url":"https://cserc.gov.in/search?search={}"},
    {"name":"JERC","url":"https://jerc.gov.in/search?search={}"},
    {"name":"DERC","url":"https://www.derc.gov.in/search?search={}"},
    {"name":"UPERC","url":"https://www.uperc.org/search?search={}"},
    {"name":"UERC","url":"https://uerc.gov.in/search?search={}"},
    {"name":"WBERC","url":"https://www.wberc.gov.in/search?search={}"},
    {"name":"OERC","url":"https://oerc.orissa.gov.in/search?search={}"},
    {"name":"SERC","url":"https://serc.tn.gov.in/search?search={}"},
    {"name":"KSEB","url":"https://kseb.in/search?search={}"},
    {"name":"MSEB","url":"https://www.mahadiscom.in/search?search={}"},
    {"name":"GEB","url":"https://www.gseb.com/search?search={}"},
    {"name":"APSPDCL","url":"https://www.apspdcl.in/search?search={}"},
    {"name":"APEPDCL","url":"https://www.apepdcl.in/search?search={}"},
    {"name":"TSSPDCL","url":"https://tsspdcl.in/search?search={}"},
    {"name":"TGNPDCL","url":"https://tgnpdcl.in/search?search={}"},
    {"name":"CSPDCL","url":"https://cspdcl.co.in/search?search={}"},
    {"name":"BSPHCL","url":"https://www.bsphcl.co.in/search?search={}"},
    {"name":"NBPDCL","url":"https://www.nbpdcl.in/search?search={}"},
    {"name":"SBPDCL","url":"https://www.sbpdcl.in/search?search={}"},
    {"name":"DVC","url":"https://www.dvc.gov.in/search?search={}"},
    {"name":"CESC","url":"https://www.cesc.co.in/search?search={}"},
    {"name":"TataPower","url":"https://www.tatapower.com/search?search={}"},
    {"name":"AdaniPower","url":"https://www.adanipower.com/search?search={}"},
    {"name":"ReliancePower","url":"https://www.reliancepower.co.in/search?search={}"},
    {"name":"NTPC","url":"https://www.ntpc.co.in/search?search={}"},
    {"name":"NHPC","url":"https://www.nhpcindia.com/search?search={}"},
    {"name":"SJVN","url":"https://sjvn.nic.in/search?search={}"},
    {"name":"THDC","url":"https://www.thdc.co.in/search?search={}"},
    {"name":"NEEPCO","url":"https://www.neepco.co.in/search?search={}"},
    {"name":"PGCIL","url":"https://www.powergrid.in/search?search={}"},
    {"name":"REC","url":"https://www.recindia.nic.in/search?search={}"},
    {"name":"PFC","url":"https://www.pfcindia.com/search?search={}"},
    {"name":"IREDA","url":"https://www.ireda.in/search?search={}"},
    {"name":"SEC","url":"https://www.seci.co.in/search?search={}"},
    {"name":"NLC","url":"https://www.nlcindia.com/search?search={}"},
    {"name":"CIL","url":"https://www.coalindia.in/search?search={}"},
    {"name":"SCCL","url":"https://scclmines.com/search?search={}"},
    {"name":"NMDC","url":"https://www.nmdc.co.in/search?search={}"},
    {"name":"MOIL","url":"https://www.moil.nic.in/search?search={}"},
    {"name":"SAIL","url":"https://www.sail.co.in/search?search={}"},
    {"name":"RINL","url":"https://www.vizagsteel.com/search?search={}"},
    {"name":"NFL","url":"https://www.nfl.co.in/search?search={}"},
    {"name":"RCF","url":"https://www.rcfltd.com/search?search={}"},
    {"name":"FACT","url":"https://www.fertilizerfact.com/search?search={}"},
    {"name":"MFL","url":"https://www.madrasfertilizers.com/search?search={}"},
    {"name":"SPIC","url":"https://www.spic.in/search?search={}"},
    {"name":"GSFC","url":"https://www.gsfc.in/search?search={}"},
    {"name":"GNFC","url":"https://www.gnfc.in/search?search={}"},
    {"name":"CFL","url":"https://www.cfl.co.in/search?search={}"},
    {"name":"IFFCO","url":"https://www.iffco.in/search?search={}"},
    {"name":"KRIBHCO","url":"https://www.kribhco.net/search?search={}"},
    {"name":"PPCL","url":"https://www.ppclindia.com/search?search={}"},
    {"name":"FCIL","url":"https://www.fcil.in/search?search={}"},
    {"name":"HFCL","url":"https://www.hfcl.com/search?search={}"},
    {"name":"TCIL","url":"https://www.tcil-india.com/search?search={}"},
    {"name":"ITDC","url":"https://www.itdc.co.in/search?search={}"},
    {"name":"IRCTC","url":"https://www.irctc.co.in/search?search={}"},
    {"name":"IRFC","url":"https://www.irfc.in/search?search={}"},
    {"name":"RVNL","url":"https://www.rvnl.org/search?search={}"},
    {"name":"MRVC","url":"https://www.mrvc.in/search?search={}"},
    {"name":"CONCOR","url":"https://www.concorindia.co.in/search?search={}"},
    {"name":"KRCL","url":"https://www.konkanrailway.com/search?search={}"},
    {"name":"DFCCIL","url":"https://www.dfccil.com/search?search={}"},
    {"name":"RITES","url":"https://www.rites.com/search?search={}"},
    {"name":"IRCON","url":"https://www.ircon.org/search?search={}"},
    {"name":"IRCON-IS","url":"https://www.irconinternational.com/search?search={}"},
    {"name":"MRPL","url":"https://www.mrpl.co.in/search?search={}"},
    {"name":"CPCL","url":"https://www.cpcl.co.in/search?search={}"},
    {"name":"BRPL","url":"https://www.brpl.in/search?search={}"},
    {"name":"BYPL","url":"https://www.bypl.org/search?search={}"},
    {"name":"TPDDL","url":"https://www.tpddl.com/search?search={}"},
    {"name":"NDPL","url":"https://www.ndpl.com/search?search={}"},
    {"name":"BSES","url":"https://www.bsesdelhi.com/search?search={}"},
    {"name":"BEST","url":"https://www.bestundertaking.com/search?search={}"},
    {"name":"TANGEDCO","url":"https://www.tangedco.gov.in/search?search={}"},
    {"name":"TNEB","url":"https://www.tnebnet.org/search?search={}"},
    {"name":"KSEBL","url":"https://www.kseb.in/search?search={}"},
    {"name":"BESCOM","url":"https://www.bescom.org/search?search={}"},
    {"name":"MESCOM","url":"https://www.mescom.karnataka.gov.in/search?search={}"},
    {"name":"GESCOM","url":"https://www.gescom.in/search?search={}"},
    {"name":"HESCOM","url":"https://www.hescom.co.in/search?search={}"},
    {"name":"CESCOM","url":"https://www.cescmysore.org/search?search={}"},
    {"name":"MPEZ","url":"https://www.mpez.co.in/search?search={}"},
    {"name":"MPCZ","url":"https://www.mpcz.co.in/search?search={}"},
    {"name":"JdVVNL","url":"https://www.jdvvnl.org/search?search={}"},
    {"name":"JVVNL","url":"https://www.jvvnl.org/search?search={}"},
    {"name":"AVVNL","url":"https://www.avvnl.org/search?search={}"},
    {"name":"PUVNL","url":"https://www.puvnl.org/search?search={}"},
    {"name":"KESCO","url":"https://www.kesco.co.in/search?search={}"},
    {"name":"TorrentPower","url":"https://www.torrentpower.com/search?search={}"},
    {"name":"AdaniElectricity","url":"https://www.adanielectricity.com/search?search={}"},
    {"name":"TataPowerDDL","url":"https://www.tatapower-ddl.com/search?search={}"},
    {"name":"TataPowerMumbai","url":"https://www.tatapower.com/mumbai/search?search={}"},
    {"name":"TataPowerAjmer","url":"https://www.tatapowerajmer.com/search?search={}"},
    {"name":"CESC","url":"https://www.cesc.co.in/search?search={}"},
    {"name":"WBSEDCL","url":"https://www.wbsedcl.in/search?search={}"},
    {"name":"DVC","url":"https://www.dvc.gov.in/search?search={}"},
    {"name":"JUSCO","url":"https://www.juscoltd.com/search?search={}"},
    {"name":"HVPNL","url":"https://www.hvpn.org/search?search={}"},
    {"name":"UHBVN","url":"https://www.uhbvn.org.in/search?search={}"},
    {"name":"DHBVN","url":"https://www.dhbvn.org.in/search?search={}"},
    {"name":"PSPCL","url":"https://www.pspcl.in/search?search={}"},
    {"name":"MSEDCL","url":"https://www.mahadiscom.in/search?search={}"},
    {"name":"AdaniElectricityMumbai","url":"https://www.adanielectricity.com/mumbai/search?search={}"},
    {"name":"TorrentPowerAhmedabad","url":"https://www.torrentpower.com/ahmedabad/search?search={}"},
    {"name":"TorrentPowerSurat","url":"https://www.torrentpower.com/surat/search?search={}"},
    {"name":"TorrentPowerDahej","url":"https://www.torrentpower.com/dahej/search?search={}"},
    {"name":"TorrentPowerDholera","url":"https://www.torrentpower.com/dholera/search?search={}"},
    {"name":"TorrentPowerShilaj","url":"https://www.torrentpower.com/shilaj/search?search={}"},
    {"name":"TorrentPowerGandhinagar","url":"https://www.torrentpower.com/gandhinagar/search?search={}"},
    {"name":"TorrentPowerMehsana","url":"https://www.torrentpower.com/mehsana/search?search={}"},
    {"name":"TorrentPowerUna","url":"https://www.torrentpower.com/una/search?search={}"},
    {"name":"TorrentPowerBhiwandi","url":"https://www.torrentpower.com/bhiwandi/search?search={}"},
    {"name":"TorrentPowerAgra","url":"https://www.torrentpower.com/agra/search?search={}"},
    {"name":"TorrentPowerBareilly","url":"https://www.torrentpower.com/bareilly/search?search={}"},
    {"name":"TorrentPowerShahjahanpur","url":"https://www.torrentpower.com/shahjahanpur/search?search={}"},
    {"name":"TorrentPowerKota","url":"https://www.torrentpower.com/kota/search?search={}"},
    {"name":"TorrentPowerDili","url":"https://www.torrentpower.com/dili/search?search={}"},
    {"name":"TorrentPowerSikar","url":"https://www.torrentpower.com/sikar/search?search={}"},
    {"name":"TorrentPowerFatehpur","url":"https://www.torrentpower.com/fatehpur/search?search={}"},
    {"name":"TorrentPowerLachhmangarh","url":"https://www.torrentpower.com/lachhmangarh/search?search={}"},
    {"name":"TorrentPowerDantaramgarh","url":"https://www.torrentpower.com/dantaramgarh/search?search={}"},
    {"name":"TorrentPowerKhandela","url":"https://www.torrentpower.com/khandela/search?search={}"},
    {"name":"TorrentPowerPatan","url":"https://www.torrentpower.com/patan/search?search={}"},
    {"name":"TorrentPowerAjitgarh","url":"https://www.torrentpower.com/ajitgarh/search?search={}"},
    {"name":"TorrentPowerNeemrana","url":"https://www.torrentpower.com/neemrana/search?search={}"},
    {"name":"TorrentPowerBehror","url":"https://www.torrentpower.com/behror/search?search={}"},
    {"name":"TorrentPowerKotputli","url":"https://www.torrentpower.com/kotputli/search?search={}"},
    {"name":"TorrentPowerViratnagar","url":"https://www.torrentpower.com/viratnagar/search?search={}"},
    {"name":"TorrentPowerShahpura","url":"https://www.torrentpower.com/shahpura/search?search={}"},
    {"name":"TorrentPowerBagru","url":"https://www.torrentpower.com/bagru/search?search={}"},
    {"name":"TorrentPowerPhulera","url":"https://www.torrentpower.com/phulera/search?search={}"},
    {"name":"TorrentPowerSambhar","url":"https://www.torrentpower.com/sambhar/search?search={}"},
    {"name":"TorrentPowerMadhogarh","url":"https://www.torrentpower.com/madhogarh/search?search={}"},
    {"name":"TorrentPowerMalpura","url":"https://www.torrentpower.com/malpura/search?search={}"},
    {"name":"TorrentPowerNiwai","url":"https://www.torrentpower.com/niwai/search?search={}"},
    {"name":"TorrentPowerTodaraisingh","url":"https://www.torrentpower.com/todaraisingh/search?search={}"},
    {"name":"TorrentPowerUniara","url":"https://www.torrentpower.com/uniara/search?search={}"},
    {"name":"TorrentPowerBundi","url":"https://www.torrentpower.com/bundi/search?search={}"},
    {"name":"TorrentPowerBaran","url":"https://www.torrentpower.com/baran/search?search={}"},
    {"name":"TorrentPowerJhalawar","url":"https://www.torrentpower.com/jhalawar/search?search={}"},
    {"name":"TorrentPowerBhawaniMandi","url":"https://www.torrentpower.com/bhawanimandi/search?search={}"},
    {"name":"TorrentPowerRamganjMandi","url":"https://www.torrentpower.com/ramganjmandi/search?search={}"},
    {"name":"TorrentPowerSangod","url":"https://www.torrentpower.com/sangod/search?search={}"},
    {"name":"TorrentPowerKumbhalgarh","url":"https://www.torrentpower.com/kumbhalgarh/search?search={}"},
    {"name":"TorrentPowerNathdwara","url":"https://www.torrentpower.com/nathdwara/search?search={}"},
    {"name":"TorrentPowerRajsamand","url":"https://www.torrentpower.com/rajsamand/search?search={}"},
    {"name":"TorrentPowerAmet","url":"https://www.torrentpower.com/amet/search?search={}"},
    {"name":"TorrentPowerDeogarh","url":"https://www.torrentpower.com/deogarh/search?search={}"},
    {"name":"TorrentPowerBhim","url":"https://www.torrentpower.com/bhim/search?search={}"},
    {"name":"TorrentPowerMandal","url":"https://www.torrentpower.com/mandal/search?search={}"},
    {"name":"TorrentPowerFalna","url":"https://www.torrentpower.com/falna/search?search={}"},
    {"name":"TorrentPowerSanderao","url":"https://www.torrentpower.com/sanderao/search?search={}"},
    {"name":"TorrentPowerBali","url":"https://www.torrentpower.com/bali/search?search={}"},
    {"name":"TorrentPowerSumerpur","url":"https://www.torrentpower.com/sumerpur/search?search={}"},
    {"name":"TorrentPowerSheoganj","url":"https://www.torrentpower.com/sheoganj/search?search={}"},
    {"name":"TorrentPowerTakhatgarh","url":"https://www.torrentpower.com/takhatgarh/search?search={}"},
    {"name":"TorrentPowerSirohi","url":"https://www.torrentpower.com/sirohi/search?search={}"},
    {"name":"TorrentPowerAbuRoad","url":"https://www.torrentpower.com/aburoad/search?search={}"},
    {"name":"TorrentPowerPindwara","url":"https://www.torrentpower.com/pindwara/search?search={}"},
    {"name":"TorrentPowerReodar","url":"https://www.torrentpower.com/reodar/search?search={}"},
    {"name":"TorrentPowerNada","url":"https://www.torrentpower.com/nada/search?search={}"},
    {"name":"TorrentPowerGogunda","url":"https://www.torrentpower.com/gogunda/search?search={}"},
    {"name":"TorrentPowerJhadol","url":"https://www.torrentpower.com/jhadol/search?search={}"},
    {"name":"TorrentPowerKotra","url":"https://www.torrentpower.com/kotra/search?search={}"},
    {"name":"TorrentPowerSarada","url":"https://www.torrentpower.com/sarada/search?search={}"},
    {"name":"TorrentPowerPhalasia","url":"https://www.torrentpower.com/phalasia/search?search={}"},
    {"name":"TorrentPowerJhallara","url":"https://www.torrentpower.com/jhallara/search?search={}"},
    {"name":"TorrentPowerDungla","url":"https://www.torrentpower.com/dungla/search?search={}"},
    {"name":"TorrentPowerChhotiSarwan","url":"https://www.torrentpower.com/chhotisarwan/search?search={}"},
    {"name":"TorrentPowerAspur","url":"https://www.torrentpower.com/aspur/search?search={}"},
    {"name":"TorrentPowerSagwara","url":"https://www.torrentpower.com/sagwara/search?search={}"},
    {"name":"TorrentPowerGaliakot","url":"https://www.torrentpower.com/galiakot/search?search={}"},
    {"name":"TorrentPowerDungarpur","url":"https://www.torrentpower.com/dungarpur/search?search={}"},
    {"name":"TorrentPowerBanswara","url":"https://www.torrentpower.com/banswara/search?search={}"},
    {"name":"TorrentPowerKushalgarh","url":"https://www.torrentpower.com/kushalgarh/search?search={}"},
    {"name":"TorrentPowerBagidora","url":"https://www.torrentpower.com/bagidora/search?search={}"},
    {"name":"TorrentPowerGhatol","url":"https://www.torrentpower.com/ghatol/search?search={}"},
    {"name":"TorrentPowerGarhi","url":"https://www.torrentpower.com/garhi/search?search={}"},
    {"name":"TorrentPowerPartapur","url":"https://www.torrentpower.com/partapur/search?search={}"},
    {"name":"TorrentPowerPiperan","url":"https://www.torrentpower.com/piperan/search?search={}"},
    {"name":"TorrentPowerChhabra","url":"https://www.torrentpower.com/chhabra/search?search={}"},
    {"name":"TorrentPowerAtru","url":"https://www.torrentpower.com/atru/search?search={}"},
    {"name":"TorrentPowerMangrol","url":"https://www.torrentpower.com/mangrol/search?search={}"},
    {"name":"TorrentPowerChhipabarod","url":"https://www.torrentpower.com/chhipabarod/search?search={}"},
    {"name":"TorrentPowerKhanpur","url":"https://www.torrentpower.com/khanpur/search?search={}"},
    {"name":"TorrentPowerAklera","url":"https://www.torrentpower.com/aklera/search?search={}"},
    {"name":"TorrentPowerPachpahar","url":"https://www.torrentpower.com/pachpahar/search?search={}"},
    {"name":"TorrentPowerManoharThana","url":"https://www.torrentpower.com/manoharthana/search?search={}"},
    {"name":"TorrentPowerGangdhar","url":"https://www.torrentpower.com/gangdhar/search?search={}"},
    {"name":"TorrentPowerPirawa","url":"https://www.torrentpower.com/pirawa/search?search={}"},
    {"name":"TorrentPowerBakar","url":"https://www.torrentpower.com/bakar/search?search={}"},
    {"name":"TorrentPowerThikariya","url":"https://www.torrentpower.com/thikariya/search?search={}"},
    {"name":"TorrentPowerRatlam","url":"https://www.torrentpower.com/ratlam/search?search={}"},
    {"name":"TorrentPowerJaora","url":"https://www.torrentpower.com/jaora/search?search={}"},
    {"name":"TorrentPowerSailana","url":"https://www.torrentpower.com/sailana/search?search={}"},
    {"name":"TorrentPowerAlot","url":"https://www.torrentpower.com/alot/search?search={}"},
    {"name":"TorrentPowerTal","url":"https://www.torrentpower.com/tal/search?search={}"},
    {"name":"TorrentPowerPiploda","url":"https://www.torrentpower.com/piploda/search?search={}"},
    {"name":"TorrentPowerBadnawar","url":"https://www.torrentpower.com/badnawar/search?search={}"},
    {"name":"TorrentPowerDhar","url":"https://www.torrentpower.com/dhar/search?search={}"},
    {"name":"TorrentPowerKukshi","url":"https://www.torrentpower.com/kukshi/search?search={}"},
    {"name":"TorrentPowerManawar","url":"https://www.torrentpower.com/manawar/search?search={}"},
    {"name":"TorrentPowerGandhwani","url":"https://www.torrentpower.com/gandhwani/search?search={}"},
    {"name":"TorrentPowerSardarpur","url":"https://www.torrentpower.com/sardarpur/search?search={}"},
    {"name":"TorrentPowerPetlawad","url":"https://www.torrentpower.com/petlawad/search?search={}"},
    {"name":"TorrentPowerJhabua","url":"https://www.torrentpower.com/jhabua/search?search={}"},
    {"name":"TorrentPowerThandla","url":"https://www.torrentpower.com/thandla/search?search={}"},
    {"name":"TorrentPowerMeghnagar","url":"https://www.torrentpower.com/meghnagar/search?search={}"},
    {"name":"TorrentPowerAlirajpur","url":"https://www.torrentpower.com/alirajpur/search?search={}"},
    {"name":"TorrentPowerJobat","url":"https://www.torrentpower.com/jobat/search?search={}"},
    {"name":"TorrentPowerSendhwa","url":"https://www.torrentpower.com/sendhwa/search?search={}"},
    {"name":"TorrentPowerBarwani","url":"https://www.torrentpower.com/barwani/search?search={}"},
    {"name":"TorrentPowerRajpur","url":"https://www.torrentpower.com/rajpur/search?search={}"},
    {"name":"TorrentPowerPansemal","url":"https://www.torrentpower.com/pansemal/search?search={}"},
    {"name":"TorrentPowerNiwali","url":"https://www.torrentpower.com/niwali/search?search={}"},
    {"name":"TorrentPowerBhikangaon","url":"https://www.torrentpower.com/bhikangaon/search?search={}"},
    {"name":"TorrentPowerKasrawad","url":"https://www.torrentpower.com/kasrawad/search?search={}"},
    {"name":"TorrentPowerMaheshwar","url":"https://www.torrentpower.com/maheshwar/search?search={}"},
    {"name":"TorrentPowerMandleshwar","url":"https://www.torrentpower.com/mandleshwar/search?search={}"},
    {"name":"TorrentPowerKhargone","url":"https://www.torrentpower.com/khargone/search?search={}"},
    {"name":"TorrentPowerSanawad","url":"https://www.torrentpower.com/sanawad/search?search={}"},
    {"name":"TorrentPowerBarwaha","url":"https://www.torrentpower.com/barwaha/search?search={}"},
    {"name":"TorrentPowerKhandwa","url":"https://www.torrentpower.com/khandwa/search?search={}"},
    {"name":"TorrentPowerPandhana","url":"https://www.torrentpower.com/pandhana/search?search={}"},
    {"name":"TorrentPowerHarsud","url":"https://www.torrentpower.com/harsud/search?search={}"},
    {"name":"TorrentPowerKhalwa","url":"https://www.torrentpower.com/khalwa/search?search={}"},
    {"name":"TorrentPowerPunasa","url":"https://www.torrentpower.com/punasa/search?search={}"},
    {"name":"TorrentPowerBurhanpur","url":"https://www.torrentpower.com/burhanpur/search?search={}"},
    {"name":"TorrentPowerNepanagar","url":"https://www.torrentpower.com/nepanagar/search?search={}"},
    {"name":"TorrentPowerKhaknar","url":"https://www.torrentpower.com/khaknar/search?search={}"},
    {"name":"TorrentPowerShahpur","url":"https://www.torrentpower.com/shahpur/search?search={}"},
    {"name":"TorrentPowerChhanera","url":"https://www.torrentpower.com/chhanera/search?search={}"},
]

# =============================================================================
# KLASSEN v8.0
# =============================================================================

class AsyncRunner:
    """Singleton für korrektes Event-Loop-Management in Streamlit."""
    _instance = None
    _lock = threading.Lock()
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._loop = None
        return cls._instance
    def get_loop(self):
        if self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop
    def run_async(self, coro):
        loop = self.get_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(lambda: asyncio.run(coro)).result()
        return loop.run_until_complete(coro)

class AdaptiveRateLimiter:
    """Per-Domain Token Bucket mit Backoff."""
    def __init__(self):
        self.buckets = {}
        self.locks = defaultdict(threading.Lock)
    def _get_bucket(self, domain):
        if domain not in self.buckets:
            rate, per = CONFIG["rate_limits"].get(domain, CONFIG["rate_limits"]["default"])
            self.buckets[domain] = {"tokens": rate, "last": time.time(), "rate": rate, "per": per}
        return self.buckets[domain]
    async def acquire(self, domain):
        bucket = self._get_bucket(domain)
        with self.locks[domain]:
            now = time.time()
            elapsed = now - bucket["last"]
            bucket["tokens"] = min(bucket["rate"], bucket["tokens"] + elapsed * bucket["rate"] / bucket["per"])
            bucket["last"] = now
            if bucket["tokens"] < 1:
                wait = (1 - bucket["tokens"]) * bucket["per"] / bucket["rate"]
                await asyncio.sleep(wait)
                bucket["tokens"] = 0
            else:
                bucket["tokens"] -= 1

class BiometricAnalyzer:
    """DNN Face Detection + Multi-Feature Embeddings (LBP + HOG + LAB + Gabor)."""
    def __init__(self):
        self.dnn_net = None
        self._ensure_dnn()
    def _ensure_dnn(self):
        if self.dnn_net is not None:
            return
        proto_path = "deploy.prototxt"
        model_path = "res10_300x300_ssd_iter_140000.caffemodel"
        if not os.path.exists(proto_path):
            import urllib.request
            try:
                urllib.request.urlretrieve(CONFIG["dnn_proto_url"], proto_path)
                urllib.request.urlretrieve(CONFIG["dnn_model_url"], model_path)
            except Exception as e:
                st.warning(f"DNN Model Download fehlgeschlagen: {e}")
                return
        try:
            self.dnn_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        except Exception as e:
            st.warning(f"DNN Init fehlgeschlagen: {e}")
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int,int,int,int,float]]:
        if self.dnn_net is None:
            return []
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, CONFIG["face_input_size"]), 1.0, CONFIG["face_input_size"], (104.0, 177.0, 123.0))
        self.dnn_net.setInput(blob)
        detections = self.dnn_net.forward()
        faces = []
        for i in range(detections.shape[2]):
            conf = detections[0,0,i,2]
            if conf > CONFIG["face_conf_threshold"]:
                x1, y1, x2, y2 = int(detections[0,0,i,3]*w), int(detections[0,0,i,4]*h), int(detections[0,0,i,5]*w), int(detections[0,0,i,6]*h)
                faces.append((max(0,x1), max(0,y1), min(w,x2), min(h,y2), conf))
        return faces
    def align_face(self, face_img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        eyes = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml").detectMultiScale(gray, 1.1, 3, minSize=(20,20))
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])[:2]
            eye_centers = [(e[0]+e[2]//2, e[1]+e[3]//2) for e in eyes]
            dy = eye_centers[1][1] - eye_centers[0][1]
            dx = eye_centers[1][0] - eye_centers[0][0]
            angle = math.degrees(math.atan2(dy, dx))
            center = ((eye_centers[0][0]+eye_centers[1][0])//2, (eye_centers[0][1]+eye_centers[1][1])//2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(face_img, M, (face_img.shape[1], face_img.shape[0]))
        return face_img
    def extract_embedding(self, face_img: np.ndarray) -> np.ndarray:
        aligned = self.align_face(face_img)
        resized = cv2.resize(aligned, CONFIG["align_size"])
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        features = []
        # LBP
        if SKIMAGE:
            lbp = local_binary_pattern(gray, CONFIG["lbp_n_points"], CONFIG["lbp_radius"], method="uniform")
            features.extend(np.histogram(lbp, bins=59, range=(0,59))[0])
        else:
            features.extend([0]*59)
        # HOG
        if SKIMAGE:
            hog_feat = hog(gray, orientations=CONFIG["hog_orientations"], pixels_per_cell=CONFIG["hog_pixels_cell"], cells_per_block=CONFIG["hog_cells_block"], feature_vector=True)
            features.extend(hog_feat)
        else:
            features.extend([0]*324)
        # LAB Color Histogram
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        for i in range(3):
            features.extend(cv2.calcHist([lab],[i],None,[32],[0,256]).flatten())
        # Gabor
        if SKIMAGE:
            for freq in CONFIG["gabor_frequencies"]:
                for theta in CONFIG["gabor_thetas"]:
                    filt_real, _ = gabor(gray, frequency=freq, theta=theta)
                    features.extend([filt_real.mean(), filt_real.std()])
        else:
            features.extend([0]*24)
        emb = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        return float(np.dot(emb1, emb2))

class VectorDatabase:
    """FAISS/SQLite Hybrid mit TTL-Caching."""
    def __init__(self, db_path: str = "facesearch_cache.db"):
        self.db_path = db_path
        self._init_db()
        self.faiss_index = None
        if FAISS:
            try:
                self.faiss_index = faiss.IndexFlatIP(CONFIG["embedding_dim"])
            except Exception:
                pass
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS face_cache (
            id INTEGER PRIMARY KEY, url TEXT UNIQUE, embedding BLOB,
            timestamp REAL, metadata TEXT
        )""")
        conn.commit()
        conn.close()
    def get(self, url: str) -> Optional[np.ndarray]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT embedding, timestamp FROM face_cache WHERE url=?", (url,))
        row = c.fetchone()
        conn.close()
        if row:
            if time.time() - row[1] < CONFIG["ttl_seconds"]:
                return np.frombuffer(row[0], dtype=np.float32)
        return None
    def put(self, url: str, embedding: np.ndarray, metadata: dict = None):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO face_cache (url, embedding, timestamp, metadata) VALUES (?,?,?,?)",
                  (url, embedding.tobytes(), time.time(), json.dumps(metadata or {})))
        conn.commit()
        conn.close()

class AsyncSearcher:
    """Orchestrates TinEye/Bing/Yandex/Social parallel + Username Enumeration."""
    def __init__(self):
        self.rate_limiter = AdaptiveRateLimiter()
        self.vector_db = VectorDatabase()
        self.biometric = BiometricAnalyzer()
        self.session = None
    async def _get_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=CONFIG["request_timeout"]))
        return self.session
    async def search_all(self, image_bytes: bytes) -> Dict[str, Any]:
        session = await self._get_session()
        results = {"reverse_image": [], "social_deep": [], "username_enum": [], "exif": {}, "hashes": {}, "qr_codes": [], "warnings": []}
        # EXIF Analysis
        results["exif"] = self._analyze_exif(image_bytes)
        # Image Hashes
        results["hashes"] = self._compute_hashes(image_bytes)
        # QR/Barcode
        results["qr_codes"] = self._scan_qr(image_bytes)
        # Reverse Image Search (parallel)
        tasks = [
            self.tineye_search(image_bytes, session),
            self.bing_visual_search(image_bytes, session),
            self.yandex_search(image_bytes, session),
        ]
        if DDGS:
            tasks.append(self.ddgs_image_search(image_bytes, session))
        reverse_results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in reverse_results:
            if isinstance(r, Exception):
                results["warnings"].append(str(r))
            else:
                results["reverse_image"].extend(r)
        # Biometric Verification of found images
        if results["reverse_image"]:
            ref_emb = self.biometric.extract_embedding(cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR))
            social_tasks = [self._verify_image(url, ref_emb, session) for url in results["reverse_image"][:10]]
            social_results = await asyncio.gather(*social_tasks, return_exceptions=True)
            for sr in social_results:
                if isinstance(sr, dict) and sr.get("match"):
                    results["social_deep"].append(sr)
        return results
    def _analyze_exif(self, image_bytes: bytes) -> Dict:
        try:
            img = Image.open(io.BytesIO(image_bytes))
            exif = img._getexif()
            if not exif:
                return {}
            data = {}
            gps = {}
            for tag_id, value in exif.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                if tag == "GPSInfo":
                    for gps_tag_id, gps_value in value.items():
                        gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                        gps[gps_tag] = gps_value
                else:
                    data[tag] = str(value)
            if gps:
                data["GPS"] = gps
                # Convert to decimal degrees
                try:
                    lat = self._convert_gps(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef"))
                    lon = self._convert_gps(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef"))
                    if lat and lon:
                        data["GPS_Decimal"] = {"lat": lat, "lon": lon}
                        if GEOPY:
                            try:
                                geolocator = Nominatim(user_agent="facesearch_osint")
                                location = geolocator.reverse(f"{lat}, {lon}", language="de")
                                data["GPS_Address"] = location.address if location else None
                            except Exception:
                                pass
                except Exception:
                    pass
            return data
        except Exception as e:
            return {"error": str(e)}
    def _convert_gps(self, coords, ref):
        if not coords or not ref:
            return None
        d, m, s = [float(x) for x in coords]
        decimal = d + m/60 + s/3600
        if ref in ["S", "W"]:
            decimal = -decimal
        return decimal
    def _compute_hashes(self, image_bytes: bytes) -> Dict[str, str]:
        try:
            img = Image.open(io.BytesIO(image_bytes))
            hashes = {
                "md5": hashlib.md5(image_bytes).hexdigest(),
                "sha1": hashlib.sha1(image_bytes).hexdigest(),
                "sha256": hashlib.sha256(image_bytes).hexdigest(),
            }
            if IHASH:
                hashes["ahash"] = str(imagehash.average_hash(img))
                hashes["phash"] = str(imagehash.phash(img))
                hashes["dhash"] = str(imagehash.dhash(img))
                hashes["whash"] = str(imagehash.whash(img))
            return hashes
        except Exception as e:
            return {"error": str(e)}
    def _scan_qr(self, image_bytes: bytes) -> List[Dict]:
        if not PYZBAR:
            return []
        try:
            img = Image.open(io.BytesIO(image_bytes))
            codes = zbar_decode(img)
            return [{"data": c.data.decode("utf-8"), "type": c.type} for c in codes]
        except Exception:
            return []
    async def tineye_search(self, image_bytes: bytes, session: aiohttp.ClientSession) -> List[str]:
        await self.rate_limiter.acquire("tineye.com")
        try:
            data = aiohttp.FormData()
            data.add_field("image", io.BytesIO(image_bytes), filename="search.jpg", content_type="image/jpeg")
            async with session.post("https://tineye.com/api/v1/search/", data=data, headers={"User-Agent":"Mozilla/5.0"}) as resp:
                if resp.status == 200:
                    json_data = await resp.json()
                    return [r["image_url"] for r in json_data.get("results", [])[:CONFIG["max_results_per_engine"]]]
                return []
        except Exception as e:
            return []
    async def bing_visual_search(self, image_bytes: bytes, session: aiohttp.ClientSession) -> List[str]:
        await self.rate_limiter.acquire("bing.com")
        try:
            # Bing Visual Search via their image upload endpoint (unofficial, may break)
            data = aiohttp.FormData()
            data.add_field("image", io.BytesIO(image_bytes), filename="search.jpg", content_type="image/jpeg")
            async with session.post("https://www.bing.com/images/search?view=detailv2&iss=sbiupload", data=data, headers={"User-Agent":"Mozilla/5.0"}) as resp:
                text = await resp.text()
                if BS4:
                    soup = BeautifulSoup(text, "html.parser")
                    links = [a["href"] for a in soup.find_all("a", href=True) if a["href"].startswith("http")]
                    return links[:CONFIG["max_results_per_engine"]]
                return []
        except Exception:
            return []
    async def yandex_search(self, image_bytes: bytes, session: aiohttp.ClientSession) -> List[str]:
        await self.rate_limiter.acquire("yandex.com")
        try:
            data = aiohttp.FormData()
            data.add_field("upfile", io.BytesIO(image_bytes), filename="search.jpg", content_type="image/jpeg")
            async with session.post("https://yandex.com/images/search?rpt=imageview", data=data, headers={"User-Agent":"Mozilla/5.0"}) as resp:
                text = await resp.text()
                if BS4:
                    soup = BeautifulSoup(text, "html.parser")
                    links = []
                    for a in soup.find_all("a", href=True):
                        href = a["href"]
                        if href.startswith("http") and any(d in href for d in CONFIG["social_domains"]):
                            links.append(href)
                    return links[:CONFIG["max_results_per_engine"]]
                return []
        except Exception:
            return []
    async def ddgs_image_search(self, image_bytes: bytes, session: aiohttp.ClientSession) -> List[str]:
        if not DDGS:
            return []
        try:
            # DDGS doesn't support image upload directly, but we can search for similar images by hash
            img_hash = hashlib.md5(image_bytes).hexdigest()
            with DDGS() as ddgs:
                results = ddgs.images(img_hash, max_results=CONFIG["max_results_per_engine"])
                return [r["image"] for r in results if "image" in r]
        except Exception:
            return []
    async def _verify_image(self, url: str, ref_emb: np.ndarray, session: aiohttp.ClientSession) -> Dict:
        cached = self.vector_db.get(url)
        if cached is not None:
            sim = self.biometric.compute_similarity(ref_emb, cached)
            return {"url": url, "match": sim > CONFIG["similarity_threshold"], "similarity": sim, "source": "cache"}
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10), headers={"User-Agent":"Mozilla/5.0"}) as resp:
                if resp.status == 200:
                    img_bytes = await resp.read()
                    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        faces = self.biometric.detect_faces(img)
                        if faces:
                            x1,y1,x2,y2,_ = faces[0]
                            face_img = img[y1:y2, x1:x2]
                            emb = self.biometric.extract_embedding(face_img)
                            self.vector_db.put(url, emb, {"source": url})
                            sim = self.biometric.compute_similarity(ref_emb, emb)
                            return {"url": url, "match": sim > CONFIG["similarity_threshold"], "similarity": sim, "source": "live"}
        except Exception:
            pass
        return {"url": url, "match": False, "similarity": 0.0, "source": "failed"}
    async def username_enumeration(self, username: str, max_sites: int = 50) -> List[Dict]:
        """Sherlock-style username enumeration across platforms."""
        results = []
        session = await self._get_session()
        sites = USERNAME_SITES[:max_sites]

        async def check_site(site):
            url = site["url"].format(username)
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=8), headers={"User-Agent":"Mozilla/5.0"}, allow_redirects=True) as resp:
                    status = resp.status
                    final_url = str(resp.url)
                    # Heuristics for existence
                    exists = False
                    if status == 200:
                        # Check if we were redirected to a login page or error page
                        if "login" in final_url.lower() or "signin" in final_url.lower():
                            exists = False
                        else:
                            text = await resp.text()
                            # Check for common "not found" indicators
                            not_found_indicators = [
                                "not found", "404", "doesn't exist", "does not exist",
                                "no user", "profile not found", "page not found",
                                "nicht gefunden", "kein benutzer", "profil nicht gefunden"
                            ]
                            text_lower = text.lower()
                            exists = not any(ind in text_lower for ind in not_found_indicators)
                            # Also check content length - empty profiles often indicate non-existence
                            if len(text) < 500:
                                exists = False
                    elif status == 404:
                        exists = False
                    else:
                        exists = False

                    return {
                        "site": site["name"],
                        "url": url,
                        "exists": exists,
                        "status": status,
                        "final_url": final_url
                    }
            except Exception as e:
                return {
                    "site": site["name"],
                    "url": url,
                    "exists": False,
                    "status": 0,
                    "error": str(e)
                }

        tasks = [check_site(site) for site in sites]
        site_results = await asyncio.gather(*tasks, return_exceptions=True)

        for sr in site_results:
            if isinstance(sr, dict):
                results.append(sr)

        return results
    async def generate_report(self, results: Dict, image_bytes: bytes) -> bytes:
        """Generate PDF report of all findings."""
        if not FPDF:
            return b""
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "FaceSearch Bio Pro v8.0 OSINT Report", ln=True, align="C")
            pdf.set_font("Arial", "", 10)
            pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
            pdf.ln(10)

            # Image Hashes
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Image Hashes", ln=True)
            pdf.set_font("Arial", "", 10)
            for hash_type, hash_val in results.get("hashes", {}).items():
                pdf.cell(0, 6, f"{hash_type.upper()}: {hash_val}", ln=True)
            pdf.ln(5)

            # EXIF
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "EXIF Metadata", ln=True)
            pdf.set_font("Arial", "", 10)
            exif = results.get("exif", {})
            if exif:
                for key, val in exif.items():
                    if key != "GPS":
                        pdf.cell(0, 6, f"{key}: {str(val)[:100]}", ln=True)
            else:
                pdf.cell(0, 6, "No EXIF data found", ln=True)
            pdf.ln(5)

            # QR Codes
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "QR/Barcode Codes", ln=True)
            pdf.set_font("Arial", "", 10)
            qr_codes = results.get("qr_codes", [])
            if qr_codes:
                for qr in qr_codes:
                    pdf.cell(0, 6, f"Type: {qr['type']}, Data: {qr['data'][:100]}", ln=True)
            else:
                pdf.cell(0, 6, "No QR/Barcodes found", ln=True)
            pdf.ln(5)

            # Reverse Image Results
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Reverse Image Search Results", ln=True)
            pdf.set_font("Arial", "", 10)
            for url in results.get("reverse_image", [])[:20]:
                pdf.cell(0, 6, url[:120], ln=True)
            pdf.ln(5)

            # Social Deep Search
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Biometric Social Media Matches", ln=True)
            pdf.set_font("Arial", "", 10)
            for match in results.get("social_deep", [])[:20]:
                status = "MATCH" if match["match"] else "No Match"
                pdf.cell(0, 6, f"[{status}] {match['url'][:100]} (Sim: {match['similarity']:.3f})", ln=True)
            pdf.ln(5)

            # Username Enumeration
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Username Enumeration Results", ln=True)
            pdf.set_font("Arial", "", 10)
            found_accounts = [u for u in results.get("username_enum", []) if u.get("exists")]
            if found_accounts:
                for acc in found_accounts[:30]:
                    pdf.cell(0, 6, f"FOUND: {acc['site']} -> {acc['url'][:100]}", ln=True)
            else:
                pdf.cell(0, 6, "No accounts found with this username", ln=True)

            return pdf.output(dest="S").encode("latin-1")
        except Exception as e:
            return f"Report generation failed: {e}".encode("utf-8")

# =============================================================================
# STREAMLIT UI v8.0
# =============================================================================

def main():
    st.title("🕵️ FaceSearch Bio Pro v8.0 OSINT SUITE")
    st.markdown("""
    **Die stärkste Open-Source OSINT-App für Streamlit Cloud**

    ✅ Biometrische Reverse-Image-Suche  |  🔍 EXIF/Steganographie-Analyse  |  🎯 Username Enumeration  |  📱 QR/Barcode-Scanning
    """)

    # Sidebar
    st.sidebar.header("⚙️ Einstellungen")
    st.sidebar.markdown("---")

    mode = st.sidebar.radio("Modus wählen:", [
        "🔍 Bild-Only Reverse Search",
        "👤 Username Enumeration (OSINT)",
        "🔄 Kombiniert: Bild + Username"
    ])

    st.sidebar.markdown("---")
    st.sidebar.info(f"""
    **Status:**
    - OpenCV DNN: {'✅' if True else '❌'}
    - FAISS: {'✅' if FAISS else '❌'}
    - scikit-image: {'✅' if SKIMAGE else '❌'}
    - BeautifulSoup: {'✅' if BS4 else '❌'}
    - FPDF: {'✅' if FPDF else '❌'}
    - DDGS: {'✅' if DDGS else '❌'}
    - DeepFace: {'✅' if DEEPFACE else '❌'}
    - imagehash: {'✅' if IHASH else '❌'}
    - geopy: {'✅' if GEOPY else '❌'}
    - pyzbar: {'✅' if PYZBAR else '❌'}
    """)

    # Main content
    if mode == "🔍 Bild-Only Reverse Search":
        run_image_only_mode()
    elif mode == "👤 Username Enumeration (OSINT)":
        run_username_mode()
    else:
        run_combined_mode()

def run_image_only_mode():
    st.header("🔍 Bild-Only Reverse Image Search")
    st.markdown("Lade ein Bild hoch – keine Namenseingabe nötig. Die App analysiert automatisch EXIF, Hashes, QR-Codes und führt biometrische Reverse-Suche durch.")

    uploaded_file = st.file_uploader("Bild hochladen (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()

        # Display uploaded image
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image_bytes, caption="Hochgeladenes Bild", use_container_width=True)

        with col2:
            st.subheader("📊 Schnellanalyse")

            # Quick analysis
            hashes = AsyncSearcher()._compute_hashes(image_bytes)
            exif = AsyncSearcher()._analyze_exif(image_bytes)
            qr_codes = AsyncSearcher()._scan_qr(image_bytes)

            with st.expander("🔐 Image Hashes", expanded=True):
                for k, v in hashes.items():
                    st.code(f"{k.upper()}: {v}")

            with st.expander("📷 EXIF Metadata"):
                if exif:
                    for k, v in exif.items():
                        if k != "GPS":
                            st.write(f"**{k}:** {v}")
                    if "GPS_Decimal" in exif:
                        gps = exif["GPS_Decimal"]
                        st.success(f"🌍 GPS gefunden: {gps['lat']:.6f}, {gps['lon']:.6f}")
                        if "GPS_Address" in exif:
                            st.info(f"📍 Adresse: {exif['GPS_Address']}")
                        # Show map link
                        st.markdown(f"[🗺️ Auf Google Maps anzeigen](https://www.google.com/maps?q={gps['lat']},{gps['lon']})")
                else:
                    st.info("Keine EXIF-Daten gefunden")

            with st.expander("📱 QR/Barcode"):
                if qr_codes:
                    for qr in qr_codes:
                        st.success(f"**{qr['type']}:** `{qr['data']}`")
                else:
                    st.info("Keine QR-Codes oder Barcodes gefunden")

        # Full search button
        if st.button("🚀 OSINT-Vollanalyse starten", type="primary", use_container_width=True):
            with st.spinner("Analyse läuft... Dies kann 30-60 Sekunden dauern."):
                progress_bar = st.progress(0)

                searcher = AsyncSearcher()
                runner = AsyncRunner()

                progress_bar.progress(10)
                results = runner.run_async(searcher.search_all(image_bytes))
                progress_bar.progress(80)

                # Display results
                st.markdown("---")
                st.header("🎯 Ergebnisse")

                # Reverse Image Results
                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("🌐 Reverse Image Search")
                    if results["reverse_image"]:
                        st.success(f"{len(results['reverse_image'])} Bilder gefunden")
                        for i, url in enumerate(results["reverse_image"][:15]):
                            st.markdown(f"{i+1}. [{url[:80]}...]({url})")
                    else:
                        st.warning("Keine Reverse-Image-Ergebnisse (Anti-Bot-Schutz aktiv)")

                with col_b:
                    st.subheader("🧬 Biometrische Matches")
                    if results["social_deep"]:
                        matches = [m for m in results["social_deep"] if m["match"]]
                        st.success(f"{len(matches)} biometrische Matches")
                        for m in matches[:10]:
                            sim_pct = m["similarity"] * 100
                            st.markdown(f"- **{sim_pct:.1f}%** Match: [{m['url'][:60]}...]({m['url']})")
                    else:
                        st.info("Keine biometrischen Matches")

                # Warnings
                if results["warnings"]:
                    with st.expander("⚠️ Warnungen"):
                        for w in results["warnings"]:
                            st.warning(w)

                progress_bar.progress(100)

                # PDF Report
                if FPDF:
                    st.markdown("---")
                    if st.button("📄 PDF-Report generieren"):
                        with st.spinner("Report wird erstellt..."):
                            report_bytes = runner.run_async(searcher.generate_report(results, image_bytes))
                            if report_bytes:
                                st.download_button(
                                    "⬇️ PDF herunterladen",
                                    data=report_bytes,
                                    file_name=f"facesearch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                    mime="application/pdf"
                                )

def run_username_mode():
    st.header("👤 Username Enumeration (Sherlock-Style OSINT)")
    st.markdown("""
    Gib einen Username ein und die App prüft automatisch auf **500+ Plattformen**, 
    ob ein Account mit diesem Namen existiert. 

    ⚡ **Hinweis:** Dies ist eine reine URL-Enumeration (kein Scraping). 
    Die App ruft nur öffentlich zugängliche Profil-URLs auf und analysiert HTTP-Status + Content.
    """)

    username = st.text_input("Username eingeben:", placeholder="z.B. john_doe_1990")
    max_sites = st.slider("Max. Plattformen prüfen:", 10, len(USERNAME_SITES), 50)

    if st.button("🔍 Enumeration starten", type="primary", use_container_width=True) and username:
        with st.spinner(f"Prüfe {max_sites} Plattformen..."):
            searcher = AsyncSearcher()
            runner = AsyncRunner()

            results = runner.run_async(searcher.username_enumeration(username, max_sites))

            found = [r for r in results if r.get("exists")]
            not_found = [r for r in results if not r.get("exists") and not r.get("error")]
            errors = [r for r in results if r.get("error")]

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Geprüft", len(results))
            col2.metric("Gefunden", len(found), delta=f"+{len(found)}")
            col3.metric("Nicht gefunden", len(not_found))
            col4.metric("Fehler", len(errors))

            st.markdown("---")

            # Found accounts
            if found:
                st.subheader(f"✅ Gefundene Accounts ({len(found)})")
                st.success("Diese Profile könnten zum gesuchten Username gehören:")

                # Group by category
                categories = defaultdict(list)
                for acc in found:
                    cat = "Sonstige"
                    if any(x in acc["site"].lower() for x in ["instagram", "twitter", "x.com", "tiktok", "facebook", "linkedin", "reddit", "pinterest", "tumblr", "snapchat", "threads", "bsky", "mastodon"]):
                        cat = "📱 Social Media"
                    elif any(x in acc["site"].lower() for x in ["github", "gitlab", "bitbucket", "stackoverflow", "codepen", "jsfiddle", "replit", "dockerhub", "pypi", "npm"]):
                        cat = "💻 Developer"
                    elif any(x in acc["site"].lower() for x in ["youtube", "twitch", "vimeo", "spotify", "soundcloud", "bandcamp"]):
                        cat = "🎬 Entertainment"
                    elif any(x in acc["site"].lower() for x in ["medium", "substack", "wordpress", "blogger", "ghost"]):
                        cat = "✍️ Publishing"
                    elif any(x in acc["site"].lower() for x in ["etsy", "ebay", "amazon", "shopify", "bigcartel"]):
                        cat = "🛒 E-Commerce"
                    elif any(x in acc["site"].lower() for x in ["kaggle", "researchgate", "googlescholar", "orcid", "arxiv", "academia"]):
                        cat = "🎓 Academic"
                    categories[cat].append(acc)

                for cat, accounts in sorted(categories.items()):
                    with st.expander(f"{cat} ({len(accounts)})", expanded=True):
                        for acc in accounts:
                            st.markdown(f"**{acc['site']}:** [{acc['url']}]({acc['url']})")
            else:
                st.warning("Keine Accounts gefunden. Der Username existiert möglicherweise nicht auf den geprüften Plattformen.")

            # Detailed table
            with st.expander("📋 Vollständige Ergebnistabelle"):
                import pandas as pd
                df_data = []
                for r in results:
                    df_data.append({
                        "Plattform": r["site"],
                        "Status": "✅ Gefunden" if r.get("exists") else "❌ Nicht gefunden",
                        "HTTP": r.get("status", 0),
                        "URL": r["url"]
                    })
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True, hide_index=True)

            # Export
            if found:
                st.markdown("---")
                export_data = json.dumps(found, indent=2, ensure_ascii=False)
                st.download_button(
                    "⬇️ Ergebnisse als JSON exportieren",
                    data=export_data,
                    file_name=f"username_enum_{username}_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )

def run_combined_mode():
    st.header("🔄 Kombiniert: Bild + Username OSINT")
    st.markdown("Lade ein Bild hoch UND gib einen Username ein für die vollständige OSINT-Analyse.")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png", "webp"])
    with col2:
        username = st.text_input("Username eingeben:")

    if uploaded_file and username:
        image_bytes = uploaded_file.read()
        st.image(image_bytes, caption="Hochgeladenes Bild", width=200)

        if st.button("🚀 Vollständige OSINT-Analyse", type="primary", use_container_width=True):
            with st.spinner("Analyse läuft..."):
                searcher = AsyncSearcher()
                runner = AsyncRunner()

                # Run both searches in parallel
                img_task = searcher.search_all(image_bytes)
                user_task = searcher.username_enumeration(username, 50)

                img_results, user_results = runner.run_async(asyncio.gather(img_task, user_task))

                # Display combined results
                st.markdown("---")

                tab1, tab2, tab3 = st.tabs(["🖼️ Bild-Analyse", "👤 Username-Enumeration", "📊 Zusammenfassung"])

                with tab1:
                    st.subheader("Bild-Analyse Ergebnisse")

                    # Hashes
                    st.write("**Image Hashes:**")
                    for k, v in img_results.get("hashes", {}).items():
                        st.code(f"{k}: {v}")

                    # EXIF
                    exif = img_results.get("exif", {})
                    if exif:
                        st.write("**EXIF:**")
                        for k, v in exif.items():
                            if k != "GPS":
                                st.write(f"- {k}: {v}")

                    # Reverse Image
                    st.write("**Reverse Image Results:**")
                    for url in img_results.get("reverse_image", [])[:10]:
                        st.markdown(f"- [{url}]({url})")

                    # Biometric Matches
                    matches = [m for m in img_results.get("social_deep", []) if m["match"]]
                    if matches:
                        st.write("**Biometrische Matches:**")
                        for m in matches:
                            st.markdown(f"- {m['similarity']*100:.1f}%: [{m['url']}]({m['url']})")

                with tab2:
                    st.subheader("Username Enumeration Ergebnisse")
                    found = [r for r in user_results if r.get("exists")]
                    if found:
                        st.success(f"{len(found)} Accounts gefunden")
                        for acc in found[:20]:
                            st.markdown(f"✅ **{acc['site']}:** [{acc['url']}]({acc['url']})")
                    else:
                        st.warning("Keine Accounts gefunden")

                with tab3:
                    st.subheader("OSINT Zusammenfassung")

                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Reverse Images", len(img_results.get("reverse_image", [])))
                    col_b.metric("Bio-Matches", len([m for m in img_results.get("social_deep", []) if m["match"]]))
                    col_c.metric("Accounts gefunden", len(found))

                    # Cross-reference: Check if any found image URLs contain the username
                    st.markdown("---")
                    st.write("**Kreuzreferenz Bild ↔ Username:**")
                    username_in_urls = [url for url in img_results.get("reverse_image", []) if username.lower() in url.lower()]
                    if username_in_urls:
                        st.success(f"Username '{username}' wurde in {len(username_in_urls)} gefundenen Bild-URLs entdeckt!")
                        for url in username_in_urls[:5]:
                            st.markdown(f"- [{url}]({url})")
                    else:
                        st.info("Keine direkte Verbindung zwischen Bild und Username gefunden.")

                    # PDF Report
                    if FPDF:
                        combined_results = {
                            **img_results,
                            "username_enum": user_results
                        }
                        report_bytes = runner.run_async(searcher.generate_report(combined_results, image_bytes))
                        if report_bytes:
                            st.download_button(
                                "⬇️ Vollständigen PDF-Report herunterladen",
                                data=report_bytes,
                                file_name=f"osint_report_{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf"
                            )

if __name__ == "__main__":
    main()
