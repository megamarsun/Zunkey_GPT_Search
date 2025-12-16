# Zunkey_GPT_Search.py
# Zunkey_GPT_Search 1.0.3
# Tkinter + DDG検索 + HTML抽出 + ローカルLLM要約(任意) + RAG + 関連ワード（旧UI維持）
# 仕様:
# ・Python実行とNuitka（standalone）双方対応
# ・GPT-OSSモデルは起動時に必ず自動ダウンロード（保存先はアプリ直下 models/gpt-oss）
# ・UIは旧版（Zunkey_GPTwiki系）を維持
# ・関連ワードの精度を上げるため、採用条件（hits/閾値）を強化し暴走を抑制

import os
import sys
import re
import json
import time
import queue
import sqlite3
import random
import hashlib
import threading
import subprocess
import urllib.parse
import webbrowser
from datetime import datetime

APP = "Zunkey_GPT_Search"
VER = "1.0.16"

# ==================== path helpers（Python/EXE両対応） ====================
def is_frozen_app() -> bool:
    if "__compiled__" in globals():  # Nuitka
        return True
    if bool(getattr(sys, "frozen", False)):  # PyInstaller等
        return True
    if hasattr(sys, "_MEIPASS"):
        return True
    return False

def exe_dir() -> str:
    try:
        p = sys.executable if is_frozen_app() else sys.argv[0]
        return os.path.dirname(os.path.abspath(p))
    except Exception:
        return os.getcwd()

BASE_DIR = exe_dir()

def must_writable_dir(path: str):
    os.makedirs(path, exist_ok=True)
    test = os.path.join(path, "_zunkey_write_test.tmp")
    with open(test, "w", encoding="utf-8") as f:
        f.write("ok")
    os.remove(test)

# アプリ直下に固定（要件）
DATA_DIR = BASE_DIR

DB_PATH = os.path.join(DATA_DIR, "zunkey_gpt_search.sqlite3")
SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")
LOG_PATH = os.path.join(DATA_DIR, "zunkey_gpt_search_runtime.log")

MODELS_DIR = os.path.join(DATA_DIR, "models")
GPTOSS_DIR = os.path.join(MODELS_DIR, "gpt-oss")
GPTOSS_FILE = "gpt-oss-20b-Q5_K_M.gguf"

# ==================== defaults ====================
DEFAULT = dict(
    topic="",

    # LLM（llama.cpp）
    use_llm=True,  # UIトグルあり
    llm_model_path=os.path.join("models", "gpt-oss", GPTOSS_FILE),  # 相対扱い（DATA_DIR基準）
    llm_model_url="https://huggingface.co/unsloth/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-Q5_K_M.gguf?download=true",
    llm_n_ctx=4096,
    llm_n_threads=max(4, os.cpu_count() or 8),
    llm_n_gpu_layers=-1,
    llm_temperature=0.2,
    llm_top_p=0.9,
    llm_repeat_penalty=1.1,
    llm_timeout_sec=90,
    llm_max_tokens=520,

    # Embedding（入っていればsentence-transformers、無ければhash）
    embed_dim=384,
    embedder_name="sentence-transformers/all-MiniLM-L6-v2",

    # Search
    search_region="jp-jp",
    force_html_search_when_frozen=True,
    search_safesearch="moderate",
    search_results=8,
    search_timeout_sec=25,
    max_pages_per_tick=2,
    tick_sleep_sec=2.0,

    # Page processing
    min_text_len=900,
    max_store_chars=25000,
    max_kw_chars=14000,

    # LLMへ渡す本文の事前カット
    chunk_chars_for_llm=2800,
    summary_chars=1000,

    # score weights
    w_rel=0.65,
    w_qua=0.35,

    # RAG
    rag_topk=8,
    rag_min_sim=0.24,  # 何でも引っかかる対策（hash環境で上げ気味）

    # related terms（旧機能復活。精度改善）
    use_related=True,
    related_use_for_query=True,   # 旧挙動：関連語をクエリに混ぜる（暴走しない条件つき）
    related_pick_prob=0.75,
    rel_sim_threshold=0.24,       # topic-term のcos類似しきい値（少し厳しく）
    related_min_hits=2,           # 複数回登場した語のみ採用
    related_min_score=58.0,       # 関連語スコア下限（暴走防止）
    related_limit=200,

    avoid_domains=["x.com", "twitter.com", "facebook.com", "bsky.app"],
)

# ==================== utils ====================
def now_hms():
    return time.strftime("%H:%M:%S")

def iso_utc():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def clamp(s: str, n: int) -> str:
    if not s:
        return ""
    return s if len(s) <= n else s[:n]

def strip_ctrl(s: str) -> str:
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", s or "")

def norm_ws(s: str) -> str:
    s = (s or "").replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8", "ignore")).hexdigest()

def stable_hash32(s: str) -> int:
    """Stable 32-bit hash for fallback embeddings (avoid Python's randomized hash())."""
    b = hashlib.blake2b((s or "").encode("utf-8", "ignore"), digest_size=4).digest()
    return int.from_bytes(b, "little", signed=False)

def norm_url(u: str) -> str:
    try:
        p = urllib.parse.urlsplit(u)
        p = p._replace(fragment="")
        return p.geturl()
    except Exception:
        return u

def get_domain(u: str) -> str:
    try:
        netloc = urllib.parse.urlparse(u).netloc.lower()
        if not netloc:
            return ""
        if "@" in netloc:
            netloc = netloc.split("@", 1)[1]
        if netloc.startswith("["):
            host = netloc.split("]")[0] + "]"
        else:
            host = netloc.split(":", 1)[0]
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""

def safe_json_obj(text: str):
    text = strip_ctrl(text or "")
    i = text.find("{")
    if i < 0:
        return None
    tail = text[i:]
    try:
        obj, _ = json.JSONDecoder().raw_decode(tail)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def write_json(path: str, obj):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def resolve_in_data(path: str) -> str:
    path = (path or "").strip()
    if not path:
        return ""
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(DATA_DIR, path))

def pip_install(pkgs, log):
    if not pkgs:
        return
    log(f"Installing missing packages: {pkgs}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", *pkgs])

def ensure_packages(log):
    # Python実行時のみpip導入。EXEでは同梱されてる前提（無ければエラーにする）
    def has(mod):
        try:
            __import__(mod)
            return True
        except Exception as e:
            log(f"import fail: {mod} ({type(e).__name__}: {e})")
            return False

    required = [
        ("httpx", "httpx>=0.28.1"),
        ("ddgs", "ddgs>=9.0.0"),
        ("bs4", "beautifulsoup4>=4.12.0"),
        ("lxml", "lxml>=4.9.4"),
        ("numpy", "numpy>=1.24.0"),
    ]
    optional = [
        ("trafilatura", "trafilatura>=1.9.0"),
    ]

    miss_req = [pip for mod, pip in required if not has(mod)]
    miss_opt = [pip for mod, pip in optional if not has(mod)]

    if is_frozen_app():
        if miss_req:
            log("EXEに必須パッケージが同梱されていません（ビルド見直し）: " + str(miss_req))
            return False
        if miss_opt:
            log("任意パッケージ未同梱 -> フォールバックで動作: " + str(miss_opt))
        return True

    if miss_req:
        pip_install(miss_req, log)
    if miss_opt:
        log("任意パッケージ未導入 -> フォールバックで動作: " + str(miss_opt))
    return True

def load_settings():
    s = dict(DEFAULT)
    j = read_json(SETTINGS_PATH)
    if isinstance(j, dict):
        s.update(j)
    return s

def save_settings(s):
    write_json(SETTINGS_PATH, s)

# ==================== DB ====================
class DB:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.lock = threading.Lock()
        cur = self.conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS docs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            domain TEXT,
            title TEXT,
            summary TEXT,
            cleaned TEXT,
            text_sha1 TEXT,
            created_at TEXT,
            relevance REAL,
            quality REAL,
            score REAL,
            emb BLOB
        );
        """)

        # related: 精度改善のため hits を追加
        cur.execute("""
        CREATE TABLE IF NOT EXISTS related(
            term TEXT PRIMARY KEY,
            score REAL,
            hits INTEGER DEFAULT 0,
            last_seen TEXT,
            source_url TEXT
        );
        """)

        self.conn.commit()
        self._ensure_columns()

    def _table_cols(self, table: str):
        c = self.conn.cursor()
        c.execute(f"PRAGMA table_info({table});")
        return {r[1] for r in c.fetchall()}

    def _ensure_columns(self):
        cols = self._table_cols("related")
        if "hits" not in cols:
            try:
                self.conn.execute("ALTER TABLE related ADD COLUMN hits INTEGER DEFAULT 0;")
                self.conn.commit()
            except Exception:
                pass

    def upsert_doc(self, url, domain, title, summary, cleaned, created_at, relevance, quality, score, emb):
        with self.lock:
            c = self.conn.cursor()
            c.execute("""
            INSERT INTO docs(url,domain,title,summary,cleaned,text_sha1,created_at,relevance,quality,score,emb)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(url) DO UPDATE SET
              domain=excluded.domain,
              title=excluded.title,
              summary=excluded.summary,
              cleaned=excluded.cleaned,
              text_sha1=excluded.text_sha1,
              created_at=excluded.created_at,
              relevance=excluded.relevance,
              quality=excluded.quality,
              score=excluded.score,
              emb=excluded.emb
            """, (
                url, domain, title, summary, cleaned, sha1(cleaned),
                created_at, float(relevance), float(quality), float(score), emb
            ))
            self.conn.commit()
            c.execute("SELECT id FROM docs WHERE url=?", (url,))
            r = c.fetchone()
            return int(r[0]) if r else -1

    def list_docs(self, limit=300):
        with self.lock:
            c = self.conn.cursor()
            c.execute("""
            SELECT id,score,relevance,quality,title,url,created_at
            FROM docs
            ORDER BY score DESC, id DESC
            LIMIT ?
            """, (int(limit),))
            return c.fetchall()

    def list_docs_full(self):
        with self.lock:
            c = self.conn.cursor()
            c.execute("""
            SELECT id,url,domain,title,summary,cleaned,created_at,relevance,quality,score
            FROM docs
            ORDER BY score DESC, id DESC
            """)
            return c.fetchall()

    def get_doc(self, doc_id: int):
        with self.lock:
            c = self.conn.cursor()
            c.execute("""
            SELECT id,url,domain,title,summary,cleaned,created_at,relevance,quality,score
            FROM docs WHERE id=?
            """, (int(doc_id),))
            r = c.fetchone()
        if not r:
            return None
        keys = ["id", "url", "domain", "title", "summary", "cleaned", "created_at", "relevance", "quality", "score"]
        return dict(zip(keys, r))

    def delete_doc(self, doc_id: int):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT url FROM docs WHERE id=?", (int(doc_id),))
            r = cur.fetchone()
            url = r[0] if r else None
            self.conn.execute("DELETE FROM docs WHERE id=?", (int(doc_id),))
            self.conn.commit()
        return url

    def delete_all_docs(self):
        with self.lock:
            self.conn.execute("DELETE FROM docs;")
            self.conn.commit()

    def upsert_related(self, term: str, score: float, source_url: str):
        term = (term or "").strip()
        if not term:
            return
        score = float(score)
        with self.lock:
            # scoreはMAXで保ちつつ、hitsを増やす
            self.conn.execute("""
            INSERT INTO related(term,score,hits,last_seen,source_url)
            VALUES(?,?,?,?,?)
            ON CONFLICT(term) DO UPDATE SET
              score=MAX(related.score, excluded.score),
              hits=related.hits + 1,
              last_seen=excluded.last_seen,
              source_url=excluded.source_url
            """, (term, score, 1, iso_utc(), source_url))
            self.conn.commit()

    def list_related(self, limit=200):
        with self.lock:
            c = self.conn.cursor()
            c.execute("""
            SELECT term,score,hits,last_seen,source_url
            FROM related
            ORDER BY score DESC, hits DESC
            LIMIT ?
            """, (int(limit),))
            return c.fetchall()

    def delete_related(self, term: str):
        term = (term or "").strip()
        if not term:
            return
        with self.lock:
            self.conn.execute("DELETE FROM related WHERE term=?", (term,))
            self.conn.commit()

    def delete_all_related(self):
        with self.lock:
            self.conn.execute("DELETE FROM related;")
            self.conn.commit()

    def iter_embeddings(self):
        with self.lock:
            c = self.conn.cursor()
            c.execute("SELECT id, emb FROM docs WHERE emb IS NOT NULL AND length(emb)>0")
            return c.fetchall()

# ==================== Embedding ====================
import numpy as np

JA = re.compile(r"[ぁ-んァ-ン一-龠々〆ヵヶー]{2,}")
EN = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{3,}")

STOP = set([
    "こと", "もの", "ため", "これ", "それ", "あれ", "ここ", "そこ", "この", "その", "あの",
    "です", "ます", "いる", "ある", "なる", "する", "よう", "など", "また", "そして",
    "ページ", "記事", "人気", "一覧", "検索", "結果", "関連", "続きを読む", "カテゴリ",
    "ログイン", "会員", "登録", "無料", "有料", "購入"
])

def tokenize(text: str):
    text = strip_ctrl(text)
    out = [t for t in JA.findall(text)]
    out += [t.lower() for t in EN.findall(text)]
    return out

class Embedder:
    def __init__(self, s, log):
        self.dim = int(s.get("embed_dim", 384))
        self.mode = "hash"
        self.model = None

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(s.get("embedder_name", DEFAULT["embedder_name"]))
            self.dim = int(self.model.get_sentence_embedding_dimension())
            self.mode = "st"
        except Exception:
            self.model = None

        log(f"Embedder: {self.mode} dim={self.dim}")

    def embed(self, texts):
        if self.model is not None:
            v = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(v, dtype=np.float32)

        arr = []
        for t in texts:
            v = np.zeros((self.dim,), dtype=np.float32)
            for w in tokenize(t):
                if w in STOP:
                    continue
                v[stable_hash32(w) % self.dim] += 1.0
            n = float(np.linalg.norm(v))
            if n > 0:
                v /= n
            arr.append(v)
        return np.vstack(arr) if arr else np.zeros((0, self.dim), dtype=np.float32)

def vec_blob(v):
    return np.asarray(v, dtype=np.float32).tobytes()

def blob_vec(b, dim):
    if not b:
        return np.zeros((dim,), dtype=np.float32)
    return np.frombuffer(b, dtype=np.float32, count=dim)

def cos(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# ==================== Web search / fetch / extract ====================
JUNK_HINTS = ["プライバシー", "利用規約", "Cookie", "クッキー", "広告", "免責", "ログイン", "会員登録",
              "シェア", "フォロー", "関連記事", "人気記事", "メニュー", "ナビ", "購読", "通知"]

def ddg_html_search(query, s, log):
    """DuckDuckGoのHTML版を直接取得してパースするフォールバック検索。
    ddgs側の仕様変更や一時的な壊れでKeyError等が出たときの保険。
    """
    import httpx
    from bs4 import BeautifulSoup

    q = (query or "").strip()
    if not q:
        return []

    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote_plus(q)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Zunkey_GPT_Search",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ja,en;q=0.8",
    }

    timeout = int(s.get("search_timeout_sec", 25))
    with httpx.Client(follow_redirects=True, timeout=timeout, headers=headers) as c:
        r = c.get(url)
        r.raise_for_status()
        html = r.text or ""

    soup = BeautifulSoup(html, "html.parser")

    out = []
    for a in soup.select("a.result__a"):
        href = a.get("href") or ""
        title = " ".join(list(a.stripped_strings))[:200]

        u = href
        if "uddg=" in href:
            mm = re.search(r"uddg=([^&]+)", href)
            if mm:
                try:
                    u = urllib.parse.unquote_plus(mm.group(1))
                except Exception:
                    u = href

        if not u.startswith("http"):
            continue

        snippet = ""
        try:
            cont = a.find_parent("div", class_=re.compile(r"result", re.I))
            if cont:
                sn = cont.select_one(".result__snippet") or cont.select_one(".result__content")
                if sn:
                    snippet = " ".join(list(sn.stripped_strings))[:400]
        except Exception:
            snippet = ""

        out.append({"url": u, "title": title or u, "snippet": snippet})
        if len(out) >= int(s.get("search_results", 8)):
            break

    log(f"DDG HTML fallback results={len(out)}")
    return out


def ddg_search(query, s, log):
    time.sleep(random.uniform(1.2, 3.4))

    res = []


    # Nuitka等で凍結（frozen）された実行環境では、ddgs側がブロック/仕様変更で落ちやすいことがある。
    # その場合はHTML版を直接パースする方式を優先する。
    if bool(s.get("force_html_search_when_frozen", True)) and is_frozen_app():
        try:
            return ddg_html_search(query, s, log)
        except Exception as e:
            log(f"DDG HTML fallback ERROR {type(e).__name__}: {e}")
            return []

    timeout = int(s.get("search_timeout_sec", 25))
    region = s.get("search_region", "jp-jp")
    safesearch = s.get("search_safesearch", "moderate")
    max_results = int(s.get("search_results", 8))

    # 1) ddgs（ライブラリ）を試す
    try:
        try:
            from ddgs import DDGS
        except Exception:
            from duckduckgo_search import DDGS

        with DDGS(timeout=timeout) as ddgs:
            try:
                it = ddgs.text(query, region=region, safesearch=safesearch, max_results=max_results)
            except TypeError:
                it = ddgs.text(keywords=query, region=region, safesearch=safesearch, max_results=max_results)

            for r in it:
                if not r:
                    continue
                url = r.get("href") or r.get("url") or ""
                if url:
                    res.append({
                        "url": url,
                        "title": r.get("title", "") or "",
                        "snippet": r.get("body", "") or r.get("snippet", "") or ""
                    })

    except KeyError as e:
        # ddgsが内部仕様変更で壊れた時に出がち
        log(f"DDG ERROR KeyError: {e} -> HTML fallback")
        try:
            res = ddg_html_search(query, s, log)
        except Exception as e2:
            log(f"DDG HTML fallback ERROR {type(e2).__name__}: {e2}")
            res = []

    except Exception as e:
        log(f"DDG ERROR {type(e).__name__}: {e} -> HTML fallback")
        try:
            res = ddg_html_search(query, s, log)
        except Exception as e2:
            log(f"DDG HTML fallback ERROR {type(e2).__name__}: {e2}")
            res = []

    return res
def fetch_html(url, timeout_sec, log):
    import httpx
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Zunkey_GPT_Search",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ja,en;q=0.8",
    }
    with httpx.Client(follow_redirects=True, timeout=timeout_sec, headers=headers) as c:
        r = c.get(url)
        r.raise_for_status()
        ct = (r.headers.get("content-type", "") or "").lower()
        if ct and ("text/html" not in ct and "application/xhtml+xml" not in ct and not ct.startswith("text/")):
            return "", ct
        r.encoding = r.encoding or "utf-8"
        return r.text, ct

def title_from_html(html):
    m = re.search(r"<title[^>]*>(.*?)</title>", html or "", flags=re.I | re.S)
    if not m:
        return ""
    t = re.sub(r"\s+", " ", m.group(1))
    t = re.sub(r"<[^>]+>", "", t)
    return t.strip()[:200]

def post_clean(text: str) -> str:
    text = norm_ws(strip_ctrl(text or ""))
    lines = [ln.strip() for ln in text.splitlines()]
    keep = []
    for ln in lines:
        if len(ln) < 10:
            continue
        if any(h in ln for h in JUNK_HINTS):
            continue
        keep.append(ln)
    return norm_ws("\n".join(keep))

def clean_text(html: str, url: str) -> str:
    html = strip_ctrl(html or "")

    # 1) trafilatura 優先
    try:
        import trafilatura
        try:
            import trafilatura.settings
            cfg = trafilatura.settings.use_config()
            cfg.set("DEFAULT", "include_comments", "False")
            cfg.set("DEFAULT", "include_tables", "False")
            cfg.set("DEFAULT", "include_links", "False")
            cfg.set("DEFAULT", "favor_precision", "True")
            t = trafilatura.extract(
                html,
                url=url,
                config=cfg,
                output_format="txt",
                include_comments=False,
                include_tables=False,
                include_links=False
            )
        except Exception:
            t = trafilatura.extract(html, url=url, output_format="txt")
        if t:
            return post_clean(t)
    except Exception:
        pass

    # 2) BeautifulSoup fallback
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "canvas", "iframe", "header", "footer", "nav", "aside", "form"]):
            tag.decompose()
        t = "\n".join(soup.stripped_strings)
        return post_clean(t)
    except Exception:
        return post_clean(html)

def extract_keywords(text: str, topic: str, limit=18):
    freq = {}
    for w in tokenize(text):
        if len(w) < 2 or len(w) > 32:
            continue
        if w in STOP:
            continue
        if topic and (w == topic or topic in w or w in topic):
            continue
        freq[w] = freq.get(w, 0) + 1

    items = list(freq.items())

    def sc(kv):
        k, c = kv
        lb = min(len(k), 24) / 24.0
        return c * 1.0 + lb * 1.4

    items.sort(key=sc, reverse=True)
    out = []
    for k, _ in items:
        if k not in out:
            out.append(k)
        if len(out) >= limit:
            break
    return out

def extractive_summary(text: str, max_chars: int) -> str:
    text = norm_ws(text)
    sents = re.split(r"(?<=[。！？\.\!\?])\s*", text)
    cands = [s.strip() for s in sents if len(s.strip()) >= 18]
    if not cands:
        return clamp(text, max_chars)
    scored = []
    for s in cands:
        u = len(set(tokenize(s)))
        scored.append((u * 1.2 + min(len(s), 200) / 70.0, s))
    scored.sort(reverse=True)
    out = []
    total = 0
    for _, s in scored:
        if total + len(s) + 1 > max_chars:
            continue
        out.append(s)
        total += len(s) + 1
        if total >= max_chars * 0.92:
            break
    return clamp("\n".join(out).strip(), max_chars)

def quality_heuristic(url: str, text_len: int) -> float:
    d = get_domain(url)
    sc = 50.0
    if any(d.endswith(x) for x in ["wikipedia.org", "github.com", "arxiv.org", "go.jp", "ac.jp"]):
        sc += 18
    if any(x in d for x in ["note.com", "qiita.com", "zenn.dev"]):
        sc += 6
    if text_len > 8000:
        sc += 10
    elif text_len > 3000:
        sc += 5
    if text_len < 1200:
        sc -= 12
    return float(max(0.0, min(100.0, sc)))

# ==================== Local LLM (llama.cpp) ====================
class LocalLLM:
    def __init__(self, s, log):
        self.s = s
        self.log = log
        self.llm = None
        self.lock = threading.Lock()
        self.downloading = threading.Event()

    def model_path_abs(self) -> str:
        return resolve_in_data(self.s.get("llm_model_path", ""))

    def ensure_model_download(self):
        # 起動時に必ず呼ぶ（要件）
        path = self.model_path_abs()
        url = (self.s.get("llm_model_url", "") or "").strip()

        os.makedirs(os.path.dirname(path), exist_ok=True)

        if os.path.exists(path) and os.path.getsize(path) > 1024 * 1024:
            self.log("LLM model already exists: " + path)
            return

        if not url:
            self.log("LLM model url is empty（自動DL不可）")
            return

        if self.downloading.is_set():
            return

        self.downloading.set()

        def _dl():
            try:
                import httpx
                self.log("LLM download start: " + url)
                tmp = path + ".part"
                with httpx.Client(follow_redirects=True, timeout=None, headers={"User-Agent": "Mozilla/5.0 Zunkey_GPT_Search"}) as c:
                    with c.stream("GET", url) as r:
                        r.raise_for_status()
                        total = int(r.headers.get("content-length") or 0)
                        got = 0
                        t0 = time.time()
                        with open(tmp, "wb") as f:
                            for ch in r.iter_bytes(chunk_size=1024 * 1024):
                                if not ch:
                                    continue
                                f.write(ch)
                                got += len(ch)
                                if total > 0:
                                    pct = got * 100.0 / total
                                    sec = max(0.001, time.time() - t0)
                                    sp = got / sec / (1024 * 1024)
                                    self.log(f"download {pct:.1f}% {got/1e9:.2f}GB/{total/1e9:.2f}GB {sp:.1f}MB/s")
                                else:
                                    self.log(f"download {got/1e9:.2f}GB")
                os.replace(tmp, path)
                self.log("LLM download complete: " + path + f" ({os.path.getsize(path)/1e9:.2f}GB)")
            except Exception as e:
                self.log(f"LLM download failed {type(e).__name__}: {e}")
            finally:
                self.downloading.clear()

        threading.Thread(target=_dl, daemon=True).start()

    def load(self):
        with self.lock:
            if self.llm is not None:
                return True

            path = self.model_path_abs()
            if not os.path.exists(path):
                self.log("LLMモデルが見つからない: " + path)
                return False

            try:
                from llama_cpp import Llama
            except Exception as e:
                if is_frozen_app():
                    self.log("EXEではllama-cpp-python未同梱の可能性: " + str(e))
                    return False
                pip_install(["llama-cpp-python>=0.3.0"], self.log)
                from llama_cpp import Llama

            self.log("LLM load: " + os.path.abspath(path))
            self.llm = Llama(
                model_path=path,
                n_ctx=int(self.s.get("llm_n_ctx", 4096)),
                n_threads=int(self.s.get("llm_n_threads", 8)),
                n_gpu_layers=int(self.s.get("llm_n_gpu_layers", -1)),
                verbose=False,
            )
            return True

    def _shrink_prompt(self, prompt: str) -> str:
        llm = self.llm
        if llm is None:
            return prompt

        max_ctx = int(self.s.get("llm_n_ctx", 4096))
        margin = 750

        try:
            toks = llm.tokenize(prompt.encode("utf-8"))
            if len(toks) <= max_ctx - margin:
                return prompt
        except Exception:
            if len(prompt) <= 8000:
                return prompt

        mark = "ページ本文:\n"
        if mark not in prompt:
            return prompt[:max(1400, int(len(prompt) * 0.6))]

        head, body = prompt.split(mark, 1)
        body = body.strip()

        for ratio in [0.7, 0.5, 0.35, 0.25]:
            nb = body[:max(900, int(len(body) * ratio))]
            p2 = head + mark + nb
            try:
                toks = llm.tokenize(p2.encode("utf-8"))
                if len(toks) <= max_ctx - margin:
                    return p2
            except Exception:
                if len(p2) <= 6000:
                    return p2

        return head + mark + body[:900]

    def summarize(self, topic, title, url, cleaned):
        if self.downloading.is_set():
            return None
        if not self.load():
            return None

        chunk = clamp(cleaned, int(self.s.get("chunk_chars_for_llm", 2800)))

        prompt = (
            "あなたは調査アシスタント。次のページ本文から要点を日本語で要約し、主キーワードとの関連度と質を評価する。\n"
            "出力はJSONのみ。必ず1行JSONで出力し、改行や余計な文章は出さない。\n"
            f"主キーワード: {topic}\n"
            f"ページタイトル: {clamp(title, 140)}\n"
            f"URL: {url}\n"
            "JSON: {\"summary_ja\":\"(最大1000文字)\",\"relevance\":0-100,\"quality\":0-100,"
            "\"keywords\":[\"(最大12個)\"],\"notes\":\"(短い補足)\"}\n"
            "ページ本文:\n" + chunk
        )

        prompt = self._shrink_prompt(prompt)

        t0 = time.time()
        buf = ""

        try:
            gen = self.llm(
                prompt,
                max_tokens=int(self.s.get("llm_max_tokens", 520)),
                temperature=float(self.s.get("llm_temperature", 0.2)),
                top_p=float(self.s.get("llm_top_p", 0.9)),
                repeat_penalty=float(self.s.get("llm_repeat_penalty", 1.1)),
                stream=True,
            )
        except Exception:
            return None

        for part in gen:
            if time.time() - t0 > int(self.s.get("llm_timeout_sec", 90)):
                self.log("LLM timeout -> fallback")
                break
            try:
                buf += part["choices"][0].get("text", "") or ""
            except Exception:
                pass
            obj = safe_json_obj(buf)
            if obj:
                return obj

        return None

# ==================== RAG memory (in-memory) ====================
class RagMem:
    def __init__(self, dim):
        self.dim = dim
        self.ids = []
        self.mat = np.zeros((0, dim), dtype=np.float32)
        self.lock = threading.Lock()

    def rebuild(self, db: DB):
        rows = db.iter_embeddings()
        ids = []
        vecs = []
        for doc_id, emb in rows:
            ids.append(int(doc_id))
            vecs.append(blob_vec(emb, self.dim))
        with self.lock:
            self.ids = ids
            self.mat = np.vstack(vecs).astype(np.float32) if vecs else np.zeros((0, self.dim), dtype=np.float32)

    def upsert(self, doc_id, vec):
        vec = np.asarray(vec, dtype=np.float32)
        with self.lock:
            if doc_id in self.ids:
                i = self.ids.index(doc_id)
                self.mat[i] = vec
            else:
                self.ids.append(doc_id)
                self.mat = np.vstack([self.mat, vec[None, :]]) if self.mat.size else vec[None, :]

    def remove(self, doc_id):
        with self.lock:
            if doc_id not in self.ids:
                return
            i = self.ids.index(doc_id)
            self.ids.pop(i)
            if self.mat.shape[0] <= 1:
                self.mat = np.zeros((0, self.dim), dtype=np.float32)
            else:
                self.mat = np.delete(self.mat, i, axis=0)

    def clear(self):
        with self.lock:
            self.ids = []
            self.mat = np.zeros((0, self.dim), dtype=np.float32)

    def search(self, qvec, topk=8):
        q = np.asarray(qvec, dtype=np.float32)
        with self.lock:
            if self.mat.shape[0] == 0:
                return []
            sims = self.mat @ q
            idx = np.argsort(-sims)[:int(topk)]
            return [(self.ids[int(i)], float(sims[int(i)])) for i in idx]

# ==================== Engine ====================
class Engine:
    def __init__(self, db, s, emb: Embedder, rag: RagMem, log, ui_event, set_status):
        self.db = db
        self.s = s
        self.emb = emb
        self.rag = rag
        self.log = log
        self.ui_event = ui_event
        self.set_status = set_status

        self.llm = LocalLLM(s, log)

        self.stop = threading.Event()
        self.th = None

        self.topic = ""
        self.topic_vec = None

        self.seen = set()

    def set_topic(self, t: str):
        self.topic = (t or "").strip()
        self.s["topic"] = self.topic
        save_settings(self.s)
        self.topic_vec = self.emb.embed([self.topic])[0]
        self.log("topic設定: " + self.topic)

    def start(self):
        if self.th and self.th.is_alive():
            return
        self.stop.clear()
        self.th = threading.Thread(target=self.loop, daemon=True)
        self.th.start()

    def halt(self):
        self.stop.set()

    def pick_query(self):
        if not bool(self.s.get("use_related", True)) or not bool(self.s.get("related_use_for_query", True)):
            return self.topic

        rels = self.db.list_related(limit=80)
        cands = []
        min_hits = int(self.s.get("related_min_hits", 2))
        min_score = float(self.s.get("related_min_score", 58.0))

        for term, score, hits, _, _ in rels:
            term = str(term).strip()
            if len(term) < 2 or len(term) > 24:
                continue
            try:
                score = float(score or 0)
                hits = int(hits or 0)
            except Exception:
                continue
            if hits < min_hits:
                continue
            if score < min_score:
                continue
            cands.append((score, hits, term))

        cands.sort(reverse=True)
        if cands and random.random() < float(self.s.get("related_pick_prob", 0.75)):
            best = cands[0][2]
            if best and best != self.topic:
                return f"{self.topic} {best}"
        return self.topic

    def loop(self):
        self.log("[start] preflight開始")
        ok = ensure_packages(self.log)
        if not ok:
            self.set_status("起動失敗: 依存パッケージ不足")
            return

        for _, _, _, _, _, url, _ in self.db.list_docs(limit=5000):
            if url:
                self.seen.add(norm_url(url))

        self.log("[start] preflight完了。調査開始")
        self.set_status("Research loop start")

        while not self.stop.is_set():
            try:
                q = self.pick_query()
                if not q:
                    self.set_status("topic未設定")
                    time.sleep(0.8)
                    continue

                self.set_status("検索: " + q)
                self.log("検索: " + q)

                res = ddg_search(q, self.s, self.log)
                self.log(f"DDG結果件数={len(res)}")

                done = 0
                for r in res:
                    if self.stop.is_set():
                        break
                    u = norm_url(r.get("url", ""))
                    if not u:
                        continue
                    if u in self.seen:
                        continue
                    d = get_domain(u)
                    avoid = [a.lower() for a in (self.s.get("avoid_domains", []) or []) if a]
                    avoid = [(a[4:] if a.startswith("www.") else a) for a in avoid]
                    if d and any((d == a) or d.endswith("." + a) for a in avoid):
                        continue

                    ok2 = self.process(u, r.get("title", "") or "")
                    self.seen.add(u)
                    if ok2:
                        self.ui_event()

                    done += 1
                    if done >= int(self.s.get("max_pages_per_tick", 2)):
                        break

                time.sleep(float(self.s.get("tick_sleep_sec", 2.0)))

            except Exception as e:
                self.log(f"loop error {type(e).__name__}: {e}")
                time.sleep(2.0)

        self.set_status("停止")
        self.log("停止")

    def process(self, url: str, title_hint: str):
        self.log("FETCH開始 " + url)
        try:
            html, _ = fetch_html(url, int(self.s.get("search_timeout_sec", 25)), self.log)
            if not html:
                return False

            title = title_from_html(html) or title_hint or url

            cleaned = clean_text(html, url)
            cleaned = clamp(cleaned, int(self.s.get("max_store_chars", 25000)))
            self.log(f"抽出文字数={len(cleaned)} title={clamp(title,120)}")

            if len(cleaned) < int(self.s.get("min_text_len", 900)):
                self.log("SKIP 抽出が短すぎる " + url)
                return False

            self.log("要約開始 " + url)

            sum_chars = int(self.s.get("summary_chars", 1000))
            summary = ""
            rel = 50.0
            qua = 50.0
            kws = []

            if bool(self.s.get("use_llm", True)):
                obj = self.llm.summarize(self.topic, title, url, cleaned)
                if obj:
                    summary = norm_ws(str(obj.get("summary_ja", "") or ""))[:sum_chars]
                    try:
                        rel = float(obj.get("relevance", 50) or 50)
                    except Exception:
                        rel = 50.0
                    try:
                        qua = float(obj.get("quality", 50) or 50)
                    except Exception:
                        qua = 50.0
                    kws = obj.get("keywords", []) if isinstance(obj.get("keywords", []), list) else []

            if not summary:
                summary = extractive_summary(cleaned, sum_chars)
                qua = quality_heuristic(url, len(cleaned))
                rel = 50.0

            emb_text = (title or "") + "\n" + (summary or "")
            svec = self.emb.embed([emb_text])[0]
            topic_sim = cos(self.topic_vec, svec) * 100.0 if self.topic_vec is not None else 0.0

            rel = max(0.0, min(100.0, rel * 0.6 + topic_sim * 0.4))
            qua = max(0.0, min(100.0, float(qua) if qua is not None else quality_heuristic(url, len(cleaned))))

            score = float(self.s.get("w_rel", 0.65)) * rel + float(self.s.get("w_qua", 0.35)) * qua
            score = max(0.0, min(100.0, score))

            did = self.db.upsert_doc(
                url=norm_url(url),
                domain=get_domain(url),
                title=title,
                summary=summary,
                cleaned=cleaned,
                created_at=iso_utc(),
                relevance=rel,
                quality=qua,
                score=score,
                emb=vec_blob(svec)
            )
            self.rag.upsert(did, svec)

            if bool(self.s.get("use_related", True)) and self.topic:
                kw_src = title + "\n" + cleaned[:int(self.s.get("max_kw_chars", 14000))]
                extracted = extract_keywords(kw_src, self.topic, limit=20)

                if kws:
                    for k in kws[:12]:
                        k = str(k).strip()
                        if k and k not in extracted:
                            extracted.insert(0, k)
                    extracted = extracted[:24]

                thr = float(self.s.get("rel_sim_threshold", 0.24))
                hash_mode = (getattr(self.emb, "mode", "hash") == "hash")
                for term in extracted:
                    if len(term) < 2 or len(term) > 24:
                        continue

                    if hash_mode:
                        # hash埋め込みは意味類似を表現できない（ほぼ完全一致しか拾えない）ので、
                        # topic-term 類似度フィルタは使わず、出現頻度ベースで関連ワードを採用する
                        cnt = kw_src.count(term)
                        if term in title:
                            cnt += 2
                        fsc = min(1.0, cnt / 8.0)  # 0..1
                        lb = min(1.0, len(term) / 10.0)
                        kval = 40.0 + 60.0 * (0.75 * fsc + 0.25 * lb)
                        tscore = score * 0.65 + kval * 0.35
                        self.db.upsert_related(term, float(tscore), url)
                        continue

                    tvec = self.emb.embed([term])[0]
                    tsim = cos(self.topic_vec, tvec)
                    if tsim < thr:
                        continue
                    tscore = score * 0.50 + tsim * 100.0 * 0.50
                    self.db.upsert_related(term, float(tscore), url)


            self.log(f"保存 score={score:.1f} rel={rel:.1f} qua={qua:.1f} url={url}")
            return True

        except Exception as e:
            self.log(f"ERROR url={url} type={type(e).__name__} msg={e}")
            return False

# ==================== UI ====================
def run():
    try:
        must_writable_dir(DATA_DIR)
    except Exception as e:
        sys.stderr.write("書き込み不可の場所です。フォルダをEドライブ等の書き込み可能な場所へ移動してください。\n")
        sys.stderr.write(str(e) + "\n")
        raise

    os.makedirs(GPTOSS_DIR, exist_ok=True)

    log_q = queue.Queue()
    ui_q = queue.Queue()

    def log(msg: str):
        line = f"[{now_hms()}] {msg}"
        log_q.put(line)
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    log(f"起動: {APP} {VER}")

    # 依存の版ズレを判定しやすいように、主要ライブラリのバージョンをログへ
    try:
        import httpx as _httpx
        log("httpx=" + getattr(_httpx, "__version__", "unknown"))
    except Exception:
        pass
    try:
        import ddgs as _ddgs
        log("ddgs=" + getattr(_ddgs, "__version__", "unknown"))
    except Exception:
        pass
    try:
        import bs4 as _bs4
        log("bs4=" + getattr(_bs4, "__version__", "unknown"))
    except Exception:
        pass

    log(f"runtime log: {os.path.abspath(LOG_PATH)}")
    log("data root: " + os.path.abspath(DATA_DIR))

    s = load_settings()
    # 起動時モデルDLはUIのインストール画面で実行（ブロック）

    db = DB(DB_PATH)
    emb = Embedder(s, log)

    # 埋め込みがhashの場合、関連ワードのcos類似しきい値が高いと拾いづらい。
    # settings.jsonがデフォルトのままなら、hash向けの値に自動調整する。
    try:
        if getattr(emb, "mode", "hash") == "hash":
            if float(s.get("rel_sim_threshold", DEFAULT["rel_sim_threshold"])) == float(DEFAULT["rel_sim_threshold"]):
                s["rel_sim_threshold"] = float(s.get("rel_sim_threshold_hash_default", DEFAULT.get("rel_sim_threshold_hash_default", 0.12)))
                save_settings(s)
                log(f"rel_sim_threshold(auto)= {s['rel_sim_threshold']:.2f}")
    except Exception:
        pass
    rag = RagMem(emb.dim)
    rag.rebuild(db)

    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog

    root = tk.Tk()
    root.title(f"{APP} {VER}")
    root.geometry("1180x780")

    status = tk.StringVar(value="Ready")

    def set_status(x):
        ui_q.put(("status", x))

    def refresh():
        ui_q.put(("refresh", None))
    engine = Engine(db, s, emb, rag, log, refresh, set_status)

    top = ttk.Frame(root, padding=8)
    top.pack(side=tk.TOP, fill=tk.X)

    ttk.Label(top, text="主キーワード").pack(side=tk.LEFT)
    topic_var = tk.StringVar(value=s.get("topic", ""))
    ttk.Entry(top, textvariable=topic_var, width=34).pack(side=tk.LEFT, padx=6)

    use_llm_var = tk.BooleanVar(value=bool(s.get("use_llm", True)))

    def toggle_llm():
        s["use_llm"] = bool(use_llm_var.get())
        save_settings(s)
        log("use_llm=" + str(s["use_llm"]))

    ttk.Checkbutton(top, text="LLM要約", variable=use_llm_var, command=toggle_llm).pack(side=tk.LEFT, padx=10)

    def set_topic():
        t = topic_var.get().strip()
        if not t:
            return messagebox.showinfo("info", "主キーワードが空です")
        engine.set_topic(t)
        refresh()

    ttk.Button(top, text="topic設定", command=set_topic).pack(side=tk.LEFT, padx=4)

    def start():

        if not preflight.get('ok', False):

            _show_installer_and_run()

            return

        if not engine.topic:
            t = topic_var.get().strip()
            if not t:
                return messagebox.showinfo("info", "主キーワードを入れてください")
            engine.set_topic(t)
        engine.start()
        log("Research start")

    def stop():
        engine.halt()
        log("Research stop requested")

    def export_all_text():
        rows = db.list_docs_full()
        if not rows:
            return messagebox.showinfo("info", "エクスポートする記事がありません")

        default_name = f"zunkey_gpt_search_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        path = filedialog.asksaveasfilename(
            title="エクスポート先を選択",
            defaultextension=".txt",
            initialfile=default_name,
            filetypes=[("Text", "*.txt"), ("All", "*.*")]
        )
        if not path:
            return

        sep = "=" * 80
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"{APP} export {VER}\n")
                f.write(f"topic: {s.get('topic','')}\n")
                f.write(f"exported_at: {iso_utc()}\n")
                f.write(sep + "\n\n")

                for (did, url, dom, title, summary, cleaned, created_at, relevance, quality, score) in rows:
                    f.write(sep + "\n")
                    f.write(f"id: {did}\n")
                    f.write(f"score: {float(score):.1f}  rel: {float(relevance):.1f}  qua: {float(quality):.1f}\n")
                    f.write(f"created_at: {created_at}\n")
                    f.write(f"domain: {dom}\n")
                    f.write(f"title: {title}\n")
                    f.write(f"url: {url}\n\n")

                    f.write("【summary】\n")
                    f.write((summary or "").strip() + "\n\n")

                    f.write("【cleaned】\n")
                    f.write((cleaned or "").strip() + "\n\n")

            log("エクスポート完了: " + os.path.abspath(path))
            messagebox.showinfo("info", "エクスポート完了")
        except Exception as e:
            log(f"エクスポート失敗 {type(e).__name__}: {e}")
            messagebox.showerror("error", f"エクスポートに失敗しました\n{e}")

    start_btn = ttk.Button(top, text="開始", command=start)
    start_btn.pack(side=tk.LEFT, padx=4)
    stop_btn = ttk.Button(top, text="停止", command=stop)
    stop_btn.pack(side=tk.LEFT, padx=4)
    export_btn = ttk.Button(top, text="全記事エクスポート", command=export_all_text)
    export_btn.pack(side=tk.LEFT, padx=(12,4))
    ttk.Label(top, textvariable=status).pack(side=tk.RIGHT)
    # ---------- startup installer (pip + GPT-OSS) ----------
    preflight = {"done": False, "ok": False, "running": False}

    def _show_installer_and_run():
        if preflight["running"] or preflight["done"]:
            return

        preflight["running"] = True
        try:
            start_btn.state(["disabled"])
        except Exception:
            pass

        dlg = tk.Toplevel(root)
        dlg.title("インストール")
        dlg.geometry("560x260")
        dlg.resizable(False, False)
        dlg.transient(root)
        dlg.grab_set()

        # 途中で閉じられると中途半端になるので基本は無効化
        dlg.protocol("WM_DELETE_WINDOW", lambda: None)

        msg_var = tk.StringVar(value="準備中...")
        ttk.Label(dlg, textvariable=msg_var, wraplength=540, justify="left").pack(anchor="w", padx=12, pady=(12, 6))

        pbar = ttk.Progressbar(dlg, orient="horizontal", mode="determinate", maximum=100)
        pbar.pack(fill=tk.X, padx=12, pady=(0, 8))

        logtxt = tk.Text(dlg, height=7, wrap=tk.WORD)
        logtxt.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))
        logtxt.configure(state="disabled")

        btn_row = ttk.Frame(dlg)
        btn_row.pack(fill=tk.X, padx=12, pady=(0, 10))
        close_btn = ttk.Button(btn_row, text="閉じる", command=lambda: (dlg.grab_release(), dlg.destroy()))
        close_btn.pack(side=tk.RIGHT)
        close_btn.state(["disabled"])

        qinst = queue.Queue()

        def inst_log(line: str):
            try:
                qinst.put(("log", line))
            except Exception:
                pass

        def _download_file(url: str, dst: str):
            import httpx
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            tmp = dst + ".part"

            exist = 0
            if os.path.exists(tmp):
                try:
                    exist = os.path.getsize(tmp)
                except Exception:
                    exist = 0

            headers = {"User-Agent": "Mozilla/5.0 Zunkey_GPT_Search"}
            if exist > 0:
                headers["Range"] = f"bytes={exist}-"

            t0 = time.time()
            last = 0.0
            got = exist

            with httpx.Client(follow_redirects=True, timeout=None, headers=headers) as c:
                with c.stream("GET", url) as r:
                    r.raise_for_status()

                    total = 0
                    cr = r.headers.get("content-range") or r.headers.get("Content-Range") or ""
                    if "/" in cr:
                        try:
                            total = int(cr.split("/")[-1])
                        except Exception:
                            total = 0
                    if total <= 0:
                        try:
                            total = int(r.headers.get("content-length") or 0) + exist
                        except Exception:
                            total = 0

                    mode = "ab" if exist > 0 else "wb"
                    with open(tmp, mode) as f:
                        for ch in r.iter_bytes(chunk_size=1024 * 1024):
                            if not ch:
                                continue
                            f.write(ch)
                            got += len(ch)

                            now = time.time()
                            if now - last < 0.15:
                                continue
                            last = now

                            if total > 0:
                                pct = max(0.0, min(100.0, got * 100.0 / total))
                                sec = max(0.001, now - t0)
                                sp = got / sec / (1024 * 1024)
                                qinst.put(("progress", pct))
                                qinst.put(("msg", f"GPT-OSS ダウンロード中... {pct:.1f}%  {got/1e9:.2f}GB/{total/1e9:.2f}GB  {sp:.1f}MB/s"))
                            else:
                                qinst.put(("progress", 0.0))
                                qinst.put(("msg", f"GPT-OSS ダウンロード中... {got/1e9:.2f}GB"))

            os.replace(tmp, dst)

        def worker():
            try:
                qinst.put(("msg", "必須パッケージ確認中..."))
                ok = ensure_packages(inst_log)
                if not ok:
                    qinst.put(("error", "必須パッケージが不足しています（EXEは同梱が必要です）。"))
                    return

                qinst.put(("msg", "GPT-OSS モデル確認中..."))
                model_path = resolve_in_data(s.get("llm_model_path", ""))
                model_url = (s.get("llm_model_url", "") or "").strip()

                if not model_path:
                    qinst.put(("error", "llm_model_path が空です。"))
                    return

                os.makedirs(os.path.dirname(model_path), exist_ok=True)

                if os.path.exists(model_path) and os.path.getsize(model_path) > 1024 * 1024:
                    qinst.put(("progress", 100.0))
                    qinst.put(("msg", "GPT-OSS 既に存在します。"))
                else:
                    if not model_url:
                        qinst.put(("error", "llm_model_url が空のため自動ダウンロードできません。"))
                        return
                    qinst.put(("progress", 0.0))
                    qinst.put(("log", f"download start: {model_path}"))
                    _download_file(model_url, model_path)
                    qinst.put(("progress", 100.0))
                    qinst.put(("msg", "GPT-OSS ダウンロード完了。"))

                qinst.put(("done", None))
            except Exception as e:
                qinst.put(("error", f"{type(e).__name__}: {e}"))

        threading.Thread(target=worker, daemon=True).start()

        def tick():
            try:
                while True:
                    kind, payload = qinst.get_nowait()
                    if kind == "msg":
                        msg_var.set(str(payload))
                    elif kind == "progress":
                        try:
                            pbar.configure(value=float(payload))
                        except Exception:
                            pass
                    elif kind == "log":
                        logtxt.configure(state="normal")
                        logtxt.insert("end", str(payload) + "\n")
                        logtxt.see("end")
                        logtxt.configure(state="disabled")
                    elif kind == "error":
                        msg_var.set("失敗: " + str(payload))
                        logtxt.configure(state="normal")
                        logtxt.insert("end", "ERROR: " + str(payload) + "\n")
                        logtxt.see("end")
                        logtxt.configure(state="disabled")
                        close_btn.state(["!disabled"])
                        preflight["done"] = True
                        preflight["ok"] = False
                        preflight["running"] = False
                        return
                    elif kind == "done":
                        preflight["done"] = True
                        preflight["ok"] = True
                        preflight["running"] = False
                        try:
                            start_btn.state(["!disabled"])
                        except Exception:
                            pass
                        try:
                            dlg.protocol("WM_DELETE_WINDOW", lambda: (dlg.grab_release(), dlg.destroy()))
                        except Exception:
                            pass
                        try:
                            dlg.grab_release()
                        except Exception:
                            pass
                        dlg.destroy()
                        return
            except queue.Empty:
                pass

            root.after(120, tick)

        tick()

    # 起動直後に必ずインストール画面（モデルDL含む）を走らせる
    try:
        start_btn.state(["disabled"])
    except Exception:
        pass
    root.after(150, _show_installer_and_run)


    pan = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
    pan.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    left = ttk.Frame(pan, padding=6)
    right = ttk.Frame(pan, padding=6)
    pan.add(left, weight=1)
    pan.add(right, weight=2)

    ttk.Label(left, text="収集ドキュメント（score順）").pack(anchor="w")
    docs = tk.Listbox(left, height=16, selectmode=tk.EXTENDED, exportselection=False)
    docs.pack(fill=tk.BOTH, expand=False)

    # Doc actions (under docs list)
    doc_act = ttk.Frame(left)
    doc_act.pack(fill=tk.X, pady=(4, 0))

    ttk.Label(left, text="関連ワード（ランキング）").pack(anchor="w", pady=(10, 0))
    rel = tk.Listbox(left, height=12, selectmode=tk.EXTENDED, exportselection=False)
    rel.pack(fill=tk.BOTH, expand=True)

    # 関連語採用の厳しさ（関連ワードの直下に配置）
    rel_thr_row = ttk.Frame(left)
    rel_thr_row.pack(fill=tk.X, pady=(4, 6))

    rel_thr_var = tk.DoubleVar(value=float(s.get("rel_sim_threshold", DEFAULT.get("rel_sim_threshold", 0.18))))
    ttk.Label(rel_thr_row, text="関連語採用の厳しさ").pack(side=tk.LEFT)

    rel_thr_val = tk.StringVar(value=f"{float(rel_thr_var.get()):.2f}")
    ttk.Label(rel_thr_row, textvariable=rel_thr_val, width=4).pack(side=tk.LEFT, padx=(6, 6))

    def _rel_thr_changed(v):
        try:
            fv = float(v)
        except Exception:
            return
        fv = max(0.05, min(0.35, round(fv, 2)))
        rel_thr_val.set(f"{fv:.2f}")

    def _save_rel_thr_event(_evt=None):
        try:
            fv = float(rel_thr_var.get())
            fv = max(0.05, min(0.35, round(fv, 2)))
            rel_thr_var.set(fv)
            rel_thr_val.set(f"{fv:.2f}")
            s["rel_sim_threshold"] = fv
            save_settings(s)
        except Exception:
            pass

    rel_thr = tk.Scale(
        rel_thr_row,
        from_=0.05,
        to=0.35,
        resolution=0.01,
        orient="horizontal",
        length=220,
        showvalue=False,
        variable=rel_thr_var,
        command=_rel_thr_changed,
    )
    rel_thr.pack(side=tk.LEFT, padx=(0, 0), fill=tk.X, expand=True)

    rel_thr.bind("<ButtonRelease-1>", _save_rel_thr_event)
    rel_thr.bind("<KeyRelease-Left>", _save_rel_thr_event)
    rel_thr.bind("<KeyRelease-Right>", _save_rel_thr_event)

    act = ttk.Frame(left)
    act.pack(fill=tk.X, pady=(4, 0))

    doc_map = {}
    rel_map = {}
    rag_map = {}

    def get_selected_doc_id():
        sel = docs.curselection()
        if not sel:
            return None
        return doc_map.get(sel[0])

    def get_selected_term():
        sel = rel.curselection()
        if not sel:
            return None
        return rel_map.get(sel[0])

    def get_selected_doc_ids():
        sels = docs.curselection()
        out = []
        seen = set()
        for i in sels:
            did = doc_map.get(i)
            if did is None:
                continue
            try:
                did_i = int(did)
            except Exception:
                continue
            if did_i in seen:
                continue
            seen.add(did_i)
            out.append(did_i)
        return out

    def get_selected_terms():
        sels = rel.curselection()
        out = []
        seen = set()
        for i in sels:
            term = rel_map.get(i)
            if not term:
                continue
            if term in seen:
                continue
            seen.add(term)
            out.append(term)
        return out


    def delete_doc_selected():
        ids = get_selected_doc_ids()
        if not ids:
            return messagebox.showinfo("info", "記事が選択されていません")

        msg = "選択した記事を削除しますか" if len(ids) == 1 else f"選択した{len(ids)}件の記事を削除しますか"
        if not messagebox.askyesno("confirm", msg):
            return

        for did in ids:
            url = None
            try:
                url = db.delete_doc(int(did))
            except Exception:
                pass
            try:
                rag.remove(int(did))
            except Exception:
                pass
            if url:
                log(f"記事削除 id={did} url={url}")
            else:
                log(f"記事削除 id={did}")
        refresh()


    def delete_doc_all():
        if not messagebox.askyesno("confirm", "全記事を削除しますか（取り消し不可）"):
            return
        db.delete_all_docs()
        rag.clear()
        engine.seen.clear()
        log("全記事削除")
        refresh()

    def delete_kw_selected():
        terms = get_selected_terms()
        if not terms:
            return messagebox.showinfo("info", "キーワードが選択されていません")

        if len(terms) == 1:
            msg = f"キーワード「{terms[0]}」を削除しますか"
        else:
            msg = f"選択した{len(terms)}個のキーワードを削除しますか"
        if not messagebox.askyesno("confirm", msg):
            return

        for term in terms:
            try:
                db.delete_related(term)
            except Exception:
                pass
            log("キーワード削除: " + str(term))
        refresh()


    def delete_kw_all():
        if not messagebox.askyesno("confirm", "全キーワードを削除しますか（取り消し不可）"):
            return
        db.delete_all_related()
        log("全キーワード削除")
        refresh()

    
    import tkinter as tk
    from tkinter import ttk
    ttk.Button(doc_act, text="記事削除", command=delete_doc_selected).pack(side="left", padx=4)
    ttk.Button(doc_act, text="全記事削除", command=delete_doc_all).pack(side="left", padx=4)
    ttk.Button(act, text="キーワード削除", command=delete_kw_selected).pack(side="left", padx=4)
    ttk.Button(act, text="全キーワード削除", command=delete_kw_all).pack(side="left", padx=4)

        # Right panel: RAG search
    rt = ttk.Frame(right)
    rt.pack(fill=tk.X)

    ttk.Label(rt, text="RAG検索").pack(side=tk.LEFT)

    qv = tk.StringVar()
    rag_entry = ttk.Entry(rt, textvariable=qv, width=44)
    rag_entry.pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)

    # RAGしきい値（厳しさ）

    rag_thr_row = ttk.Frame(right)
    rag_thr_row.pack(fill=tk.X, pady=(2, 2))

    rag_thr_var = tk.DoubleVar(value=float(s.get("rag_min_sim", 0.22)))
    ttk.Label(rag_thr_row, text="RAG一致の厳しさ").pack(side=tk.LEFT)

    rag_thr_val = tk.StringVar(value=f"{float(rag_thr_var.get()):.2f}")
    ttk.Label(rag_thr_row, textvariable=rag_thr_val, width=4).pack(side=tk.LEFT, padx=(6, 6))

    def _rag_thr_changed(v):
        try:
            fv = float(v)
        except Exception:
            return
        fv = max(0.05, min(0.40, round(fv, 2)))
        rag_thr_val.set(f"{fv:.2f}")

    def _save_rag_thr_event(_evt=None):
        try:
            fv = float(rag_thr_var.get())
            fv = max(0.05, min(0.40, round(fv, 2)))
            rag_thr_var.set(fv)
            rag_thr_val.set(f"{fv:.2f}")
            s["rag_min_sim"] = fv
            save_settings(s)
        except Exception:
            pass

    rag_thr = tk.Scale(
        rag_thr_row,
        from_=0.05,
        to=0.40,
        resolution=0.01,
        orient="horizontal",
        length=240,
        showvalue=False,
        variable=rag_thr_var,
        command=_rag_thr_changed,
    )
    rag_thr.pack(side=tk.LEFT, padx=(0, 8), fill=tk.X, expand=True)

    rag_thr.bind("<ButtonRelease-1>", _save_rag_thr_event)
    rag_thr.bind("<KeyRelease-Left>", _save_rag_thr_event)
    rag_thr.bind("<KeyRelease-Right>", _save_rag_thr_event)

    ttk.Label(right, text="RAG結果").pack(anchor="w", pady=(6, 0))

    rag_list = tk.Listbox(right, height=6)
    rag_list.pack(fill=tk.X, pady=(0, 6))

    def do_rag():
        q = qv.get().strip()
        if not q:
            return
        qvec = emb.embed([q])[0]
        hits = rag.search(qvec, topk=int(s.get("rag_topk", 8)))

        rag_list.delete(0, tk.END)
        rag_map.clear()

        min_sim = float(s.get("rag_min_sim", 0.24))
        i = 0
        for did, sim in hits:
            if sim < min_sim:
                continue
            doc = db.get_doc(did)
            if not doc:
                continue
            rag_list.insert(tk.END, f"{sim*100:.1f} | {clamp(doc.get('title',''),70)}")
            rag_map[i] = did
            i += 1

        if i == 0:
            rag_list.insert(tk.END, "該当なし（しきい値で除外）")

    ttk.Button(rt, text="検索", command=do_rag).pack(side=tk.LEFT, padx=(6, 0))
    rag_entry.bind("<Return>", lambda _e: do_rag())

    title_var = tk.StringVar(value="")
    ttk.Label(right, textvariable=title_var, font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(8, 0))

    meta_var = tk.StringVar(value="")
    ttk.Label(right, textvariable=meta_var).pack(anchor="w")

    btns = ttk.Frame(right)
    btns.pack(fill=tk.X, pady=4)

    cur_doc = {"doc": None}

    def open_url():
        if cur_doc["doc"]:
            u = cur_doc["doc"].get("url", "")
            if u:
                webbrowser.open(u)

    ttk.Button(btns, text="URLを開く", command=open_url).pack(side=tk.LEFT, padx=4)

    # Doc display
    tf = ttk.Frame(right)
    tf.pack(fill=tk.BOTH, expand=True)

    txt = tk.Text(tf, wrap=tk.WORD)
    txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    sc = ttk.Scrollbar(tf, orient="vertical", command=txt.yview)
    sc.pack(side=tk.RIGHT, fill=tk.Y)
    txt.configure(yscrollcommand=sc.set)

    ttk.Label(right, text="ログ").pack(anchor="w", pady=(6, 0))

    logbox = tk.Text(right, height=10, wrap=tk.NONE)
    logbox.pack(fill=tk.BOTH, expand=False)

    def refresh_lists():
        doc_map.clear()
        docs.delete(0, tk.END)
        for i, (did, score, relv, quav, title, url, created) in enumerate(db.list_docs(limit=300)):
            docs.insert(tk.END, f"{float(score):5.1f} r{float(relv):4.1f} q{float(quav):4.1f} | {clamp(title,60)}")
            doc_map[i] = int(did)

        rel_map.clear()
        rel.delete(0, tk.END)
        for i, (term, score, hits, last_seen, src) in enumerate(db.list_related(limit=int(s.get("related_limit", 200)))):
            rel.insert(tk.END, f"{float(score):5.1f} h{int(hits):3d} | {term}")
            rel_map[i] = str(term)

    def show_doc(did: int):
        d = db.get_doc(did)
        cur_doc["doc"] = d
        if not d:
            return
        title_var.set(d.get("title", ""))
        meta_var.set(
            f"{float(d.get('score',0)):.1f} | rel={float(d.get('relevance',0)):.1f} qua={float(d.get('quality',0)):.1f} | "
            f"{d.get('domain','')} | {d.get('created_at','')}"
        )

        txt.delete("1.0", "end")
        txt.insert("end", (d.get("summary", "") or "").strip() + "\n\n")
        txt.insert("end", "---- cleaned text（先頭） ----\n")
        txt.insert("end", clamp(d.get("cleaned", "") or "", 4200))

    def on_docs(_=None):
        sel = docs.curselection()
        if not sel:
            return
        did = doc_map.get(sel[0])
        if did:
            show_doc(did)

    docs.bind("<<ListboxSelect>>", on_docs)
    def _select_all_docs(_=None):
        docs.select_set(0, tk.END)
        return "break"

    def _select_all_terms(_=None):
        rel.select_set(0, tk.END)
        return "break"

    docs.bind("<Delete>", lambda _e: delete_doc_selected())
    rel.bind("<Delete>", lambda _e: delete_kw_selected())
    docs.bind("<Control-a>", _select_all_docs)
    docs.bind("<Control-A>", _select_all_docs)
    rel.bind("<Control-a>", _select_all_terms)
    rel.bind("<Control-A>", _select_all_terms)


    def on_rag(_=None):
        sel = rag_list.curselection()
        if not sel:
            return
        did = rag_map.get(sel[0])
        if did:
            show_doc(did)

    rag_list.bind("<<ListboxSelect>>", on_rag)

    def ui_tick():
        try:
            while True:
                ev, payload = ui_q.get_nowait()
                if ev == "status":
                    status.set(str(payload))
                elif ev == "refresh":
                    refresh_lists()
        except queue.Empty:
            pass

        try:
            while True:
                line = log_q.get_nowait()
                logbox.insert("end", line + "\n")
                logbox.see("end")
        except queue.Empty:
            pass

        root.after(250, ui_tick)

    refresh_lists()
    ui_tick()
    root.mainloop()

if __name__ == "__main__":
    run()
