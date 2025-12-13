# Zunkey_GPT_Search.py
# Zunkey_GPT_Search 1.0.0
# Tkinter + DDG検索 + HTML抽出 + ローカルLLM要約(任意) + RAG + ローカル検索DB
# 追加: 記事削除 / キーワード削除 / 全記事テキストエクスポート
# 改善: RAG検索に閾値(min_sim)導入 / hash埋め込みを安定ハッシュへ（起動ごとに変わらない）

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
VER = "1.0.0"

DB_PATH = "zunkey_gpt_search.sqlite3"
SETTINGS_PATH = "settings.json"
LOG_PATH = "zunkey_gpt_search_runtime.log"

DEFAULT = dict(
    topic="",

    # LLM要約（llama.cpp）
    use_llm=True,
    llm_model_path=r"models\gpt-oss\gpt-oss-20b-Q5_K_M.gguf",
    llm_model_url="",  # optional（直リンクを入れたら自動DLする）
    llm_n_ctx=4096,
    llm_n_threads=max(4, os.cpu_count() or 8),
    llm_n_gpu_layers=-1,   # -1: 可能な限りGPUへ
    llm_temperature=0.2,
    llm_top_p=0.9,
    llm_repeat_penalty=1.1,
    llm_timeout_sec=75,
    llm_max_tokens=520,

    # Embedding（sentence-transformersがあれば使う。無ければ安定hash）
    embed_dim=384,
    embedder_name="sentence-transformers/all-MiniLM-L6-v2",

    # Search
    search_region="jp-jp",
    search_safesearch="moderate",
    search_results=8,
    fetch_timeout_sec=25,
    max_pages_per_tick=2,
    tick_sleep_sec=2.0,

    # Page processing
    min_text_len=900,
    max_store_chars=25000,
    max_kw_chars=14000,

    # LLMへ渡す本文の事前カット（ctx対策）
    chunk_chars_for_llm=2800,
    summary_chars=1000,

    # Related terms scoring
    rel_sim_threshold=0.22,  # topicとのcos類似（0-1）
    w_rel=0.65,
    w_qua=0.35,

    # RAG
    rag_topk=8,
    rag_min_sim=0.28,  # 0.25〜0.35くらいで調整
    rag_embed_title_summary=True,  # True: title+summary を埋め込みに使う

    # Avoid domains（固定の除外）
    avoid_domains=["x.com", "twitter.com", "facebook.com", "bsky.app"],
)

# ---------------- utils ----------------
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

def norm_url(u: str) -> str:
    try:
        p = urllib.parse.urlsplit(u)
        p = p._replace(fragment="")
        return p.geturl()
    except Exception:
        return u

def get_domain(u: str) -> str:
    try:
        return urllib.parse.urlparse(u).netloc.lower()
    except Exception:
        return ""

def domain_is_avoided(domain: str, avoid_list) -> bool:
    d = (domain or "").lower().strip()
    if not d:
        return False
    for a in (avoid_list or []):
        a = (a or "").lower().strip()
        if not a:
            continue
        if d == a or d.endswith("." + a):
            return True
    return False

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

def pip_install(pkgs, log):
    if not pkgs:
        return
    log(f"Installing missing packages: {pkgs}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", *pkgs])

def ensure_packages(log):
    miss = []
    def has(mod):
        try:
            __import__(mod)
            return True
        except Exception:
            return False

    if not has("httpx"):
        miss.append("httpx>=0.28.1")
    if not has("ddgs"):
        miss.append("ddgs>=9.0.0")
    if not has("bs4"):
        miss.append("beautifulsoup4>=4.12.0")
    if not has("lxml"):
        miss.append("lxml>=4.9.4")
    if not has("trafilatura"):
        miss.append("trafilatura>=1.9.0")
    if not has("numpy"):
        miss.append("numpy>=1.24.0")

    if miss:
        pip_install(miss, log)

def load_settings():
    s = dict(DEFAULT)
    if os.path.exists(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                j = json.load(f)
            if isinstance(j, dict):
                s.update(j)
        except Exception:
            pass
    return s

def save_settings(s):
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# ---------------- DB ----------------
class Store:
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
        );""")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS related(
            term TEXT PRIMARY KEY,
            score REAL,
            last_seen TEXT,
            source_url TEXT
        );""")

        self.conn.commit()

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
        with self.lock:
            self.conn.execute("""
            INSERT INTO related(term,score,last_seen,source_url)
            VALUES(?,?,?,?)
            ON CONFLICT(term) DO UPDATE SET
              score=MAX(related.score, excluded.score),
              last_seen=excluded.last_seen,
              source_url=excluded.source_url
            """, (term, float(score), iso_utc(), source_url))
            self.conn.commit()

    def list_related(self, limit=200):
        with self.lock:
            c = self.conn.cursor()
            c.execute("""
            SELECT term,score,last_seen,source_url
            FROM related
            ORDER BY score DESC
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

# ---------------- Embedding ----------------
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

def stable_hash32(s: str) -> int:
    b = (s or "").encode("utf-8", "ignore")
    d = hashlib.md5(b).digest()
    return int.from_bytes(d[:4], "little", signed=False)

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
                idx = stable_hash32(w) % self.dim
                v[idx] += 1.0
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
    # サイズ不整合は捨てる（embedderのdim変更や旧DB対策）
    if len(b) < dim * 4:
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

# ---------------- Web search / fetch / extract ----------------
JUNK_HINTS = [
    "プライバシー", "利用規約", "Cookie", "クッキー", "広告", "免責", "ログイン", "会員登録",
    "シェア", "フォロー", "関連記事", "人気記事", "メニュー", "ナビ", "購読", "通知"
]

def ddg_search(query, s, log):
    time.sleep(random.uniform(1.2, 3.4))
    try:
        from ddgs import DDGS
    except Exception:
        from duckduckgo_search import DDGS

    res = []
    try:
        with DDGS(timeout=int(s.get("fetch_timeout_sec", 25))) as ddgs:
            for r in ddgs.text(
                query,
                region=s.get("search_region", "jp-jp"),
                safesearch=s.get("search_safesearch", "moderate"),
                max_results=int(s.get("search_results", 8))
            ):
                if not r:
                    continue
                url = r.get("href") or r.get("url") or ""
                if not url:
                    continue
                res.append({
                    "url": url,
                    "title": r.get("title", "") or "",
                    "snippet": r.get("body", "") or r.get("snippet", "") or ""
                })
    except Exception as e:
        log(f"DDG ERROR {type(e).__name__}: {e}")
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
        soup = BeautifulSoup(html, "lxml")
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

# ---------------- Local LLM (llama.cpp) ----------------
class LocalLLM:
    def __init__(self, s, log):
        self.s = s
        self.log = log
        self.llm = None
        self.lock = threading.Lock()

    def _download(self, url, dst):
        import httpx
        self.log("LLM download: " + url)
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        tmp = dst + ".part"
        with httpx.Client(follow_redirects=True, timeout=None, headers={"User-Agent": "Mozilla/5.0 Zunkey_GPT_Search"}) as c:
            with c.stream("GET", url) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length") or 0)
                got = 0
                with open(tmp, "wb") as f:
                    for ch in r.iter_bytes(chunk_size=1024 * 1024):
                        if not ch:
                            continue
                        f.write(ch)
                        got += len(ch)
                        if total > 0:
                            self.log(f"download {got*100/total:.1f}% {got/1e9:.2f}GB/{total/1e9:.2f}GB")
                        else:
                            self.log(f"download {got/1e9:.2f}GB")
        os.replace(tmp, dst)
        self.log("LLM download complete: " + dst)

    def load(self):
        with self.lock:
            if self.llm is not None:
                return True

            path = (self.s.get("llm_model_path", "") or "").strip()
            if not path:
                self.log("LLM model path is empty")
                return False

            if not os.path.exists(path):
                url = (self.s.get("llm_model_url", "") or "").strip()
                if url:
                    self._download(url, path)

            if not os.path.exists(path):
                self.log("LLMモデルが見つからない: " + os.path.abspath(path))
                return False

            try:
                from llama_cpp import Llama
            except Exception:
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
        if not self.load():
            raise RuntimeError("LLM not ready")

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

        gen = self.llm(
            prompt,
            max_tokens=int(self.s.get("llm_max_tokens", 520)),
            temperature=float(self.s.get("llm_temperature", 0.2)),
            top_p=float(self.s.get("llm_top_p", 0.9)),
            repeat_penalty=float(self.s.get("llm_repeat_penalty", 1.1)),
            stream=True,
        )

        for part in gen:
            if time.time() - t0 > int(self.s.get("llm_timeout_sec", 75)):
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

# ---------------- RAG index (in-memory) ----------------
class RagIndex:
    def __init__(self, dim):
        self.dim = dim
        self.ids = []
        self.mat = np.zeros((0, dim), dtype=np.float32)
        self.lock = threading.Lock()

    def rebuild(self, store: Store):
        rows = store.iter_embeddings()
        ids = []
        vecs = []
        for doc_id, emb in rows:
            v = blob_vec(emb, self.dim)
            if v is None or v.shape[0] != self.dim:
                continue
            # 念のため正規化
            n = float(np.linalg.norm(v))
            if n > 0:
                v = v / n
            ids.append(int(doc_id))
            vecs.append(v)
        with self.lock:
            self.ids = ids
            self.mat = np.vstack(vecs).astype(np.float32) if vecs else np.zeros((0, self.dim), dtype=np.float32)

    def upsert(self, doc_id, vec):
        vec = np.asarray(vec, dtype=np.float32)
        n = float(np.linalg.norm(vec))
        if n > 0:
            vec = vec / n
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

    def search(self, qvec, topk=8, min_sim=0.28):
        q = np.asarray(qvec, dtype=np.float32)
        nq = float(np.linalg.norm(q))
        if nq > 0:
            q = q / nq
        with self.lock:
            if self.mat.shape[0] == 0:
                return []
            sims = self.mat @ q
            order = np.argsort(-sims)

            out = []
            for i in order:
                sim = float(sims[int(i)])
                if sim < float(min_sim):
                    break
                out.append((self.ids[int(i)], sim))
                if len(out) >= int(topk):
                    break
            return out

# ---------------- Engine ----------------
class SearchEngine:
    def __init__(self, store, s, emb: Embedder, rag: RagIndex, log, ui_event, set_status):
        self.store = store
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
        rels = self.store.list_related(limit=80)
        cands = []
        for term, score, _, _ in rels:
            term = str(term).strip()
            if len(term) < 2 or len(term) > 24:
                continue
            cands.append((float(score), term))
        cands.sort(reverse=True)

        if cands and random.random() < 0.75:
            best = cands[0][1]
            if best and best != self.topic:
                return f"{self.topic} {best}"
        return self.topic

    def loop(self):
        self.log("[start] preflight開始")
        ensure_packages(self.log)

        # 既知URL
        for _, _, _, _, _, url, _ in self.store.list_docs(limit=5000):
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
                    u = norm_url(r.get("url", "") or "")
                    if not u:
                        continue
                    if u in self.seen:
                        continue

                    d = get_domain(u)
                    if domain_is_avoided(d, self.s.get("avoid_domains", [])):
                        self.seen.add(u)
                        continue

                    ok = self.process(u, r.get("title", "") or "")
                    self.seen.add(u)
                    if ok:
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
            html, _ = fetch_html(url, int(self.s.get("fetch_timeout_sec", 25)), self.log)
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
                try:
                    obj = self.llm.summarize(self.topic, title, url, cleaned)
                    if obj:
                        summary = norm_ws(str(obj.get("summary_ja", "") or ""))[:sum_chars]
                        rel = float(obj.get("relevance", 50) or 50)
                        qua = float(obj.get("quality", 50) or 50)
                        kws = obj.get("keywords", []) if isinstance(obj.get("keywords", []), list) else []
                except Exception as e:
                    self.log(f"LLM要約失敗 -> フォールバック: {type(e).__name__} {e}")

            if not summary:
                summary = extractive_summary(cleaned, sum_chars)
                qua = quality_heuristic(url, len(cleaned))
                rel = 50.0

            # Embedding + score
            if bool(self.s.get("rag_embed_title_summary", True)):
                emb_text = (title or "") + "\n" + (summary or "")
            else:
                emb_text = summary or ""

            svec = self.emb.embed([emb_text])[0]
            topic_sim = cos(self.topic_vec, svec) * 100.0

            # relevance は LLM評価と embedding類似の混合
            rel = max(0.0, min(100.0, rel * 0.6 + topic_sim * 0.4))
            qua = max(0.0, min(100.0, float(qua) if qua is not None else quality_heuristic(url, len(cleaned))))

            score = float(self.s.get("w_rel", 0.65)) * rel + float(self.s.get("w_qua", 0.35)) * qua
            score = max(0.0, min(100.0, score))

            did = self.store.upsert_doc(
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

            # related terms（LLMのkeywordsがあれば混ぜる）
            kw_src = title + "\n" + cleaned[:int(self.s.get("max_kw_chars", 14000))]
            extracted = extract_keywords(kw_src, self.topic, limit=20)
            if kws:
                for k in kws[:12]:
                    k = str(k).strip()
                    if k and k not in extracted:
                        extracted.insert(0, k)
                extracted = extracted[:24]

            for term in extracted:
                tvec = self.emb.embed([term])[0]
                tsim = cos(self.topic_vec, tvec)
                if tsim < float(self.s.get("rel_sim_threshold", 0.22)):
                    continue
                tscore = score * 0.55 + tsim * 100.0 * 0.45
                self.store.upsert_related(term, tscore, url)

            self.log(f"保存 score={score:.1f} rel={rel:.1f} qua={qua:.1f} url={url}")
            return True

        except Exception as e:
            self.log(f"ERROR url={url} type={type(e).__name__} msg={e}")
            return False

# ---------------- UI ----------------
def run():
    try:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    except Exception:
        pass

    os.makedirs("models/gpt-oss", exist_ok=True)

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
    log(f"runtime log: {os.path.abspath(LOG_PATH)}")

    s = load_settings()
    store = Store(DB_PATH)
    emb = Embedder(s, log)
    rag = RagIndex(emb.dim)
    rag.rebuild(store)

    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog

    root = tk.Tk()
    root.title(f"{APP} {VER}")
    root.geometry("1200x820")

    status = tk.StringVar(value="Ready")

    def set_status(x):
        ui_q.put(("status", x))

    def refresh():
        ui_q.put(("refresh", None))

    engine = SearchEngine(store, s, emb, rag, log, refresh, set_status)

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

    ttk.Button(top, text="開始", command=start).pack(side=tk.LEFT, padx=4)
    ttk.Button(top, text="停止", command=stop).pack(side=tk.LEFT, padx=4)

    ttk.Label(top, text="RAG閾値").pack(side=tk.LEFT, padx=(18, 4))
    rag_min_var = tk.StringVar(value=str(s.get("rag_min_sim", 0.28)))

    def apply_rag_min():
        try:
            v = float(rag_min_var.get().strip())
            v = max(0.0, min(0.99, v))
            s["rag_min_sim"] = v
            save_settings(s)
            log(f"rag_min_sim={v:.3f}")
        except Exception:
            pass

    e_min = ttk.Entry(top, textvariable=rag_min_var, width=8)
    e_min.pack(side=tk.LEFT)
    ttk.Button(top, text="反映", command=apply_rag_min).pack(side=tk.LEFT, padx=4)

    ttk.Label(top, textvariable=status).pack(side=tk.RIGHT)

    pan = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
    pan.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    left = ttk.Frame(pan, padding=6)
    right = ttk.Frame(pan, padding=6)
    pan.add(left, weight=1)
    pan.add(right, weight=2)

    ttk.Label(left, text="収集ドキュメント（score順）").pack(anchor="w")
    docs = tk.Listbox(left, height=16)
    docs.pack(fill=tk.BOTH, expand=False)

    ttk.Label(left, text="関連ワード（ランキング）").pack(anchor="w", pady=(10, 0))
    rel = tk.Listbox(left, height=12)
    rel.pack(fill=tk.BOTH, expand=True)

    act = ttk.LabelFrame(left, text="操作", padding=6)
    act.pack(fill=tk.X, pady=8)

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

    def delete_doc_selected():
        did = get_selected_doc_id()
        if not did:
            return messagebox.showinfo("info", "記事が選択されていません")
        if not messagebox.askyesno("confirm", "選択した記事を削除しますか"):
            return
        url = store.delete_doc(int(did))
        rag.remove(int(did))
        log(f"記事削除 id={did} url={url}")
        refresh()

    def delete_doc_all():
        if not messagebox.askyesno("confirm", "全記事を削除しますか（取り消し不可）"):
            return
        store.delete_all_docs()
        rag.clear()
        engine.seen.clear()
        log("全記事削除")
        refresh()

    def delete_kw_selected():
        term = get_selected_term()
        if not term:
            return messagebox.showinfo("info", "キーワードが選択されていません")
        if not messagebox.askyesno("confirm", f"キーワード「{term}」を削除しますか"):
            return
        store.delete_related(term)
        log("キーワード削除: " + term)
        refresh()

    def delete_kw_all():
        if not messagebox.askyesno("confirm", "全キーワードを削除しますか（取り消し不可）"):
            return
        store.delete_all_related()
        log("全キーワード削除")
        refresh()

    def export_all_text():
        rows = store.list_docs_full()
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

    ttk.Button(act, text="記事削除", command=delete_doc_selected).pack(side="left", padx=4)
    ttk.Button(act, text="全記事削除", command=delete_doc_all).pack(side="left", padx=4)
    ttk.Button(act, text="キーワード削除", command=delete_kw_selected).pack(side="left", padx=4)
    ttk.Button(act, text="全キーワード削除", command=delete_kw_all).pack(side="left", padx=4)
    ttk.Button(act, text="全記事エクスポート", command=export_all_text).pack(side="left", padx=4)

    # Right panel: RAG search
    rt = ttk.Frame(right)
    rt.pack(fill=tk.X)

    ttk.Label(rt, text="RAG検索").pack(side=tk.LEFT)
    qv = tk.StringVar()
    ttk.Entry(rt, textvariable=qv, width=48).pack(side=tk.LEFT, padx=6)

    rag_list = tk.Listbox(right, height=7)
    rag_list.pack(fill=tk.X)

    def do_rag():
        q = qv.get().strip()
        if not q or len(q) < 2:
            return

        qvec = emb.embed([q])[0]
        topk = int(s.get("rag_topk", 8))
        try:
            min_sim = float(s.get("rag_min_sim", 0.28))
        except Exception:
            min_sim = 0.28

        hits = rag.search(qvec, topk=topk, min_sim=min_sim)

        rag_list.delete(0, tk.END)
        rag_map.clear()

        if not hits:
            rag_list.insert(tk.END, f"一致なし（sim >= {min_sim:.2f}）")
            return

        for i, (did, sim) in enumerate(hits):
            doc = store.get_doc(did)
            if not doc:
                continue
            rag_list.insert(tk.END, f"{sim*100:.1f} | {clamp(doc.get('title',''),80)}")
            rag_map[i] = did

    ttk.Button(rt, text="検索", command=do_rag).pack(side=tk.LEFT)

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
        for i, (did, score, relv, quav, title, url, created) in enumerate(store.list_docs(limit=300)):
            docs.insert(tk.END, f"{float(score):5.1f} r{float(relv):4.1f} q{float(quav):4.1f} | {clamp(title,60)}")
            doc_map[i] = int(did)

        rel_map.clear()
        rel.delete(0, tk.END)
        for i, (term, score, last_seen, src) in enumerate(store.list_related(limit=200)):
            rel.insert(tk.END, f"{float(score):5.1f} | {term}")
            rel_map[i] = str(term)

    def show_doc(did: int):
        d = store.get_doc(did)
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
        txt.insert("end", clamp(d.get("cleaned", "") or "", 5200))

    def on_docs(_=None):
        sel = docs.curselection()
        if not sel:
            return
        did = doc_map.get(sel[0])
        if did:
            show_doc(did)

    docs.bind("<<ListboxSelect>>", on_docs)

    def on_rag(_=None):
        sel = rag_list.curselection()
        if not sel:
            return
        did = rag_map.get(sel[0])
        if did:
            show_doc(did)

    rag_list.bind("<<ListboxSelect>>", on_rag)

    def ui_tick():
        # UI event queue
        try:
            while True:
                ev, payload = ui_q.get_nowait()
                if ev == "status":
                    status.set(str(payload))
                elif ev == "refresh":
                    refresh_lists()
        except queue.Empty:
            pass

        # log queue
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
