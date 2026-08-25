import os
import re
import json
import math
from pathlib import Path
from collections import Counter

from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from dotenv import load_dotenv

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
BOOK_DIR = BASE_DIR / "data" / "books"
MANIFEST_FILE = BASE_DIR / "books.json"

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
API_KEY = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)
CORS(app)

client = genai.Client(api_key=API_KEY) if API_KEY else None

BOOKS = {}
CHUNKS = []

STOPWORDS = {
    "yang","dan","di","ke","dari","ini","itu","untuk","dengan","pada",
    "dalam","adalah","atau","apa","bagaimana","mengapa","saya","kita",
    "anda","om","tije","buku","tentang","sebagai","sebuah","yang"
}

def normalize(text):
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9à-ÿ\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def tokens(text):
    return [w for w in normalize(text).split() if len(w) >= 3 and w not in STOPWORDS]

def likely_heading(text):
    lines = [x.strip() for x in str(text).splitlines() if x.strip()]
    for line in lines[:8]:
        clean = re.sub(r"\s+", " ", line)
        if 4 <= len(clean) <= 100:
            letters = re.sub(r"[^A-Za-zÀ-ÿ]", "", clean)
            upper = sum(c.isupper() for c in letters)
            if letters and upper / len(letters) > 0.72:
                return clean
            if re.match(r"^(bab|bagian|pendahuluan|kesimpulan|penutup|pengantar)\b", clean, re.I):
                return clean
    return ""

def add_chunk(book_id, title, page, text):
    text = str(text or "").strip()
    if len(text) < 30:
        return
    # Overlapping chunks keep page references exact.
    words = text.split()
    step = 170
    size = 230
    for start in range(0, len(words), step):
        part = " ".join(words[start:start+size]).strip()
        if len(part) < 30:
            continue
        CHUNKS.append({
            "book_id": book_id,
            "book": title,
            "page": page,
            "section": likely_heading(part),
            "text": part,
            "tokens": Counter(tokens(part)),
        })
        if start + size >= len(words):
            break

def load_manifest():
    if not MANIFEST_FILE.exists():
        return {}
    try:
        data = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        return {x["id"]: x for x in data}
    except Exception:
        return {}

def display_title_from_filename(path):
    return re.sub(r"\s+", " ", path.stem.replace("-", " ")).strip().title()

def load_books():
    global BOOKS, CHUNKS
    BOOKS = {}
    CHUNKS = []
    manifest = load_manifest()

    for book_id, item in manifest.items():
        BOOKS[book_id] = {
            "id": book_id,
            "title": item.get("title") or book_id,
            "file": item.get("file") or f"{book_id}.pdf",
            "pages": 0,
            "status": "belum ada file",
        }

    # Also discover future PDF/TXT/MD files automatically.
    for path in sorted(BOOK_DIR.iterdir()) if BOOK_DIR.exists() else []:
        if path.suffix.lower() not in {".pdf",".txt",".md"}:
            continue

        book_id = path.stem
        if book_id not in BOOKS:
            BOOKS[book_id] = {
                "id": book_id,
                "title": display_title_from_filename(path),
                "file": path.name,
                "pages": 0,
                "status": "terdeteksi",
            }

    for book_id, book in BOOKS.items():
        path = BOOK_DIR / book["file"]
        if not path.exists():
            # Fallback: exact id with common extensions.
            for ext in (".pdf",".txt",".md"):
                candidate = BOOK_DIR / f"{book_id}{ext}"
                if candidate.exists():
                    path = candidate
                    break

        if not path.exists():
            continue

        try:
            if path.suffix.lower() == ".pdf":
                if PdfReader is None:
                    book["status"] = "pypdf belum terpasang"
                    continue
                reader = PdfReader(str(path))
                book["pages"] = len(reader.pages)
                for page_no, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    add_chunk(book_id, book["title"], page_no, text)
            else:
                text = path.read_text(encoding="utf-8", errors="ignore")
                book["pages"] = 1
                add_chunk(book_id, book["title"], 1, text)

            book["status"] = "siap"
        except Exception as exc:
            book["status"] = f"gagal dibaca: {exc}"

def score_chunk(question, chunk):
    q = Counter(tokens(question))
    if not q:
        return 0.0
    overlap = sum(min(q[w], chunk["tokens"].get(w, 0)) for w in q)
    qset = set(q)
    cset = set(chunk["tokens"])
    union = len(qset | cset) or 1
    jaccard = len(qset & cset) / union
    title_bonus = 0
    for w in tokens(question):
        if w in tokens(chunk["book"]):
            title_bonus += 0.35
    return overlap + (jaccard * 8) + title_bonus

def retrieve(question, book_id=None, limit=7):
    pool = [c for c in CHUNKS if not book_id or c["book_id"] == book_id]
    ranked = sorted(pool, key=lambda c: score_chunk(question, c), reverse=True)
    return [c for c in ranked[:limit] if score_chunk(question, c) > 0]

def format_sources(chunks):
    out = []
    seen = set()
    for c in chunks:
        key = (c["book_id"], c["page"])
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "book": c["book"],
            "chapter": c.get("section") or "",
            "page": c["page"]
        })
    return out

SYSTEM_RULES = """
Anda adalah Om Tije.

Anda berbicara langsung kepada pembaca dengan suara seorang penulis,
pembicara, dan manusia yang sedang diajak ngobrol.

Jangan bertindak sebagai narator, pengulas, atau peringkas buku.
Jangan menjelaskan buku dari sudut pandang orang ketiga.

GAYA PERCAKAPAN:
1. Gunakan Bahasa Indonesia yang natural, hangat, sederhana, dan reflektif.
2. Jawaban harus terasa seperti Om Tije sedang menjawab langsung.
3. Untuk pertanyaan sederhana, jawab singkat. Umumnya 2-5 kalimat sudah cukup.
4. Jangan mengulang pertanyaan pembaca.
5. Jangan menggunakan kalimat pembuka generik seperti:
   "Pertanyaan yang sangat menarik."
   "Tentu saja."
   "Sebagai AI..."
6. Jangan menggunakan Markdown.
7. Jangan gunakan tanda **, ##, ---, atau bullet Markdown.
8. Jangan menyebut Gemini, AI, model, API, database, server,
   File Search, prompt, sistem, atau teknologi internal.

KETIKA PEMBACA MEMILIH BUKU:
1. Gunakan isi buku yang dipilih sebagai dasar utama jawaban.
2. Bicaralah sebagai Om Tije, bukan sebagai orang yang sedang mengulas buku.
3. Jangan mengatakan:
   "Berdasarkan buku ini..."
   "Dalam pandangan buku ini..."
   "Buku ini menjelaskan..."
   "Menurut buku..."
   "Dalam buku tersebut..."
   "Menurut penulis..."
   "Tri Tjahyono mengatakan..."
4. Jangan menyebut nama buku di dalam narasi jawaban kecuali pertanyaan
   pembaca memang meminta nama buku atau konteksnya membutuhkan itu.
5. Jangan mengatakan "saya menulis dalam buku..." kecuali pembaca memang
   bertanya langsung tentang proses penulisan atau pengalaman penulis.
6. Ambil gagasan dari sumber lalu sampaikan kembali secara natural sebagai
   jawaban Om Tije.
7. Jangan menyalin paragraf panjang dari sumber.
8. Jika sumber yang ditemukan tidak cukup untuk menjawab pertanyaan,
   katakan dengan jujur bahwa bagian itu belum cukup ditemukan.

CONTOH GAYA YANG BENAR:
"Menurut saya, penyuluh tidak berhenti pada menyampaikan informasi.
Yang lebih penting adalah hadir, mendengarkan, dan membantu orang
menemukan kesadarannya sendiri."

CONTOH YANG SALAH:
"Berdasarkan buku Penyuluh Hebat, penyuluh adalah..."
"Dalam pandangan buku ini..."
"Buku tersebut menjelaskan bahwa..."

KETIKA TANYA UMUM:
1. Jawab sebagai Om Tije secara langsung.
2. Gunakan wawasan umum dan pola gagasan yang relevan dari sumber yang tersedia.
3. Jangan mengatakan bahwa jawaban berasal dari "koleksi buku".
4. Jangan memaksakan hubungan dengan buku jika memang tidak relevan.
5. Jika ada gagasan yang kuat dan relevan dari sumber, sistem akan menampilkan
   sumbernya di luar narasi jawaban.

SUMBER:
1. Jangan membuat bagian "Sumber:" di dalam jawaban.
2. Jangan menulis nama buku, halaman, atau daftar referensi di dalam jawaban.
3. Metadata sumber akan dikirim terpisah oleh server kepada tampilan web.
4. Jangan mengarang halaman.

SALAM:
1. Jangan mengawali setiap jawaban dengan "Assalamu 'alaikum wr wb."
2. Salam pembuka hanya digunakan oleh tampilan awal percakapan.
3. Jangan mengulang salam pada setiap pertanyaan.
4. "Wassalamu 'alaikum wr wb." hanya digunakan ketika pembaca benar-benar
   mengakhiri percakapan. Jika sistem sudah menambahkan salam penutup,
   jangan menambahkannya lagi.
"""

def clean_answer(text):
    text = str(text or "")
    text = text.replace("**", "")
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"(?im)^\s*(sumber|referensi)\s*:\s*.*$", "", text)
    text = re.sub(r"(?im)^\s*(halaman|page)\s*:\s*\d+\s*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def ask_gemini(question, selected_book, chunks):
    if not client:
        raise RuntimeError("GEMINI_API_KEY belum diatur.")

    mode = (
        f"MODE: BUKU KHUSUS\nBuku yang dipilih: {selected_book['title']}"
        if selected_book else
        "MODE: TANYA UMUM\nTidak ada buku yang dipilih."
    )

    if chunks:
        context = "\n\n".join(
            f"[Sumber {i+1} | {c['book']} | halaman {c['page']}]\n{c['text']}"
            for i,c in enumerate(chunks)
        )
    else:
        context = "(Tidak ada potongan sumber yang cocok ditemukan.)"

    prompt = f"""{SYSTEM_RULES}

{mode}

Pertanyaan pembaca:
{question}

Potongan sumber yang tersedia:
{context}

Tugas:
Jawab langsung pertanyaan pembaca sebagai Om Tije.

Gunakan sumber di atas sebagai bahan pemahaman, bukan sebagai sesuatu
yang harus diceritakan kembali dengan gaya ulasan buku.

Jangan memulai jawaban dengan nama buku.
Jangan mengatakan "berdasarkan buku", "dalam buku ini", "buku ini menjelaskan",
atau frasa serupa.

Jika mode buku, jawab seolah-olah Om Tije sedang berbicara langsung
kepada pembaca dengan pemahaman terhadap gagasan dalam buku yang dipilih.

Jika mode umum, jawab secara alami. Sumber hanya membantu memperkaya
jawaban jika memang relevan.

Jangan tulis sumber, nomor halaman, atau referensi di dalam jawaban.
Server akan menampilkan metadata sumber secara terpisah.
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )
    return clean_answer(response.text)

@app.route("/", methods=["GET"])
def home():
    ready = sum(1 for b in BOOKS.values() if b["status"] == "siap")
    return jsonify({
        "status": "ok",
        "service": "Tanya Om Tije",
        "model": MODEL,
        "architecture": "local-book-index",
        "file_search_store": False,
        "books": len(BOOKS),
        "books_ready": ready,
    })

@app.route("/api/books", methods=["GET"])
def books():
    return jsonify({
        "success": True,
        "books": list(BOOKS.values())
    })

@app.route("/api/reload", methods=["POST"])
def reload_books():
    load_books()
    return jsonify({
        "success": True,
        "books": len(BOOKS),
        "chunks": len(CHUNKS)
    })

@app.route("/api/tanya", methods=["POST"])
def tanya():
    try:
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()
        book_id = (data.get("book_id") or "").strip() or None

        if not question:
            return jsonify({"success": False, "message": "Pertanyaan belum diisi."}), 400

        selected = BOOKS.get(book_id) if book_id else None

        if book_id and not selected:
            return jsonify({"success": False, "message": "Buku yang dipilih belum terdaftar."}), 400

        if selected and selected["status"] != "siap":
            return jsonify({
                "success": False,
                "message": f'Isi buku "{selected["title"]}" belum tersedia di server.'
            }), 400

        chunks = retrieve(question, book_id=book_id, limit=8)

        # General mode: search across the collection.
        if not book_id:
            chunks = retrieve(question, book_id=None, limit=10)

        answer = ask_gemini(
            question,
            {"title": selected["title"], "id": selected["id"]} if selected else None,
            chunks
        )

        sources = format_sources(chunks)

        # Frontend currently expects a single source object.
        source = sources[0] if sources else None

        return jsonify({
            "success": True,
            "answer": answer,
            "source": source,
            "sources": sources,
            "mode": "book" if selected else "general"
        })

    except Exception as exc:
        print("=" * 70)
        print("ERROR TANYA OM TIJE")
        print(repr(exc))
        print("=" * 70)
        return jsonify({
            "success": False,
            "message": "Om Tije sedang mengalami kendala teknis. Periksa server dan koleksi buku."
        }), 500

if __name__ == "__main__":
    BOOK_DIR.mkdir(parents=True, exist_ok=True)
    load_books()

    # Lokal tetap memakai 127.0.0.1:5000.
    # Hosting seperti Render menyediakan PORT melalui environment variable.
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"

    print("=" * 70)
    print("TANYA OM TIJE - BOOK INDEX")
    print("=" * 70)
    print("Model       :", MODEL)
    print("Store       : TIDAK DIGUNAKAN")
    print("Buku        :", len(BOOKS))
    print("Siap        :", sum(1 for b in BOOKS.values() if b["status"] == "siap"))
    print("Chunks      :", len(CHUNKS))
    print("Server      :", f"http://{host}:{port}")
    print("Debug       :", debug)
    print("=" * 70)

    app.run(host=host, port=port, debug=debug)
