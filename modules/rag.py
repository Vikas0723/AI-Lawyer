
import os
import re
import json
import numpy as np
import pymupdf
import faiss

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ==================================================
# CONFIGURATION
# ==================================================

_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_model = None
_bm25_index = None
_bm25_chunks_id = None

DATA_DIR = "data"

INDEX_PATH = os.path.join(DATA_DIR, "legal.faiss")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.json")


# ==================================================
# EMBEDDING MODEL
# ==================================================

def embedding_model():

    global _model

    if _model is None:
        _model = SentenceTransformer(_EMBED_MODEL)

    return _model


# ==================================================
# TEXT EXTRACTION
# ==================================================

def extract_pages(file_path):

    """Extract text page by page from PDF or TXT."""

    pages = []

    if file_path.lower().endswith(".pdf"):

        doc = pymupdf.open(file_path)

        for page_number, page in enumerate(doc, start=1):

            text = page.get_text("text").strip()

            if text:
                pages.append((page_number, text))

        doc.close()

    elif file_path.lower().endswith(".txt"):

        with open(
            file_path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            pages.append((1, f.read()))

    return pages


# ==================================================
# TEXT CHUNKING
# ==================================================

def chunk_text(text, chunk_size=900, overlap=140):

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))

        piece = text[start:end].strip()

        if piece:
            chunks.append(piece)

        if end == len(text):
            break

        start = end - overlap

    return chunks


# ==================================================
# BM25 TOKENIZATION
# ==================================================

def tokenize(text):

    """
    Basic multilingual tokenizer.

    Preserves:
    - English words
    - Hindi words
    - Numbers
    - Section identifiers
    """

    return re.findall(
        r"[a-zA-Z]+|\d+|[^\W\d_]+",
        text.lower(),
        flags=re.UNICODE
    )


# ==================================================
# BUILD FAISS INDEX
# ==================================================

def create_faiss_index(texts):

    vectors = embedding_model().encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    vectors = np.asarray(
        vectors,
        dtype="float32"
    )

    index = faiss.IndexFlatIP(vectors.shape[1])

    index.add(vectors)

    return index


# ==================================================
# BUILD DOCUMENT INDEX
# ==================================================

def build_index_from_paths(file_paths):

    texts = []
    metadata = []

    for file_path in file_paths:

        pages = extract_pages(file_path)

        for page_number, page_text in pages:

            pieces = chunk_text(page_text)

            for i, piece in enumerate(pieces):

                texts.append(piece)

                metadata.append({
                    "title": os.path.basename(file_path),
                    "source": os.path.basename(file_path),
                    "page": page_number,
                    "chunk": i + 1
                })

    if not texts:

        raise ValueError(
            "No readable text found in the documents."
        )

    index = create_faiss_index(texts)

    chunks = [
        {"text": text, "metadata": meta}
        for text, meta in zip(texts, metadata)
    ]

    return index, chunks


# ==================================================
# SAVE INDEX
# ==================================================

def save_index(index, chunks):

    os.makedirs(DATA_DIR, exist_ok=True)

    faiss.write_index(index, INDEX_PATH)

    with open(
        CHUNKS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2
        )


# ==================================================
# LOAD INDEX
# ==================================================

def load_index():

    if not (
        os.path.exists(INDEX_PATH)
        and os.path.exists(CHUNKS_PATH)
    ):

        return None, []

    index = faiss.read_index(INDEX_PATH)

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        chunks = json.load(f)

    if index.ntotal != len(chunks):

        raise ValueError(
            "FAISS index and metadata are inconsistent."
        )

    return index, chunks


# ==================================================
# BUILD BM25 INDEX
# ==================================================

def build_bm25(chunks):

    global _bm25_index, _bm25_chunks_id

    if not chunks:
        _bm25_index = None
        _bm25_chunks_id = None
        return None

    tokenized_chunks = [
        tokenize(item["text"])
        for item in chunks
    ]

    _bm25_index = BM25Okapi(tokenized_chunks)

    _bm25_chunks_id = id(chunks)

    return _bm25_index


# ==================================================
# FAISS RETRIEVAL
# ==================================================

def retrieve_faiss(
    query,
    index,
    chunks,
    top_k=10
):

    if index is None or not chunks:
        return []

    vector = embedding_model().encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    vector = np.asarray(
        vector,
        dtype="float32"
    )

    scores, ids = index.search(
        vector,
        min(top_k, len(chunks))
    )

    results = []

    for score, idx in zip(scores[0], ids[0]):

        if idx < 0:
            continue

        item = dict(chunks[int(idx)])

        item["faiss_score"] = float(score)

        results.append(item)

    return results


# ==================================================
# BM25 RETRIEVAL
# ==================================================

def retrieve_bm25(
    query,
    chunks,
    top_k=10
):

    global _bm25_index, _bm25_chunks_id

    if not chunks:
        return []

    # Rebuild BM25 if a different chunk collection is used
    if (
        _bm25_index is None
        or _bm25_chunks_id != id(chunks)
    ):

        build_bm25(chunks)

    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    scores = _bm25_index.get_scores(query_tokens)

    top_ids = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in top_ids:

        score = float(scores[idx])

        if score <= 0:
            continue

        item = dict(chunks[int(idx)])

        item["bm25_score"] = score

        results.append(item)

    return results


# ==================================================
# RECIPROCAL RANK FUSION
# ==================================================

def reciprocal_rank_fusion(
    faiss_results,
    bm25_results,
    k=60,
    top_k=5
):

    """
    Combine FAISS and BM25 rankings.

    RRF score = sum(1 / (k + rank))
    """

    fused = {}

    # FAISS ranking
    for rank, item in enumerate(
        faiss_results,
        start=1
    ):

        chunk_id = (
            item["metadata"]["source"],
            item["metadata"]["page"],
            item["metadata"]["chunk"]
        )

        if chunk_id not in fused:

            fused[chunk_id] = {
                "item": item,
                "rrf_score": 0.0
            }

        fused[chunk_id]["rrf_score"] += (
            1.0 / (k + rank)
        )

    # BM25 ranking
    for rank, item in enumerate(
        bm25_results,
        start=1
    ):

        chunk_id = (
            item["metadata"]["source"],
            item["metadata"]["page"],
            item["metadata"]["chunk"]
        )

        if chunk_id not in fused:

            fused[chunk_id] = {
                "item": item,
                "rrf_score": 0.0
            }

        fused[chunk_id]["rrf_score"] += (
            1.0 / (k + rank)
        )

    ranked = sorted(
        fused.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    results = []

    for rank, entry in enumerate(
        ranked[:top_k],
        start=1
    ):

        item = dict(entry["item"])

        item["rrf_score"] = entry["rrf_score"]
        item["rank"] = rank

        results.append(item)

    return results


# ==================================================
# HYBRID RETRIEVAL
# ==================================================

def retrieve(
    query,
    index,
    chunks,
    top_k=4
):

    if index is None or not chunks:
        return []

    # Retrieve candidates from both methods
    faiss_results = retrieve_faiss(
        query,
        index,
        chunks,
        top_k=20
    )

    bm25_results = retrieve_bm25(
        query,
        chunks,
        top_k=20
    )

    # Merge their rankings
    results = reciprocal_rank_fusion(
        faiss_results,
        bm25_results,
        top_k=top_k
    )

    return results


# ==================================================
# DEBUG RETRIEVAL
# ==================================================

def retrieve_debug(
    query,
    index,
    chunks,
    top_k=5
):

    if index is None or not chunks:
        return []

    faiss_results = retrieve_faiss(
        query,
        index,
        chunks,
        top_k=20
    )

    bm25_results = retrieve_bm25(
        query,
        chunks,
        top_k=20
    )

    return reciprocal_rank_fusion(
        faiss_results,
        bm25_results,
        top_k=top_k
    )


# ==================================================
# BUILD INDEX FROM UPLOADED FILES
# ==================================================

def build_index_from_uploaded_files(files):

    texts = []
    metadata = []

    for file in files:

        data = file.getvalue()

        if file.name.lower().endswith(".pdf"):

            doc = pymupdf.open(
                stream=data,
                filetype="pdf"
            )

            for page_number, page in enumerate(
                doc,
                start=1
            ):

                page_text = page.get_text("text")

                pieces = chunk_text(page_text)

                for i, piece in enumerate(pieces):

                    texts.append(piece)

                    metadata.append({
                        "title": file.name,
                        "source": file.name,
                        "page": page_number,
                        "chunk": i + 1
                    })

            doc.close()

        elif file.name.lower().endswith(".txt"):

            raw = data.decode(
                "utf-8",
                errors="ignore"
            )

            for i, piece in enumerate(
                chunk_text(raw)
            ):

                texts.append(piece)

                metadata.append({
                    "title": file.name,
                    "source": file.name,
                    "page": 1,
                    "chunk": i + 1
                })

    if not texts:

        raise ValueError(
            "No readable text found in uploaded documents."
        )

    index = create_faiss_index(texts)

    chunks = [
        {"text": text, "metadata": meta}
        for text, meta in zip(texts, metadata)
    ]

    # Build BM25 for uploaded documents
    build_bm25(chunks)

    return index, chunks
