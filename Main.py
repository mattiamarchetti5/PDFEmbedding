"""
Sistema RAG (Retrieval Augmented Generation) locale
----------------------------------------------------
Estrae testo da un PDF, lo indicizza con embedding ONNX (nomic-embed-text-v1.5),
salva i vettori in ChromaDB, e risponde a domande usando un LLM GGUF (Gemma-2B).

Autore: GitHub Copilot
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ─── Configurazione logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SearchAI")

# ─── Percorsi ────────────────────────────────────────────────────────────────
# Usa __file__ per garantire percorsi assoluti indipendentemente
# dalla directory di lavoro da cui viene lanciato lo script.
BASE_DIR        = Path(__file__).resolve().parent
RESOURCES_DIR   = BASE_DIR / "resources"

PDF_PATH        = RESOURCES_DIR / "Riposo15gg.pdf"
ONNX_MODEL_PATH = RESOURCES_DIR / "model_int8.onnx"
LLM_MODEL_PATH  = RESOURCES_DIR / "gemma-2b.gguf"
TOKENIZER_PATH  = RESOURCES_DIR / "nomic-tokenizer"
CHROMA_DB_PATH  = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "pdf_rag"

# ─── Parametri ────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 500          # caratteri per chunk
CHUNK_OVERLAP: int = 50        # overlap tra chunk consecutivi
TOP_K: int = 4                 # chunk da recuperare per query
LLM_MAX_TOKENS: int = 512
LLM_TEMPERATURE: float = 0.2
LLM_CONTEXT_WINDOW: int = 4096


# ══════════════════════════════════════════════════════════════════════════════
# 1. PDF LOADER
# ══════════════════════════════════════════════════════════════════════════════

class PDFLoader:
    """Estrae il testo grezzo da un file PDF."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> str:
        """Restituisce il testo concatenato di tutte le pagine del PDF."""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("Installa pypdf: pip install pypdf")

        logger.info("Caricamento PDF: %s", self.path)
        if not self.path.exists():
            raise FileNotFoundError(f"PDF non trovato: {self.path}")

        reader = PdfReader(str(self.path))
        pages_text: List[str] = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages_text.append(text.strip())
            logger.debug("Pagina %d estratta (%d caratteri)", i + 1, len(text))

        full_text = "\n\n".join(pages_text)
        logger.info("Testo estratto: %d caratteri totali", len(full_text))
        return full_text


# ══════════════════════════════════════════════════════════════════════════════
# 2. TEXT CHUNKER
# ══════════════════════════════════════════════════════════════════════════════

class TextChunker:
    """Divide il testo in chunk sovrapposti."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        """Restituisce una lista di chunk di testo."""
        chunks: List[str] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size

            if end >= text_len:
                chunks.append(text[start:].strip())
                break

            # Cerca il punto di split più vicino (spazio)
            split_at = text.rfind(" ", start, end)
            if split_at == -1 or split_at <= start:
                split_at = end

            chunk = text[start:split_at].strip()
            if chunk:
                chunks.append(chunk)

            # Avanza con overlap
            start = split_at - self.overlap

        logger.info("Testo spezzato in %d chunk", len(chunks))
        return chunks


# ══════════════════════════════════════════════════════════════════════════════
# 3. ONNX EMBEDDER
# ══════════════════════════════════════════════════════════════════════════════

class ONNXEmbedder:
    """
    Genera embedding con il modello ONNX nomic-embed-text-v1.5.
    Usa il tokenizer HuggingFace e l'inferenza ONNX Runtime.
    """

    TOKENIZER_ID = "nomic-ai/nomic-embed-text-v1.5"

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._session = None
        self._tokenizer = None
        self._load()

    def _load(self) -> None:
        """Carica il tokenizer e la sessione ONNX."""
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("Installa: pip install onnxruntime transformers")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Modello ONNX non trovato: {self.model_path}")

        if not TOKENIZER_PATH.exists():
            raise FileNotFoundError(f"Tokenizer non trovato: {TOKENIZER_PATH}")

        logger.info("Caricamento tokenizer da locale: %s", TOKENIZER_PATH)
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(TOKENIZER_PATH),
            local_files_only=True,
        )

        logger.info("Caricamento modello ONNX: %s", self.model_path)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        logger.info("Modello ONNX pronto")

    def _mean_pooling(self, token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Applica mean pooling sugli hidden states pesato dalla attention mask."""
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalizzazione L2."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-12)

    def encode(self, texts: List[str], prefix: str = "search_document") -> np.ndarray:
        """
        Codifica una lista di testi aggiungendo il prefisso nomic.

        Args:
            texts: Lista di stringhe da codificare.
            prefix: 'search_document' per i chunk, 'search_query' per le domande.

        Returns:
            Array numpy di shape (n, embedding_dim).
        """
        prefixed = [f"{prefix}: {t}" for t in texts]

        encoded = self._tokenizer(
            prefixed,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )

        input_ids = encoded["input_ids"].astype(np.int64)
        attention_mask = encoded["attention_mask"].astype(np.int64)

        # Inferenza ONNX
        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        # Aggiunge token_type_ids se richiesto dal modello
        if "token_type_ids" in [inp.name for inp in self._session.get_inputs()]:
            token_type_ids = encoded.get("token_type_ids", np.zeros_like(input_ids))
            ort_inputs["token_type_ids"] = token_type_ids.astype(np.int64)

        outputs = self._session.run(None, ort_inputs)
        token_embeddings = outputs[0]  # (batch, seq_len, hidden_dim)

        pooled = self._mean_pooling(token_embeddings, attention_mask)
        normalized = self._normalize(pooled)
        return normalized


# ══════════════════════════════════════════════════════════════════════════════
# 4. VECTOR STORE (ChromaDB)
# ══════════════════════════════════════════════════════════════════════════════

class VectorStore:
    """Gestisce l'archiviazione e il recupero degli embedding con ChromaDB."""

    def __init__(self, persist_path: str = CHROMA_DB_PATH, collection: str = COLLECTION_NAME) -> None:
        try:
            import chromadb
        except ImportError:
            raise ImportError("Installa: pip install chromadb")

        self._client = chromadb.PersistentClient(path=persist_path)
        self._collection_name = collection
        self._collection = None

    def create_collection(self) -> None:
        """Crea (o ricrea) la collection cancellando quella esistente."""
        import chromadb
        try:
            self._client.delete_collection(self._collection_name)
            logger.info("Collection esistente rimossa")
        except Exception:
            pass

        self._collection = self._client.create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection '%s' creata", self._collection_name)

    def load_or_create(self) -> bool:
        """
        Tenta di caricare la collection esistente.
        Ritorna True se esiste già, False se va creata.
        """
        import chromadb
        try:
            self._collection = self._client.get_collection(self._collection_name)
            count = self._collection.count()
            if count > 0:
                logger.info("Collection esistente caricata (%d chunk)", count)
                return True
        except Exception:
            pass
        return False

    def add_documents(self, chunks: List[str], embeddings: np.ndarray) -> None:
        """Inserisce chunk ed embedding nella collection."""
        if self._collection is None:
            self.create_collection()

        ids = [str(i) for i in range(len(chunks))]
        self._collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=chunks,
        )
        logger.info("Aggiunti %d chunk alla collection", len(chunks))

    def query(self, query_embedding: np.ndarray, n_results: int = TOP_K) -> List[Tuple[str, float]]:
        """
        Restituisce i chunk più simili con il loro score di distanza.

        Returns:
            Lista di tuple (testo_chunk, distanza).
        """
        results = self._collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=min(n_results, self._collection.count()),
        )
        docs = results["documents"][0]
        distances = results["distances"][0]
        return list(zip(docs, distances))


# ══════════════════════════════════════════════════════════════════════════════
# 5. LLM GENERATOR (llama-cpp-python)
# ══════════════════════════════════════════════════════════════════════════════

class LLMGenerator:
    """Genera risposte usando un LLM GGUF locale tramite llama-cpp-python."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._llm = None
        self._load()

    def _load(self) -> None:
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("Installa: pip install llama-cpp-python")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Modello LLM non trovato: {self.model_path}")

        logger.info("Caricamento LLM: %s", self.model_path)
        self._llm = Llama(
            model_path=str(self.model_path),
            n_ctx=LLM_CONTEXT_WINDOW,
            n_threads=os.cpu_count() or 4,
            verbose=False,
        )
        logger.info("LLM pronto")

    def generate(self, question: str, context_chunks: List[str]) -> str:
        """
        Genera una risposta data la domanda e i chunk di contesto.

        Args:
            question: La domanda dell'utente.
            context_chunks: Lista di testi estratti dal PDF.

        Returns:
            La risposta generata dal modello.
        """
        context = "\n\n---\n\n".join(context_chunks)

        prompt = (
            "Sei un assistente esperto. Rispondi alla domanda basandoti SOLO "
            "sul contesto fornito. Se la risposta non è nel contesto, dillo chiaramente.\n\n"
            f"### Contesto:\n{context}\n\n"
            f"### Domanda:\n{question}\n\n"
            "### Risposta:"
        )

        output = self._llm(
            prompt,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            stop=["###", "\n\n\n"],
            echo=False,
        )
        return output["choices"][0]["text"].strip()


# ══════════════════════════════════════════════════════════════════════════════
# 6. PIPELINE RAG
# ══════════════════════════════════════════════════════════════════════════════

class RAGPipeline:
    """Orchestratore del sistema RAG end-to-end."""

    def __init__(self) -> None:
        logger.info("Inizializzazione pipeline RAG...")
        self.embedder = ONNXEmbedder(ONNX_MODEL_PATH)
        self.vector_store = VectorStore()
        self.llm = LLMGenerator(LLM_MODEL_PATH)

    def index_pdf(self, pdf_path: Path, force_reindex: bool = False) -> None:
        """
        Indicizza il PDF nel vector store.
        Se il DB esiste già e force_reindex=False, salta l'indicizzazione.
        """
        if not force_reindex and self.vector_store.load_or_create():
            logger.info("Indicizzazione saltata: DB già esistente")
            return

        # Estrazione testo
        loader = PDFLoader(pdf_path)
        text = loader.load()

        # Chunking
        chunker = TextChunker()
        chunks = chunker.chunk(text)

        # Embedding
        logger.info("Creazione embedding per %d chunk...", len(chunks))
        embeddings = self.embedder.encode(chunks, prefix="search_document")

        # Salvataggio
        self.vector_store.create_collection()
        self.vector_store.add_documents(chunks, embeddings)
        logger.info("Indicizzazione completata!")

    def answer(self, question: str) -> str:
        """
        Risponde a una domanda cercando i chunk rilevanti e generando la risposta.

        Args:
            question: La domanda dell'utente.

        Returns:
            La risposta generata dal LLM.
        """
        # Embedding della query
        query_emb = self.embedder.encode([question], prefix="search_query")

        # Ricerca chunk simili
        results = self.vector_store.query(query_emb, n_results=TOP_K)
        context_chunks = [doc for doc, _ in results]

        logger.debug("Chunk recuperati: %d", len(context_chunks))
        for i, (doc, dist) in enumerate(results, 1):
            logger.debug("  [%d] distanza=%.4f | %s...", i, dist, doc[:80])

        # Generazione risposta
        response = self.llm.generate(question, context_chunks)
        return response


# ══════════════════════════════════════════════════════════════════════════════
# 7. ENTRY POINT – CLI INTERATTIVA
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Avvia il sistema RAG e il loop interattivo."""
    print("\n" + "═" * 60)
    print("  🔍  SearchAI – Sistema RAG Locale")
    print("═" * 60)

    # Controlla se forzare il re-indicizzazione
    force_reindex = "--reindex" in sys.argv

    # Avvio pipeline
    rag = RAGPipeline()

    # Indicizzazione PDF
    rag.index_pdf(PDF_PATH, force_reindex=force_reindex)

    print("\n✅ Sistema pronto! Digita la tua domanda (o 'quit' per uscire).\n")

    while True:
        try:
            question = input("❓ Domanda: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nArrivederci!")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Arrivederci!")
            break

        print("\n⏳ Elaborazione in corso...")
        try:
            answer = rag.answer(question)
            print(f"\n🤖 Risposta:\n{answer}\n")
            print("─" * 60)
        except Exception as e:
            logger.error("Errore durante la risposta: %s", e)
            print(f"❌ Errore: {e}\n")


if __name__ == "__main__":
    main()

