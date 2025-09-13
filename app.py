'''
Exemplo de aplicação RAG (Retrieval-Augmented Generation) com FastAPI e Hugging Face.

Esta aplicação permite:
1. Ingerir documentos de texto em um índice vetorial (FAISS).
2. Fazer perguntas e obter respostas geradas por um LLM (Hugging Face Inference API),
   com base no contexto recuperado do índice.
'''

import os
from typing import List

import faiss
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# --- Configurações --- #
HF_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
HF_MODEL = "microsoft/DialoGPT-medium"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INFERENCE_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

# --- Validação de Configuração --- #
if not HF_API_TOKEN:
    raise ValueError("A variável de ambiente HUGGINGFACE_API_TOKEN não foi definida. Crie um arquivo .env ou exporte a variável.")

# --- Modelos e Banco de Dados --- #
print("Carregando modelo de embeddings...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
dim = embedder.get_sentence_embedding_dimension()

# Usando IndexFlatIP para produto interno, que é equivalente à similaridade de cosseno em vetores normalizados
index = faiss.IndexFlatIP(dim)
documents: List[str] = []

print("Modelos e FAISS prontos.")

# --- API com FastAPI --- #
app = FastAPI(
    title="RAG com Hugging Face",
    description="Uma API para realizar Retrieval-Augmented Generation usando FastAPI, FAISS e Hugging Face Inference API.",
    version="1.0.0",
)

# --- Modelos de Dados (Pydantic) --- #
class IngestRequest(BaseModel):
    text: str

class IngestResponse(BaseModel):
    status: str
    indexed_text: str

class AskRequest(BaseModel):
    question: str
    top_k: int = 3

class AskResponse(BaseModel):
    answer: str
    context: str

# --- Endpoints da API --- #
@app.post("/ingest", response_model=IngestResponse, summary="Adicionar texto ao índice")
def ingest_text(item: IngestRequest):
    '''
    Recebe um texto, gera seu embedding e o adiciona ao índice FAISS.
    '''
    try:
        # Normalizar os embeddings para usar com IndexFlatIP (similaridade de cosseno)
        vec = embedder.encode([item.text], convert_to_numpy=True, normalize_embeddings=True)
        index.add(vec)
        documents.append(item.text)
        return {"status": "added", "indexed_text": item.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao ingerir texto: {e}")

@app.post("/ask", response_model=AskResponse, summary="Fazer uma pergunta ao modelo RAG")
def ask_question(item: AskRequest):
    '''
    Recebe uma pergunta, busca o contexto relevante no índice e gera uma resposta usando o LLM.
    '''
    try:
        # 1. Recuperação (Retrieval)
        question_embedding = embedder.encode([item.question], convert_to_numpy=True, normalize_embeddings=True)
        distances, indices = index.search(question_embedding, item.top_k)

        # Garante que temos resultados válidos
        retrieved_docs = [documents[i] for i in indices[0] if i < len(documents)]
        context = "\n".join(retrieved_docs) if retrieved_docs else "Nenhum contexto disponível."

        # 2. Geração (Generation)
        prompt = f'''Use o seguinte contexto para responder à pergunta. Se o contexto não contiver a resposta, diga que não sabe.

Contexto:
{context}

Pergunta: {item.question}

Resposta:'''

        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 150,
                "temperature": 0.7,
                "return_full_text": False,
            },
        }

        response = requests.post(INFERENCE_API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Lança exceção para respostas de erro HTTP

        data = response.json()
        answer = data[0]["generated_text"].strip() if data and isinstance(data, list) else "Não foi possível gerar uma resposta."

        return {"answer": answer, "context": context}
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Erro ao comunicar com a Hugging Face API: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a pergunta: {e}")

@app.get("/", summary="Status da API")
def read_root():
    '''
    Endpoint raiz para verificar o status da API.
    '''
    return {
        "status": "online",
        "indexed_documents": index.ntotal,
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": HF_MODEL,
    }

# --- Para rodar localmente --- #
if __name__ == "__main__":
    import uvicorn
    print("Iniciando a aplicação FastAPI com uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
