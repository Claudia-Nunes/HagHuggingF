# Pesquisa sobre RAG com Hugging Face

## Principais Abordagens Encontradas

### 1. RAG Simples com Ollama (Tutorial Hugging Face)
- **Modelo de Embedding**: CompendiumLabs/bge-base-en-v1.5-gguf
- **Modelo de Linguagem**: bartowski/Llama-3.2-1B-Instruct-GGUF
- **Banco Vetorial**: Implementação simples em memória
- **Similaridade**: Cosine similarity implementada manualmente
- **Limitação**: Usa Ollama, não diretamente Hugging Face API

### 2. RAG Oficial do Hugging Face (Transformers)
- **Modelos**: facebook/rag-sequence-nq, facebook/rag-token-nq
- **Componentes**: RagTokenizer, RagRetriever, RagSequenceForGeneration
- **Dataset**: wiki_dpr (Wikipedia DPR)
- **Índice**: Compressed FAISS index
- **Vantagem**: Integração nativa com Transformers
- **Limitação**: Modelos específicos, menos flexível

### 3. RAG Avançado com LangChain
- **Embeddings**: sentence-transformers (ex: thenlper/gte-small)
- **Banco Vetorial**: FAISS
- **Chunking**: RecursiveCharacterTextSplitter
- **Separadores Markdown**: Hierárquicos para preservar estrutura
- **Chunk Size**: 1000 caracteres com overlap de 100
- **Consideração**: Limite de 512 tokens do modelo de embedding

## Bibliotecas e Dependências Identificadas

### Para RAG com Hugging Face Inference API (baseado no slide):
```
fastapi
uvicorn
sentence-transformers
faiss-cpu
requests
```

### Para RAG Avançado:
```
torch
transformers
accelerate
bitsandbytes
langchain
sentence-transformers
faiss-cpu
datasets
langchain-community
```

## Componentes Essenciais de um Sistema RAG

1. **Modelo de Embeddings**: Converte texto em vetores
   - sentence-transformers/all-MiniLM-L6-v2 (popular)
   - thenlper/gte-small (usado no cookbook)

2. **Banco Vetorial**: Armazena e busca embeddings
   - FAISS (Facebook AI Similarity Search)
   - IndexFlatIP para produto interno

3. **Chunking**: Divisão de documentos
   - RecursiveCharacterTextSplitter
   - Separadores hierárquicos para Markdown

4. **Modelo de Geração**: LLM para resposta final
   - Hugging Face Inference API
   - TinyLlama/TinyLlama-1.1B-Chat-v1.0 (sugerido no slide)

5. **API Framework**: Interface para o sistema
   - FastAPI (sugerido no slide)

## Fluxo Típico de RAG

1. **Indexação**:
   - Carregar documentos
   - Dividir em chunks
   - Gerar embeddings
   - Armazenar no banco vetorial

2. **Consulta**:
   - Receber pergunta do usuário
   - Gerar embedding da pergunta
   - Buscar chunks mais similares
   - Construir prompt com contexto
   - Gerar resposta com LLM

## Considerações de Performance

- **Chunk Size**: Balancear entre contexto e precisão
- **Top-k**: Número de chunks recuperados (3-5 típico)
- **Overlap**: Evitar corte de ideias (10% do chunk size)
- **Max Sequence Length**: Respeitar limite do modelo de embedding
- **Lost-in-the-middle**: Não sobrecarregar o LLM com muito contexto
