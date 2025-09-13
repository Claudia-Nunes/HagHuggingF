# Análise dos Requisitos - RAG com Hugging Face

## Objetivo
Criar um exemplo em Python de RAG (Retrieval-Augmented Generation) que conecte no Hugging Face.

## Conceitos Principais do Slide

### RAG (Retrieval-Augmented Generation)
- Combina **recuperação de informação** (retrieval) com **geração de texto** (generation)
- O modelo não depende apenas do que foi treinado; ele busca informações externas (documentos, bases de dados) e depois gera a resposta
- Vantagem: respostas mais precisas e atualizadas, mesmo para assuntos que a LLM não viu durante o treinamento

### Hugging Face Hub
- Plataforma com centenas de modelos pré-treinados
- Acesso via **Transformers** (biblioteca Python) ou **Inference API** (API pronta para uso)
- Permite testar, treinar e integrar modelos de forma rápida

### Exemplo Fornecido
O slide mostra um exemplo básico com:
- **FastAPI** para criar a API
- **sentence-transformers** para embeddings
- **FAISS** para indexação e busca vetorial
- **Hugging Face Inference API** para geração de texto
- Modelo sugerido: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

### Dependências Necessárias
```
fastapi
uvicorn
sentence-transformers
faiss-cpu
requests
```

### Fluxo do Exemplo
1. **Ingestão**: Adicionar textos ao índice FAISS
2. **Pergunta**: Fazer embedding da pergunta
3. **Recuperação**: Buscar contexto mais similar no índice
4. **Geração**: Enviar contexto + pergunta para o modelo Hugging Face
5. **Resposta**: Retornar a resposta gerada

### Pontos de Discussão Mencionados
- Latência e recursos: modelos grandes podem ser lentos
- Custo: Hugging Face oferece API gratuita limitada, mas o modelo local consome GPU/CPU
