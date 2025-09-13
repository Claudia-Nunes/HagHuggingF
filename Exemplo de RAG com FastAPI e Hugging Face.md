# Exemplo de RAG com FastAPI e Hugging Face

Este projeto demonstra a implementação de um sistema de **Retrieval-Augmented Generation (RAG)** utilizando Python, FastAPI, FAISS e a Hugging Face Inference API. A aplicação permite indexar documentos de texto e, em seguida, responder a perguntas com base no conteúdo desses documentos, aproveitando o poder de um Large Language Model (LLM).

Este exemplo foi desenvolvido com base nas instruções fornecidas no arquivo `PLN - Aula 3.pdf`.

## Funcionalidades

- **API RESTful**: Construída com FastAPI para uma interface clara e interativa.
- **Indexação de Vetores**: Utiliza `FAISS` para criar um índice de vetores eficiente para busca de similaridade.
- **Embeddings de Sentenças**: Emprega a biblioteca `sentence-transformers` para converter texto em embeddings vetoriais de alta qualidade.
- **Geração de Texto com LLM**: Conecta-se à **Hugging Face Inference API** para gerar respostas contextuais com base nos documentos recuperados.

## Estrutura do Projeto

```
.gitignore
app.py
README.md
requirements.txt
```

- `app.py`: O código principal da aplicação FastAPI, contendo a lógica para RAG.
- `requirements.txt`: Lista de dependências Python necessárias para o projeto.
- `README.md`: Este arquivo, com a documentação do projeto.

## Como Configurar e Executar

Siga os passos abaixo para configurar e executar o projeto em seu ambiente local.

### 1. Pré-requisitos

- Python 3.8 ou superior
- `pip` (gerenciador de pacotes do Python)

### 2. Obtenha um Token da Hugging Face

Para usar a Inference API, você precisa de um token de acesso da Hugging Face.

1.  Acesse o site [Hugging Face](https://huggingface.co/).
2.  Crie uma conta ou faça login.
3.  Vá para **Settings > Access Tokens** no seu perfil.
4.  Crie um novo token com permissão de `leitura` (read).
5.  Copie o token gerado. Ele será usado no próximo passo.

### 3. Instale as Dependências

Clone ou baixe este repositório e, no diretório raiz, instale as dependências listadas no `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Configure a Variável de Ambiente

Crie um arquivo chamado `.env` no diretório raiz do projeto. Este arquivo armazenará seu token da Hugging Face de forma segura. Adicione a seguinte linha ao arquivo, substituindo `SEU_TOKEN_AQUI` pelo token que você copiou:

```
HUGGINGFACE_API_TOKEN="SEU_TOKEN_AQUI"
```

O `app.py` está configurado para carregar esta variável de ambiente automaticamente.

### 5. Execute a Aplicação

Com as dependências instaladas e o token configurado, inicie o servidor FastAPI:

```bash
python app.py
```

O servidor estará disponível em `http://127.0.0.1:8000`.

## Como Usar a API

Após iniciar o servidor, você pode interagir com a API através da documentação interativa do Swagger UI, acessível em `http://127.0.0.1:8000/docs`.

### 1. Adicionar Texto ao Índice (`/ingest`)

Primeiro, você precisa popular seu banco de dados vetorial com o conhecimento que o RAG usará. Use o endpoint `/ingest` para adicionar textos.

**Exemplo de Requisição (usando `curl`):**

```bash
curl -X POST "http://127.0.0.1:8000/ingest" \
-H "Content-Type: application/json" \
-d '{"text": "A capital da França é Paris."}'

curl -X POST "http://127.0.0.1:8000/ingest" \
-H "Content-Type: application/json" \
-d '{"text": "O Brasil é o maior produtor de café do mundo."}'
```

### 2. Fazer uma Pergunta (`/ask`)

Com os documentos indexados, você pode fazer perguntas ao sistema. O RAG encontrará os documentos mais relevantes e os usará como contexto para gerar uma resposta.

**Exemplo de Requisição (usando `curl`):**

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
-H "Content-Type: application/json" \
-d '{"question": "Qual é a capital da França?"}'
```

**Exemplo de Resposta:**

```json
{
  "answer": "A capital da França é Paris.",
  "context": "A capital da França é Paris."
}
```

## Modelo Utilizado

- **Modelo de Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` - Um modelo leve e eficiente para gerar embeddings de alta qualidade.
- **Modelo de Geração (LLM)**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` - Um modelo de linguagem pequeno e rápido, ideal para experimentação e prototipagem.

Ambos os modelos podem ser substituídos por outros disponíveis no Hugging Face Hub, alterando as constantes no topo do arquivo `app.py`.

