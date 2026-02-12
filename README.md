
# 🏥 Medical Fact-Checker RAG System

A specialized **Retrieval-Augmented Generation (RAG)** pipeline designed to extract, process, and query medical fact sheets from the World Health Organization (WHO). This system mitigates LLM hallucinations by forcing the model to answer questions based strictly on verified, real-time clinical data scraped directly from the source.

---

## 🌟 Key Features

* **Dynamic Hub-Scraping:** Automatically discovers and follows links from the WHO Newsroom to ingest the latest medical fact sheets.
* **Medical Text Normalization:** Intelligent cleaning of HTML to remove scripts, styles, and navigation, focusing purely on clinical content.
* **Context-Aware Chunking:** Implements recursive character splitting with a priority on paragraph and sentence boundaries to maintain medical context.
* **Local Vector Storage:** Uses **FAISS** (Facebook AI Similarity Search) for high-speed similarity search using 768-dimension embeddings.
* **Private Inference:** Powered by **Ollama**, ensuring that medical queries are processed locally without sending data to external APIs.

---

## 🛠️ Tech Stack

* **Orchestration:** [LangChain](https://python.langchain.com/)
* **Local LLM:** `mistral` (via Ollama)
* **Embeddings:** `mxbai-embed-large`
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Scraping:** BeautifulSoup4 & Requests
* **Analysis:** Pandas

---

## 📐 Architecture

The system follows a standard RAG pattern but is optimized for the dense, technical nature of medical documents:

1. **Ingestion:** The system crawls the WHO hub and converts HTML articles into `Document` objects.
2. **Splitting:** Documents are broken into chunks (approx. 800–1000 characters) to fit the embedding model's context window.
3. **Embedding:** Each chunk is converted into a vector representation.
4. **Retrieval:** When a user asks a question, the system finds the top `k` most similar chunks.
5. **Generation:** The LLM receives the question + the retrieved chunks and generates a grounded response.

---

## 📋 Prerequisites

1. **Install Ollama:** [Download here](https://ollama.ai/)
2. **Download Required Models:**
```bash
ollama pull mistral
ollama pull mxbai-embed-large

```



---

## 🔧 Installation & Setup

1. **Clone the Repository:**
```bash
git clone https://github.com/AYK175/Medical-fact-checker-RAG-System-.git
cd Medical-fact-checker-RAG-System-

```


2. **Install Dependencies:**
```bash
pip install langchain-ollama langchain-community langchain-text-splitters \
            faiss-cpu beautifulsoup4 pandas tqdm requests

```



---

## 🚀 Execution

Run the pipeline with the following command:

```bash
python new.py

```

Upon execution, the system will:

* Scan the WHO hub page.
* Build a local vector store.
* Generate answers for a set of clinical evaluation questions.
* Output results to `RAG_results.csv`.

---

## 🧪 Evaluation Results

The system successfully retrieves and answers complex queries such as:

* **Unintended Pregnancy:** "Six out of 10 (61%) unintended pregnancies end in induced abortion."
* **Alcohol Mortality:** "2.6 million deaths were caused by alcohol consumption in 2019."
* **Snakebites:** "Nearly 5 million people are bitten by snakes worldwide every year."

---

## 📁 Project Structure

* `new.py` - The core application logic (Scraper + RAG).
* `RAG_results.csv` - Evaluation output comparing question vs. generated answer.
* `.gitignore` - Configured to ignore `venv/` and local temp files.

---

### Would you like me to generate a `requirements.txt` file content now so you can add that to your repo as well?
