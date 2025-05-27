
# 📰 News Research Analyst 🎓🔍

An AI-powered research assistant designed to help stock brokers and financial analysts make smarter, faster, and more informed decisions by leveraging cutting-edge Generative AI technologies.

## 💡 Overview

In today’s fast-paced financial world, keeping up with real-time market news is a critical but overwhelming task. Traditional manual analysis is slow and error-prone. **News Research Analyst** addresses this challenge by integrating OpenAI's LLMs, semantic search, and vector databases into a unified, intelligent financial research tool.

---

## 🚀 Key Features

- 🔍 **Semantic News Search**: Retrieve contextually relevant articles using embeddings and FAISS.
- 📊 **Actionable Insights**: Summarize and analyze news using LLMs for decision-making support.
- 🧠 **Generative AI Integration**: Combines OpenAI's GPT models with real-time news data.
- 🧾 **Chunked Text Processing**: Efficient handling of long documents using `CharacterTextSplitter`.
- 📈 **Streamlit UI**: Simple, interactive web interface for end users.

---

## 🏗️ Architecture

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**:
  - [LangChain](https://www.langchain.com/)
  - [FAISS](https://github.com/facebookresearch/faiss) for vector search
  - [OpenAI GPT models](https://platform.openai.com/)
- **Data Pipeline**:
  - Web scraping / open API ingestion
  - Text chunking & embedding generation
  - Vector storage & semantic retrieval

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/news-research-analyst.git
   cd news-research-analyst

2. **Install dependencies**
   - pip install -r requirements.txt

3. **Set up environment variables
   Create a .env file and add your OpenAI API key:**
   - OPENAI_API_KEY=your_openai_api_key


4. **Run the application**
   - streamlit run app.py

---

## 🧠 How It Works
- Input a URL or query

- Scrape or fetch content

- Split content into manageable chunks

- Generate embeddings using OpenAI

- Index chunks in FAISS

- User asks questions

- Relevant chunks are retrieved and sent to LLM

- LLM returns summarized or analytical output







   
