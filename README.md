**Soil Analyzer — AI-Powered Soil Report Interpreter**

An intelligent web app that analyzes soil test reports (PDFs) using LangChain, Google Gemini, and Chroma Vector Database.
It extracts, indexes, and interprets soil data — providing quality assessment, crop recommendations, and improvement suggestions.

**Features**

📄 PDF Upload & Text Extraction – Upload your soil test reports directly.
🧠 LLM-based Soil Analysis – Uses Google Gemini (via LangChain) to interpret test data.
🔎 Vector Database Integration – Stores all uploaded reports in Chroma DB for semantic retrieval.
💬 Contextual RAG (Retrieval-Augmented Generation) – Enriches current analysis using similar historical data.
🧾 Structured Output – Results are parsed into a Pydantic schema for consistent display.
⚙️ Streamlit Interface – Interactive, easy-to-use web UI.

**Tech Stack**
| Component                | Technology                                                          |
| ------------------------ | ------------------------------------------------------------------- |
| **Language Model (LLM)** | [Google Gemini](https://ai.google.dev) via `langchain-google-genai` |
| **Framework**            | [LangChain](https://python.langchain.com)                           |
| **Vector Database**      | [Chroma](https://www.trychroma.com)                                 |
| **Embeddings**           | `text-embedding-004`                                                |
| **Frontend**             | [Streamlit](https://streamlit.io)                                   |
| **Parser**               | PydanticOutputParser                                                |
| **File Processing**      | PyMuPDF (for PDF text extraction)                                   |

**Project Structure**
SoilAnalyzer/
│
├── main.py               # Streamlit entry point (UI + orchestration)
├── model.py              # LLM configuration (Gemini setup)
├── prompt.py             # Prompt template + Pydantic schema
├── db.py                 # Vector DB functions (Chroma setup, chunking, retrieval)
├── requirements.txt      # Python dependencies
├── .gitignore            # Ignored files (venv, chroma_db, etc.)
└── chroma_db/            # Local Chroma vector database (auto-created)

**Setup Instructions**
1. Clone the repository
git clone https://github.com/PS-Saiyad-Uveshali/SoilAnalyzer.git
cd SoilAnalyzer

2. Install dependencies
pip install -r requirements.txt

3. Add your API key
  Create a Streamlit secrets file:
    mkdir -p .streamlit
  Create .streamlit/secrets.toml:
    GOOGLE_API_KEY = "your-google-genai-api-key-here"

**How It Works**
🧾 Step 1 — Upload PDF

User uploads a soil test report (PDF).
PyMuPDF extracts the text page-by-page.

🔍 Step 2 — Index into Vector DB

Text is chunked (≈1200 chars overlap 200).
Embeddings are created using Google’s text-embedding-004.
Stored locally in ChromaDB (chroma_db/) with metadata (source, page).

💡 Step 3 — Retrieval + LLM Analysis

When analyzing, the app retrieves the top relevant chunks from the vector DB.
The retrieved text becomes context.
The LangChain pipeline passes both soil text and context into Gemini.

🧩 Step 4 — Structured Output

**Gemini’s raw output is parsed via a Pydantic model:**
class SoilAnalysis(BaseModel):
    quality: str = Field(..., description="Brief summary of soil quality")
    recommended_crops: list[str] = Field(..., description="List of suitable crops")
    suggestions: str = Field(..., description="Detailed suggestions to improve soil")

**The response is displayed cleanly on Streamlit with sections for:**
🧪 Soil Quality
🌾 Recommended Crops
💡 Suggestions

**Run the App:**
streamlit run main.py
