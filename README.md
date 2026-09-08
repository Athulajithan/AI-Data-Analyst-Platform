# 📊 Enterprise AI Data Analyst Platform 10.0

> **Fully automated end-to-end data analysis platform** powered by AI — from raw data ingestion to executive-level insights, dashboards, strategic recommendations, ML forecasting, and a RAG-powered AI chatbot.

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-green)](https://langchain.com)
[![Gemini](https://img.shields.io/badge/Google-Gemini%202.5-orange)](https://ai.google.dev)

---

## 🌟 Features

| Category | Capability |
|---|---|
| **Data Ingestion** | CSV, Excel (XLSX/XLS), Parquet, JSON with PII detection & type inference |
| **Data Profiling** | Data Quality Score (0–100), missing values, duplicates, column statistics |
| **Data Cleaning** | Non-destructive pipeline: dedup, whitespace trim, null standardization, median/mean imputation |
| **Data Validation** | Pre/post cleaning audit grid with transformation log |
| **EDA** | Pearson/Spearman correlations, segmentation, time trends, IQR anomaly detection |
| **Statistical Analysis** | Welch's t-test, Chi-Square, ANOVA with H0/H1 + p-value reporting |
| **KPI Engine** | Sales & Revenue, Customer metrics, Operational Quality KPIs |
| **Visualization** | Plotly auto-recommendation engine (6+ chart types), Custom Chart Studio |
| **AI Text-to-SQL** | NLP → SQL query generation + read-only SQLite execution |
| **ML & Forecasting** | AutoML Random Forest, linear trend forecasting with 95% CI, SHAP XAI |
| **Insights** | 4-level evidence-based AI insights (Descriptive → Prescriptive) |
| **Recommendations** | Strategic framework: PROBLEM / GOAL / HOW / IMPACT / NEXT STEP |
| **🤖 RAG Chatbot** | Retrieval-Augmented Generation chatbot grounded in your actual dataset analysis |
| **Export Center** | Cleaned CSV, Markdown report, Standalone HTML Dashboard, Printable PDF Briefing |

---

## 🏗️ Architecture

```
Raw Data (CSV/Excel/Parquet/JSON)
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                     DATA ENGINE                              │
│  DataLoader → DataProfiler → DataCleaner → DataValidator    │
│  → EDAEngine → StatisticalAnalyzer → KPIEngine              │
│  → VisualizationEngine → ForecastingEngine → MLEngine       │
└─────────────────────────────────┬───────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│                     AGENT CORE                               │
│  InsightGenerator → RecommendationEngine → ReportBuilder    │
│  TextToSQLAssistant → RAGChatbot (ChromaDB + Gemini)        │
└─────────────────────────────────┬───────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│                     EXPORT CENTER                            │
│  CSV · Excel · Markdown Report · HTML Dashboard · PDF       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🤖 RAG Chatbot Architecture

```
Your Question
     │
     ▼
Query Embedding (Google text-embedding-004)
     │
     ▼
ChromaDB Vector Search (MMR, Top-5 docs)
     │ ← Retrieves from: EDA · KPIs · Insights · Recommendations · Full Report
     ▼
Gemini 2.5-Flash LLM + Conversation Memory (8 turns)
     │
     ▼
Answer (with source attribution)

Fallback: BM25 Keyword Retriever → Deterministic text search (no API key needed)
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/ai-data-analyst-platform.git
cd ai-data-analyst-platform
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Set Gemini API Key
Get a free key at [Google AI Studio](https://makersuite.google.com/app/apikey), then either:
```bash
# Option A: Environment variable
set GEMINI_API_KEY=your_key_here   # Windows
export GEMINI_API_KEY=your_key_here  # Linux/Mac

# Option B: Create .env file
echo GEMINI_API_KEY=your_key_here > .env
```
> The platform works **without a Gemini key** using deterministic AI — add it to unlock full AI-powered insights, natural language chatbot, and Text-to-SQL intelligence.

### 4. Launch the Platform
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
├── app.py                          # Main Streamlit web application (7 tabs)
├── cli.py                          # Headless CLI pipeline
├── requirements.txt
├── sample_data/
│   └── sample_ecommerce.csv        # Sample dataset for demo
├── src/
│   ├── data_engine/
│   │   ├── loader.py               # Multi-format ingestion + PII detection
│   │   ├── profiler.py             # Data Quality Score computation
│   │   ├── cleaner.py              # Non-destructive cleaning pipeline
│   │   ├── validator.py            # Pre/post cleaning audit
│   │   ├── eda.py                  # EDA: correlations, anomalies, trends
│   │   ├── statistical_analyzer.py # Hypothesis testing suite
│   │   ├── kpi_engine.py           # Sales / Customer / Ops KPI calculators
│   │   ├── viz_engine.py           # Plotly auto-recommendation engine
│   │   ├── ml_engine.py            # AutoML Random Forest baseline
│   │   ├── forecasting_engine.py   # Time-series + SHAP XAI
│   │   ├── sql_engine.py           # Read-only SQLite engine
│   │   ├── db_connectors.py        # Multi-DB connector
│   │   └── duckdb_accelerator.py   # DuckDB / SQLite acceleration
│   ├── agent_core/
│   │   ├── llm_provider.py         # Gemini / OpenAI wrapper
│   │   ├── insight_generator.py    # 4-level evidence-based insights
│   │   ├── recommendation_engine.py # Strategic recommendation framework
│   │   ├── report_builder.py       # 23-section report compiler
│   │   ├── text_to_sql.py          # NLP → SQL assistant
│   │   └── rag_chatbot.py          # RAG chatbot (ChromaDB + Gemini)
│   └── exports/
│       ├── exporter.py             # Export orchestrator
│       ├── dashboard_builder.py    # Standalone HTML dashboard
│       └── pdf_exporter.py         # Printable executive PDF briefing
└── tests/
    └── test_pipeline.py            # Enterprise unit tests (5 tests)
```

---

## 🖥️ Application Tabs

| Tab | Description |
|---|---|
| 🖥️ **Executive Dashboard** | KPI cards, trend charts, correlation heatmap, anomaly maps, data quality |
| 🎨 **Custom Chart Studio** | Build any chart — choose X/Y axes, chart type, color, filters |
| 💬 **AI Text-to-SQL & Query Lab** | Ask in plain English → auto-generated SQL + manual SQL console |
| 🎯 **Insights & Recommendations** | AI-generated + custom strategic recommendation builder |
| 📉 **ML & Forecasting** | AutoML, trend forecasting with 95% CI, SHAP feature attribution |
| 📥 **Export Center** | Download cleaned data, full report, HTML dashboard, PDF briefing |
| 🤖 **AI Data Chatbot (RAG)** | Ask anything about your dataset in natural language |

---

## 📊 Supported Data Formats
- **CSV** (any delimiter)
- **Excel** (`.xlsx`, `.xls`)
- **Parquet**
- **JSON** (records or list of objects)

---

## 🧪 Running Tests
```bash
python -m pytest tests/ -v
```

---

## 🔐 Security & Privacy
- **No data is sent externally** unless you add a Gemini API key for AI features.
- All SQL execution is **read-only** with destructive keyword guardrails.
- PII detection flags sensitive columns for review.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit 1.30+, Plotly 5.15+ |
| Data Processing | Pandas, NumPy, SciPy, Statsmodels |
| Machine Learning | Scikit-learn, XGBoost |
| AI / LLM | Google Gemini 2.5-Flash, LangChain |
| RAG | ChromaDB (vector store), BM25 (keyword fallback) |
| SQL Engine | SQLite (in-memory, read-only) |

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with ❤️ as an enterprise-grade AI Data Analyst automation platform.*
