"""
RAG Chatbot — AI Data Analyst Platform 10.0
============================================
Retrieval-Augmented Generation chatbot grounded in the analyzed dataset context.

Architecture:
  1. DocumentBuilder   — serializes all analysis outputs into LangChain Documents
  2. RAGVectorStore    — embeds + indexes documents using ChromaDB (in-memory)
  3. RAGChatbot        — LangChain RetrievalQA chain + conversational memory
  4. Fallback engine   — keyword-based deterministic answers when no API key present

Embeddings: Google Generative AI Embeddings (gemini-embedding-exp-03-07)
LLM:        ChatGoogleGenerativeAI (gemini-2.5-flash) or deterministic fallback
VectorDB:   Chroma (in-memory, per-session, no persistence required)
"""

from __future__ import annotations

import os
import json
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from src.agent_core.web_search import format_search_summary, search_web

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 ─ DOCUMENT BUILDER
# Converts all analysis artifacts → LangChain Document objects
# ─────────────────────────────────────────────────────────────────────────────

def _safe_str(obj: Any, max_len: int = 8000) -> str:
    """Convert arbitrary object to a clean, truncated string."""
    if obj is None:
        return "Not available."
    try:
        if isinstance(obj, str):
            return obj[:max_len]
        return json.dumps(obj, indent=2, default=str)[:max_len]
    except Exception:
        return str(obj)[:max_len]


def build_rag_documents(
    dataset_name: str,
    metadata: Dict,
    profiling: Dict,
    validation: Dict,
    eda_res: Dict,
    stat_res: Dict,
    kpi_res: Dict,
    insights: List[Dict],
    recommendations: List[Dict],
    report_md: str,
    df_sample_csv: str = "",
) -> List[Any]:
    """
    Build a list of LangChain Document objects from all analysis artifacts.
    Each document has rich metadata for source attribution in chat responses.
    """
    from langchain_core.documents import Document

    docs: List[Document] = []

    # ── 1. Dataset Overview ───────────────────────────────────────────────────
    docs.append(Document(
        page_content=f"""
DATASET OVERVIEW — {dataset_name}
==================================
Shape: {metadata.get('shape', 'N/A')}
Columns: {metadata.get('columns', [])}
Data Types: {_safe_str(metadata.get('dtypes', {}))}
Column Buckets:
  - Numerical: {metadata.get('column_buckets', {}).get('numerical_columns', [])}
  - Categorical: {metadata.get('column_buckets', {}).get('categorical_columns', [])}
  - Datetime: {metadata.get('column_buckets', {}).get('datetime_columns', [])}
  - Text: {metadata.get('column_buckets', {}).get('text_columns', [])}
PII Detected: {metadata.get('pii_detected', 'None')}
Inferred Domain: {metadata.get('inferred_domain', 'General')}
""",
        metadata={"source": "dataset_overview", "dataset": dataset_name}
    ))

    # ── 2. Data Quality & Profiling ───────────────────────────────────────────
    docs.append(Document(
        page_content=f"""
DATA QUALITY PROFILE — {dataset_name}
======================================
Data Quality Score: {profiling.get('data_quality_score', 'N/A')}/100
Missing Values Summary: {_safe_str(profiling.get('missing_counts', {}))}
Duplicate Rows: {profiling.get('duplicate_count', 0)}
Descriptive Statistics:
{_safe_str(profiling.get('descriptive_stats', {}), max_len=4000)}
Column-level Quality:
{_safe_str(profiling.get('column_quality', {}), max_len=3000)}
""",
        metadata={"source": "data_quality", "dataset": dataset_name}
    ))

    # ── 3. Cleaning Validation ────────────────────────────────────────────────
    docs.append(Document(
        page_content=f"""
DATA CLEANING & VALIDATION RESULTS — {dataset_name}
=====================================================
Pre-Clean Row Count: {validation.get('pre_clean', {}).get('rows', 'N/A')}
Post-Clean Row Count: {validation.get('post_clean', {}).get('rows', 'N/A')}
Rows Removed: {validation.get('rows_removed', 0)}
Columns Added: {validation.get('columns_added', [])}
Transformations Applied:
{_safe_str(validation.get('transformations_applied', []), max_len=3000)}
""",
        metadata={"source": "cleaning_validation", "dataset": dataset_name}
    ))

    # ── 4. EDA Results ────────────────────────────────────────────────────────
    docs.append(Document(
        page_content=f"""
EXPLORATORY DATA ANALYSIS (EDA) — {dataset_name}
==================================================
Correlation Matrix (Pearson):
{_safe_str(eda_res.get('correlations', {}).get('pearson', {}), max_len=3000)}
Anomalies / Outliers Detected:
{_safe_str(eda_res.get('anomalies', {}), max_len=2000)}
Segmentation Analysis:
{_safe_str(eda_res.get('segmentation', {}), max_len=2000)}
Time Trends:
{_safe_str(eda_res.get('time_trends', {}), max_len=2000)}
""",
        metadata={"source": "eda", "dataset": dataset_name}
    ))

    # ── 5. Statistical Test Results ───────────────────────────────────────────
    docs.append(Document(
        page_content=f"""
STATISTICAL TEST RESULTS — {dataset_name}
==========================================
{_safe_str(stat_res, max_len=5000)}
""",
        metadata={"source": "statistical_analysis", "dataset": dataset_name}
    ))

    # ── 6. KPI Results ────────────────────────────────────────────────────────
    docs.append(Document(
        page_content=f"""
KEY PERFORMANCE INDICATORS (KPIs) — {dataset_name}
====================================================
{_safe_str(kpi_res, max_len=5000)}
""",
        metadata={"source": "kpis", "dataset": dataset_name}
    ))

    # ── 7. Insights ───────────────────────────────────────────────────────────
    if insights:
        insight_text = "\n\n".join([
            f"[{ins.get('level', 'Insight')}] {ins.get('title', '')}\n"
            f"Finding: {ins.get('finding', '')}\n"
            f"Business Implication: {ins.get('business_implication', '')}\n"
            f"Evidence: {ins.get('evidence', '')}"
            for ins in insights[:20]
        ])
        docs.append(Document(
            page_content=f"""
AI-GENERATED INSIGHTS — {dataset_name}
========================================
{insight_text}
""",
            metadata={"source": "insights", "dataset": dataset_name}
        ))

    # ── 8. Recommendations ───────────────────────────────────────────────────
    if recommendations:
        rec_text = "\n\n".join([
            f"[{r.get('priority', '')}] {r.get('problem', '')}\n"
            f"Problem: {r.get('what_is_the_problem', '')}\n"
            f"Goal: {r.get('what_to_solve', '')}\n"
            f"How: {r.get('how_to_solve', '')}\n"
            f"Impact: {r.get('expected_impact', '')}\n"
            f"Next Step: {r.get('next_step', '')}"
            for r in recommendations[:15]
        ])
        docs.append(Document(
            page_content=f"""
STRATEGIC RECOMMENDATIONS — {dataset_name}
============================================
{rec_text}
""",
            metadata={"source": "recommendations", "dataset": dataset_name}
        ))

    # ── 9. Full Report (chunked at 3000 chars each) ───────────────────────────
    if report_md:
        chunk_size = 3000
        overlap = 200
        for i in range(0, len(report_md), chunk_size - overlap):
            chunk = report_md[i: i + chunk_size]
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "source": "full_report",
                    "dataset": dataset_name,
                    "chunk": i // (chunk_size - overlap)
                }
            ))

    # ── 10. Data Sample ───────────────────────────────────────────────────────
    if df_sample_csv:
        docs.append(Document(
            page_content=f"""
DATA SAMPLE (First 20 Rows) — {dataset_name}
=============================================
{df_sample_csv[:4000]}
""",
            metadata={"source": "data_sample", "dataset": dataset_name}
        ))

    return docs


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 ─ VECTOR STORE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_vector_store(documents: List[Any], gemini_api_key: Optional[str] = None) -> Any:
    """
    Build an in-memory Chroma vector store from documents.
    Uses Google Generative AI Embeddings if API key is available,
    falls back to a simple TF-IDF-style BM25 retriever otherwise.
    """
    if not documents:
        return None

    if gemini_api_key:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            from langchain_chroma import Chroma

            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=gemini_api_key
            )
            # Use a unique collection per dataset to avoid collision
            collection_name = "rag_" + hashlib.md5(
                documents[0].page_content[:100].encode()
            ).hexdigest()[:8]

            vector_store = Chroma.from_documents(
                documents,
                embedding=embeddings,
                collection_name=collection_name,
            )
            return ("chroma", vector_store)
        except Exception as e:
            print(f"[RAG] ChromaDB embedding failed: {e}. Using keyword fallback.")

    # ── Fallback: BM25 keyword retriever (no API key needed) ─────────────────
    try:
        from langchain_community.retrievers import BM25Retriever
        retriever = BM25Retriever.from_documents(documents, k=4)
        return ("bm25", retriever)
    except Exception as e:
        print(f"[RAG] BM25 retriever failed: {e}. Using simple text search.")
        return ("raw", documents)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 ─ RAG CHATBOT ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class RAGChatbot:
    """
    Main RAG Chatbot engine.
    Wraps a Chroma/BM25 retriever + Gemini LLM into a conversational QA interface.
    Maintains in-session chat history for multi-turn conversations.
    """

    SYSTEM_PROMPT = """You are an Expert AI Data Analyst embedded inside a BI analytics platform.
You have access to a detailed analysis report, EDA results, KPIs, statistical tests, and strategic recommendations about the dataset.

Your job is to:
1. Answer questions accurately using ONLY the retrieved context (no hallucination).
2. Provide data-backed, quantitative answers when possible.
3. Flag when information is not in the context rather than inventing answers.
4. Frame insights like a senior data analyst or data scientist would.
5. Be concise, professional, and actionable.

When the user asks about problems, always cite specific numbers, column names, or statistics from the data.
If asked for recommendations, reference the strategic framework (WHAT IS THE PROBLEM / WHAT TO SOLVE / HOW TO SOLVE).
"""

    def __init__(
        self,
        vector_store_result: Any,
        gemini_api_key: Optional[str] = None,
        chat_history: Optional[List] = None,
    ):
        self.vector_store_result = vector_store_result
        self.gemini_api_key = gemini_api_key
        self.chat_history: List[Tuple[str, str]] = chat_history or []
        self._chain = None
        self._retriever = None
        self._raw_docs = None
        self._setup()

    def _setup(self):
        """Initialize the retriever and QA chain."""
        if self.vector_store_result is None:
            return

        store_type, store_obj = self.vector_store_result

        if store_type == "chroma":
            self._retriever = store_obj.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 10}
            )
        elif store_type == "bm25":
            self._retriever = store_obj
        else:
            # raw documents
            self._raw_docs = store_obj

        # Build chain only if Gemini key is available
        if self.gemini_api_key and self._retriever:
            try:
                self._build_gemini_chain()
            except Exception as e:
                print(f"[RAG] Chain build failed: {e}. Using retriever-only mode.")

    def _build_gemini_chain(self):
        """Build LangChain RetrievalQA chain with Gemini."""
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain.chains import ConversationalRetrievalChain
        from langchain.memory import ConversationBufferWindowMemory

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=self.gemini_api_key,
            temperature=0.1,
            convert_system_message_to_human=True,
        )

        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True,
            k=8  # Keep last 8 turns in context
        )

        self._chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self._retriever,
            memory=memory,
            return_source_documents=True,
            output_key="answer",
            verbose=False,
        )

    def _retrieve_context(self, query: str) -> str:
        """Retrieve relevant document chunks for a query."""
        if self._retriever is None and self._raw_docs is None:
            return ""
        try:
            if self._retriever:
                if hasattr(self._retriever, 'get_relevant_documents'):
                    relevant_docs = self._retriever.get_relevant_documents(query)
                elif hasattr(self._retriever, 'invoke'):
                    relevant_docs = self._retriever.invoke(query)
                else:
                    return ""
            else:
                # Fallback: simple keyword search in raw documents
                query_lower = query.lower()
                relevant_docs = [
                    d for d in self._raw_docs
                    if any(word in d.page_content.lower() for word in query_lower.split())
                ][:5]

            return "\n\n---\n\n".join([d.page_content for d in relevant_docs[:5]])
        except Exception as e:
            print(f"[RAG] Retrieval error: {e}")
            return ""

    def _deterministic_answer(self, query: str, context: str) -> str:
        """
        Deterministic keyword-based fallback when no Gemini API key.
        Returns a structured answer based on retrieved context + query matching.
        """
        q = query.lower()
        lines = context.split("\n") if context else []

        # Try to extract relevant lines
        relevant = []
        keywords = [w for w in q.split() if len(w) > 3]
        for line in lines:
            if any(kw in line.lower() for kw in keywords):
                relevant.append(line.strip())

        if relevant:
            answer_lines = relevant[:10]
            answer = "\n".join(answer_lines)
            return (
                f"**Based on the dataset analysis:**\n\n{answer}\n\n"
                f"> 💡 *For richer AI-powered answers, add your GEMINI_API_KEY to the sidebar.*"
            )

        # Generic fallback
        if "quality" in q or "score" in q:
            for line in lines:
                if "quality score" in line.lower() or "data quality" in line.lower():
                    return f"📊 {line.strip()}"

        if "recommend" in q or "problem" in q:
            recs = [l for l in lines if "priority" in l.lower() or "problem" in l.lower()]
            if recs:
                return "**Strategic Recommendations found in analysis:**\n" + "\n".join(recs[:5])

        return (
            "I found relevant context in your dataset analysis, but I need more specific keywords "
            "to provide a precise answer. Please try asking about:\n"
            "- KPIs & Revenue metrics\n"
            "- Data quality issues\n"
            "- Recommendations & next steps\n"
            "- Statistical test results\n"
            "- Specific column names or categories from your data\n\n"
            f"> 💡 *Add your GEMINI_API_KEY for AI-powered natural language answers.*"
        )

    def _gemini_direct_answer(self, query: str, context: str) -> str:
        """Direct Gemini call for answering without LangChain chain (simpler path)."""
        try:
            from google import genai

            client = genai.Client(api_key=self.gemini_api_key)

            # Build conversation history context
            history_text = ""
            if self.chat_history:
                history_text = "\n\nPREVIOUS CONVERSATION:\n"
                for user_msg, ai_msg in self.chat_history[-4:]:
                    history_text += f"User: {user_msg}\nAssistant: {ai_msg}\n\n"

            full_prompt = f"""{self.SYSTEM_PROMPT}

RETRIEVED CONTEXT FROM DATASET ANALYSIS:
==========================================
{context}
==========================================
{history_text}
USER QUESTION: {query}

Provide a clear, data-backed answer. If referencing specific numbers, quote them directly from context.
"""
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt
            )
            return response.text or "I couldn't generate a response. Please try again."
        except Exception as e:
            return f"❌ Gemini API error: {str(e)}"

    def chat(self, user_query: str) -> Tuple[str, List[str]]:
        """
        Process a user query and return (answer, source_list).
        Tries Gemini chain → Gemini direct → deterministic fallback in order.

        Returns:
            answer (str): The AI-generated or deterministic answer
            sources (List[str]): Source document names used for the answer
        """
        if not user_query.strip():
            return "Please ask a question about your dataset.", []

        # Retrieve context first
        context = self._retrieve_context(user_query)
        sources = []

        external_wants = any(term in user_query.lower() for term in [
            "benchmark", "industry", "market trend", "market", "competitor",
            "external", "benchmarking", "industry standard", "best practice",
            "research", "sector", "web", "public data"
        ])
        external_context = ""
        if external_wants:
            results = search_web(user_query, max_results=3, timeout=12)
            if results:
                external_context = format_search_summary(user_query, results, max_items=3)

        # Method 1: LangChain ConversationalRetrievalChain (best quality)
        if self._chain is not None:
            try:
                result = self._chain.invoke({"question": user_query})
                answer = result.get("answer", "")
                source_docs = result.get("source_documents", [])
                sources = list(set([
                    d.metadata.get("source", "analysis")
                    for d in source_docs
                ]))
                if answer:
                    if external_context:
                        answer = f"{answer}\n\n---\n\n**External benchmark scan:**\n{external_context}"
                    self.chat_history.append((user_query, answer))
                    return answer, sources + (["external_research"] if external_context else [])
            except Exception as e:
                print(f"[RAG] Chain invoke failed: {e}. Falling back to direct Gemini.")

        # Method 2: Direct Gemini API call with context
        if self.gemini_api_key and context:
            answer = self._gemini_direct_answer(user_query, context)
            if external_context:
                answer = f"{answer}\n\n---\n\n**External benchmark scan:**\n{external_context}"
            self.chat_history.append((user_query, answer))
            sources = ["full_report", "insights", "kpis"]
            if external_context:
                sources.append("external_research")
            return answer, sources

        # Method 3: Deterministic keyword fallback
        base_answer = self._deterministic_answer(user_query, context)
        if external_context:
            answer = f"{base_answer}\n\n---\n\n**External benchmark scan:**\n{external_context}"
        else:
            answer = base_answer
        self.chat_history.append((user_query, answer))
        return answer, ["dataset_analysis"] + (["external_research"] if external_context else [])

    def get_suggested_questions(self, metadata: Dict) -> List[str]:
        """Generate context-aware suggested questions for this specific dataset."""
        num_cols = metadata.get("column_buckets", {}).get("numerical_columns", [])
        cat_cols = metadata.get("column_buckets", {}).get("categorical_columns", [])
        dt_cols = metadata.get("column_buckets", {}).get("datetime_columns", [])

        questions = [
            "What is the overall data quality score and what are the main issues?",
            "What are the top strategic recommendations from the analysis?",
            "What are the most important KPIs and how do they look?",
            "What anomalies or outliers were detected in the data?",
        ]

        if num_cols:
            questions.append(f"What are the descriptive statistics for {num_cols[0]}?")
            if len(num_cols) >= 2:
                questions.append(f"Is there a significant correlation between {num_cols[0]} and {num_cols[1]}?")

        if cat_cols:
            questions.append(f"Which {cat_cols[0]} category has the highest performance?")

        if dt_cols:
            questions.append(f"What is the sales trend over time based on {dt_cols[0]}?")

        questions += [
            "What data cleaning steps were applied to fix quality issues?",
            "What are the prescriptive recommendations to improve the business?",
            "Which features are most important for predicting the target variable?",
            "What statistical tests were run and what do they reveal?",
        ]

        return questions[:10]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 ─ SESSION-CACHED RAG BUILDER
# Prevents rebuilding the vector store on every Streamlit rerun
# ─────────────────────────────────────────────────────────────────────────────

def build_rag_chatbot_cached(
    dataset_name: str,
    metadata: Dict,
    profiling: Dict,
    validation: Dict,
    eda_res: Dict,
    stat_res: Dict,
    kpi_res: Dict,
    insights: List[Dict],
    recommendations: List[Dict],
    report_md: str,
    df_sample_csv: str,
    gemini_api_key: Optional[str],
) -> RAGChatbot:
    """
    Build (or retrieve from session cache) the RAGChatbot.
    The vector store is only rebuilt when the dataset changes.
    """
    # Build document corpus
    documents = build_rag_documents(
        dataset_name=dataset_name,
        metadata=metadata,
        profiling=profiling,
        validation=validation,
        eda_res=eda_res,
        stat_res=stat_res,
        kpi_res=kpi_res,
        insights=insights,
        recommendations=recommendations,
        report_md=report_md,
        df_sample_csv=df_sample_csv,
    )

    # Build vector store
    vector_store_result = build_vector_store(documents, gemini_api_key=gemini_api_key)

    # Instantiate chatbot
    chatbot = RAGChatbot(
        vector_store_result=vector_store_result,
        gemini_api_key=gemini_api_key,
    )

    return chatbot
