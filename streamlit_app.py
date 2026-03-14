"""
Streamlit UI for RAG Repo Copilot.
Provides a visual interface for:
  1. Submitting a GitHub repo URL for ingestion
  2. Asking natural-language questions about the code
  3. Viewing answers with cited source code
  4. Multi-language support (English / 中文)
"""

import streamlit as st
import requests
import os

# ---- Configuration ----

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# ---- Multi-language support ----

TRANSLATIONS = {
    "en": {
        "page_title": "RAG Repo Copilot",
        "page_subtitle": "AI-powered Code Repository Q&A System",
        "sidebar_title": "Settings",
        "language_label": "Language / 语言",
        "repo_section": "1. Submit a Repository",
        "repo_input": "GitHub Repository URL",
        "repo_placeholder": "https://github.com/user/repo",
        "repo_button": "Ingest Repository",
        "repo_ingesting": "Cloning and processing repository...",
        "repo_success": "Repository ingested successfully!",
        "repo_files": "Files found",
        "repo_chunks": "Code chunks created",
        "repo_embedded": "Chunks embedded",
        "repo_error": "Error ingesting repository",
        "ask_section": "2. Ask a Question",
        "ask_input": "Your question about the code",
        "ask_placeholder": "How does authentication work?",
        "ask_button": "Ask",
        "ask_thinking": "Searching and generating answer...",
        "ask_answer": "Answer",
        "ask_sources": "Source Code References",
        "ask_source_file": "File",
        "ask_source_name": "Function/Class",
        "ask_error": "Error getting answer",
        "search_method": "Search method",
        "sidebar_hybrid": "Use hybrid search (semantic + BM25)",
        "sidebar_rerank": "Use LLM reranking",
        "sidebar_topk": "Number of results",
        "sidebar_about": "About",
        "sidebar_about_text": (
            "RAG Repo Copilot uses OpenAI embeddings, ChromaDB, "
            "hybrid retrieval (semantic + BM25 + RRF), and GPT reranking "
            "to answer questions about any Python codebase."
        ),
        "sidebar_features": "Features",
        "feature_1": "AST-based code chunking",
        "feature_2": "Hybrid search (semantic + BM25)",
        "feature_3": "LLM reranking for accuracy",
        "feature_4": "Multi-language support",
        "no_repo_warning": "Please ingest a repository first.",
        "api_unreachable": "Cannot connect to API server. Make sure it's running.",
    },
    "zh": {
        "page_title": "RAG Repo Copilot",
        "page_subtitle": "AI 驱动的代码仓库问答系统",
        "sidebar_title": "设置",
        "language_label": "Language / 语言",
        "repo_section": "1. 提交代码仓库",
        "repo_input": "GitHub 仓库地址",
        "repo_placeholder": "https://github.com/user/repo",
        "repo_button": "导入仓库",
        "repo_ingesting": "正在克隆并处理仓库...",
        "repo_success": "仓库导入成功！",
        "repo_files": "发现文件数",
        "repo_chunks": "代码块数",
        "repo_embedded": "已嵌入向量数",
        "repo_error": "导入仓库出错",
        "ask_section": "2. 提问",
        "ask_input": "关于代码的问题",
        "ask_placeholder": "这个项目的认证机制是怎么实现的？",
        "ask_button": "提问",
        "ask_thinking": "正在搜索并生成回答...",
        "ask_answer": "回答",
        "ask_sources": "源代码引用",
        "ask_source_file": "文件",
        "ask_source_name": "函数/类",
        "ask_error": "获取回答出错",
        "search_method": "搜索方式",
        "sidebar_hybrid": "使用混合搜索（语义 + BM25）",
        "sidebar_rerank": "使用 LLM 重排序",
        "sidebar_topk": "返回结果数量",
        "sidebar_about": "关于",
        "sidebar_about_text": (
            "RAG Repo Copilot 使用 OpenAI 嵌入、ChromaDB、"
            "混合检索（语义 + BM25 + RRF）和 GPT 重排序，"
            "来回答关于任意 Python 代码仓库的问题。"
        ),
        "sidebar_features": "功能特点",
        "feature_1": "基于 AST 的代码分块",
        "feature_2": "混合搜索（语义 + BM25）",
        "feature_3": "LLM 重排序提升准确性",
        "feature_4": "多语言支持",
        "no_repo_warning": "请先导入一个代码仓库。",
        "api_unreachable": "无法连接到 API 服务器，请确保服务已启动。",
    },
}


def t(key: str) -> str:
    """Get translated string for current language."""
    lang = st.session_state.get("language", "en")
    return TRANSLATIONS[lang].get(key, key)


# ---- Page config ----

st.set_page_config(
    page_title="RAG Repo Copilot",
    page_icon="🔍",
    layout="wide",
)

# ---- Custom CSS ----

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .main-header h1 {
        color: #1E3A5F;
        margin-bottom: 0;
    }
    .main-header p {
        color: #666;
        font-size: 1.1rem;
    }
    .source-card {
        background-color: #f8f9fa;
        border-left: 4px solid #1E3A5F;
        padding: 12px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 16px;
        border-radius: 10px;
        text-align: center;
    }
    .search-badge {
        display: inline-block;
        background-color: #e8f4f8;
        color: #1E3A5F;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ---- Session state init ----

if "repo_ingested" not in st.session_state:
    st.session_state.repo_ingested = False
if "language" not in st.session_state:
    st.session_state.language = "en"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---- Sidebar ----

with st.sidebar:
    st.header(t("sidebar_title"))

    # Language selector
    lang_options = {"English": "en", "中文": "zh"}
    selected_lang = st.selectbox(
        t("language_label"),
        options=list(lang_options.keys()),
        index=0 if st.session_state.language == "en" else 1,
    )
    st.session_state.language = lang_options[selected_lang]

    st.divider()

    # Search settings
    use_hybrid = st.checkbox(t("sidebar_hybrid"), value=True)
    use_rerank = st.checkbox(t("sidebar_rerank"), value=True)
    top_k = st.slider(t("sidebar_topk"), min_value=1, max_value=10, value=5)

    st.divider()

    # About section
    st.subheader(t("sidebar_about"))
    st.write(t("sidebar_about_text"))

    st.subheader(t("sidebar_features"))
    st.markdown(f"- {t('feature_1')}")
    st.markdown(f"- {t('feature_2')}")
    st.markdown(f"- {t('feature_3')}")
    st.markdown(f"- {t('feature_4')}")


# ---- Main content ----

st.markdown(f"""
<div class="main-header">
    <h1>🔍 {t('page_title')}</h1>
    <p>{t('page_subtitle')}</p>
</div>
""", unsafe_allow_html=True)

# ---- Section 1: Repo ingestion ----

st.subheader(t("repo_section"))

col1, col2 = st.columns([4, 1])
with col1:
    repo_url = st.text_input(
        t("repo_input"),
        placeholder=t("repo_placeholder"),
        label_visibility="collapsed",
    )
with col2:
    ingest_clicked = st.button(t("repo_button"), use_container_width=True, type="primary")

if ingest_clicked and repo_url:
    with st.spinner(t("repo_ingesting")):
        try:
            response = requests.post(
                f"{API_BASE}/repos",
                json={"repo_url": repo_url},
                timeout=300,
            )
            if response.status_code == 200:
                data = response.json()
                st.session_state.repo_ingested = True
                st.session_state.chat_history = []  # Reset chat for new repo
                st.success(t("repo_success"))

                # Show metrics
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric(t("repo_files"), data["files_found"])
                with m2:
                    st.metric(t("repo_chunks"), data["chunks_created"])
                with m3:
                    st.metric(t("repo_embedded"), data["chunks_embedded"])
            else:
                error_detail = response.json().get("detail", "Unknown error")
                st.error(f"{t('repo_error')}: {error_detail}")
        except requests.exceptions.ConnectionError:
            st.error(t("api_unreachable"))
        except Exception as e:
            st.error(f"{t('repo_error')}: {str(e)}")

st.divider()

# ---- Section 2: Ask questions ----

st.subheader(t("ask_section"))

if not st.session_state.repo_ingested:
    st.info(t("no_repo_warning"))

# Display chat history
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        if entry.get("sources"):
            with st.expander(f"📄 {t('ask_sources')} ({len(entry['sources'])})"):
                for src in entry["sources"]:
                    st.markdown(f"""
<div class="source-card">
    <strong>{t('ask_source_file')}:</strong> {src.get('file_path', 'N/A')}<br>
    <strong>{t('ask_source_name')}:</strong> {src.get('name', 'N/A')}
</div>
                    """, unsafe_allow_html=True)
                    if src.get("content"):
                        st.code(src["content"][:500], language="python")

# Chat input
question = st.chat_input(t("ask_placeholder"))

if question:
    # Show user message immediately
    with st.chat_message("user"):
        st.write(question)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner(t("ask_thinking")):
            try:
                response = requests.post(
                    f"{API_BASE}/ask",
                    json={
                        "question": question,
                        "top_k": top_k,
                        "use_hybrid": use_hybrid,
                        "use_rerank": use_rerank,
                    },
                    timeout=120,
                )
                if response.status_code == 200:
                    data = response.json()

                    # Display answer
                    st.write(data["answer"])

                    # Search method badge
                    st.markdown(
                        f'<span class="search-badge">{t("search_method")}: {data["search_method"]}</span>',
                        unsafe_allow_html=True,
                    )

                    # Display sources
                    if data.get("sources"):
                        with st.expander(f"📄 {t('ask_sources')} ({len(data['sources'])})"):
                            for src in data["sources"]:
                                st.markdown(f"""
<div class="source-card">
    <strong>{t('ask_source_file')}:</strong> {src.get('file_path', 'N/A')}<br>
    <strong>{t('ask_source_name')}:</strong> {src.get('name', 'N/A')}
</div>
                                """, unsafe_allow_html=True)
                                if src.get("content"):
                                    st.code(src["content"][:500], language="python")

                    # Save to chat history
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": data["answer"],
                        "sources": data.get("sources", []),
                        "search_method": data.get("search_method", ""),
                    })
                else:
                    error_detail = response.json().get("detail", "Unknown error")
                    st.error(f"{t('ask_error')}: {error_detail}")

            except requests.exceptions.ConnectionError:
                st.error(t("api_unreachable"))
            except Exception as e:
                st.error(f"{t('ask_error')}: {str(e)}")
