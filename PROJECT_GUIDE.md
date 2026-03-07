# RAG Repo Copilot — 项目指南 & 每日进度

## 这个项目是什么？

一句话：**用户丢一个 GitHub repo 进来，用自然语言提问，系统返回答案 + 相关代码片段。**

比如用户问："这个项目的用户认证是怎么实现的？"
系统会找到相关代码，结合 GPT 生成一段解释，并附上具体的文件和行号。

这个技术叫 **RAG（Retrieval-Augmented Generation）**，是目前 AI 工程面试最热门的话题之一。

---

## 项目结构一览

```
rag-repo-copilot/
│
├── src/                        ← 所有核心代码
│   ├── config.py               ← 全局配置（API key、模型参数等）
│   ├── ingestion/              ← 步骤 1: 加载 repo + 切分代码
│   ├── embedding/              ← 步骤 2: 代码块 → 向量 → 存数据库
│   ├── retrieval/              ← 步骤 3: 用户提问 → 找最相关的代码块
│   └── api/                    ← 步骤 4: HTTP 接口，对外提供服务
│       └── main.py             ← FastAPI 入口
│
├── tests/                      ← 测试代码
├── scripts/                    ← 工具脚本
│
├── requirements.txt            ← Python 依赖包清单
├── .env.example                ← 环境变量模板（API key）
├── .gitignore                  ← Git 忽略规则
└── README.md                   ← 项目说明（面试官第一眼看的）
```

---

## 数据流：一个问题从提出到回答

```
用户: "这个 repo 的数据库连接怎么实现的？"
         │
         ▼
   ┌─ API 层 (FastAPI) ──────────────────────┐
   │  接收 HTTP 请求，解析问题                  │
   └─────────────────────────────────────────┘
         │
         ▼
   ┌─ Retrieval 层 ─────────────────────────┐
   │  把问题变成向量，在数据库里找最相似的代码块  │
   │  方法1: 语义搜索（向量相似度）              │
   │  方法2: 关键词搜索（BM25）                 │
   │  方法3: 两者结合（Hybrid Search）          │
   └─────────────────────────────────────────┘
         │
         ▼
   ┌─ LLM 层 ──────────────────────────────┐
   │  把「问题 + 找到的代码块」一起发给 GPT     │
   │  GPT 生成自然语言答案 + 引用具体代码       │
   └─────────────────────────────────────────┘
         │
         ▼
   返回给用户: 答案 + 相关文件路径 + 代码片段
```

---

## 核心概念速查

| 概念 | 一句话解释 | 在项目里的位置 |
|------|-----------|--------------|
| **RAG** | 先检索相关文档，再让 LLM 基于文档回答，减少幻觉 | 整个项目的核心思路 |
| **Chunk（分块）** | 把一整个文件切成小段，方便检索 | `src/ingestion/` |
| **AST** | 抽象语法树，按代码结构（函数/类）来切分，比按行切更聪明 | `src/ingestion/` |
| **Embedding（向量化）** | 把文本变成一串数字（向量），语义相近的文本向量也相近 | `src/embedding/` |
| **Vector Store（向量数据库）** | 专门存向量的数据库，支持"找最相似的" | Chroma（本地）→ Pinecone（生产） |
| **BM25** | 经典关键词搜索算法，擅长精确匹配 | `src/retrieval/` |
| **Hybrid Search** | 语义搜索 + 关键词搜索结合，取长补短 | `src/retrieval/` |
| **Reranker** | 对搜索结果二次排序，提高准确率 | `src/retrieval/` |
| **FastAPI** | Python Web 框架，自动生成 API 文档 | `src/api/main.py` |
| **LangChain** | LLM 应用开发框架，简化 RAG 流程 | 贯穿整个项目 |

---

## 每日计划 & 进度

### Day 1: 环境搭建 + 项目骨架 ✅

**做了什么：**
- 创建项目目录结构
- 写好 `.gitignore`, `.env.example`, `requirements.txt`
- FastAPI hello world（`src/api/main.py`）
- README 初版

**你还需要做：**
- [ ] GitHub 上创建 `rag-repo-copilot` repo
- [ ] 本地 `git init` → `git push`
- [ ] `pip install -r requirements.txt`
- [ ] `uvicorn src.api.main:app --reload` 确认能跑
- [ ] 访问 `http://localhost:8000/docs` 看到 Swagger 文档

**完成标准：** 浏览器打开 localhost:8000 看到返回的 JSON

---

### Day 2: Repo 加载 + 代码切分（预计 2h）

**目标：** 把一个 GitHub repo 克隆下来，用 AST 按函数/类切成 chunks

**会写的代码：**
- `src/ingestion/loader.py` — 克隆 repo、遍历文件
- `src/ingestion/chunker.py` — AST 切分代码
- `tests/test_ingestion.py` — 测试切分效果

**核心知识点：**
- Python `ast` 模块解析代码
- 为什么按函数/类切比按行切好（保留语义完整性）
- GitPython 库克隆 repo

**完成标准：** 输入一个 repo URL，输出一组代码 chunks（每个包含文件路径、函数名、代码内容）

---

### Day 3: Embedding + 向量存储（预计 2h）

**目标：** 把 Day 2 的 chunks 变成向量，存进 Chroma

**会写的代码：**
- `src/embedding/embedder.py` — 调用 OpenAI embedding API
- `src/embedding/store.py` — Chroma 存储和查询
- `tests/test_embedding.py`

**核心知识点：**
- 什么是 embedding，为什么能做语义搜索
- Chroma 的基本用法（add, query）
- 向量维度、距离度量（cosine similarity）

**完成标准：** 存入向量后，用一个问题能查出相关的代码 chunks

---

### Day 4: LLM 回答生成（预计 2h）

**目标：** 检索到的代码块 + 用户问题 → 发给 GPT → 生成答案

**会写的代码：**
- `src/retrieval/qa_chain.py` — LangChain QA chain
- `src/retrieval/prompts.py` — Prompt 模板
- `tests/test_qa.py`

**核心知识点：**
- Prompt Engineering（如何让 GPT 基于代码回答）
- LangChain 的 RetrievalQA chain
- Temperature、token limit 的含义

**完成标准：** 在命令行里问一个问题，得到带代码引用的答案

---

### Day 5: API 端点 + 端到端串联（预计 2h）

**目标：** 把 Day 2-4 的代码通过 FastAPI 暴露出来

**会写的代码：**
- 完善 `src/api/main.py` — `/repos` 和 `/ask` 端点
- 端到端测试

**核心知识点：**
- FastAPI 的 POST 请求处理
- Pydantic 数据校验
- 异步处理长时间任务

**完成标准：** 用 curl 或 Swagger UI 完成完整流程：提交 repo → 提问 → 得到答案

---

### Day 6: Hybrid Search — 语义 + 关键词（预计 2h）

**目标：** 在纯向量搜索基础上加入 BM25 关键词搜索

**会写的代码：**
- `src/retrieval/hybrid.py` — 融合两种搜索结果
- 对比测试

**核心知识点：**
- BM25 算法原理（TF-IDF 的升级版）
- 为什么需要 hybrid（向量搜索擅长语义，BM25 擅长精确匹配）
- 结果融合策略（Reciprocal Rank Fusion）

**完成标准：** 对比 hybrid vs 纯语义搜索的效果，hybrid 更好

---

### Day 7: Reranking + 评估（预计 2h）

**目标：** 用 reranker 对结果二次排序，建立评估体系

**会写的代码：**
- `src/retrieval/reranker.py`
- `scripts/evaluate.py` — 评估脚本

**核心知识点：**
- Cross-encoder vs Bi-encoder
- 评估指标（MRR, Hit Rate, NDCG）
- 为什么 reranking 能提升效果

**完成标准：** 有数据证明 reranking 提升了回答质量

---

### Day 8: Docker + 部署（预计 2h）

**目标：** 容器化 + 部署到云端

**会写的代码：**
- `Dockerfile`
- `docker-compose.yml`
- 部署配置

**核心知识点：**
- Docker 基础（镜像、容器、端口映射）
- 环境变量管理
- 云部署选项（Railway / Render / AWS）

**完成标准：** 有一个可访问的在线 URL

---

### Day 9: 文档 + 架构图（预计 1.5h）

**目标：** 让 README 达到面试展示水平

**要做的事：**
- 完善 README（架构图、使用示例、性能数据）
- 画系统架构图
- 录一个 demo GIF 或截图

**完成标准：** 面试官看 README 5 秒内能理解项目价值

---

### Day 10: 最终打磨 + 简历更新（预计 1.5h）

**要做的事：**
- 代码清理、添加注释
- 简历上添加项目描述
- 准备面试常见问题的回答

**简历描述参考：**
> Built a RAG-based code Q&A system using LangChain, OpenAI, and ChromaDB. Implemented hybrid search (semantic + BM25) with reranking, achieving X% improvement in retrieval accuracy. Deployed via Docker with FastAPI backend.

---

## 面试高频问题预览

这些问题在做项目的过程中自然会理解，先列在这里，最后一天回来复习：

1. **为什么用 RAG 而不是直接把代码全丢给 LLM？** → token 限制 + 成本 + 准确性
2. **你的 chunking 策略是什么？为什么？** → AST-based，保留代码语义完整性
3. **语义搜索 vs 关键词搜索各自的优劣？** → 语义理解同义词，关键词精确匹配
4. **什么是 embedding？** → 把文本映射到高维空间，语义相近的点距离近
5. **Reranking 为什么有效？** → Cross-encoder 能看到 query 和 document 的交互
6. **如何评估 RAG 系统的效果？** → MRR, Hit Rate, NDCG + 人工评估
7. **生产环境会怎么改进？** → Pinecone 替换 Chroma, 缓存, 异步处理, 监控
