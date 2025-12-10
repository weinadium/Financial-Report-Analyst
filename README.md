# Financial Report Analyst Agent (金融研报智能分析助手)

## Project Overview (项目简介)
这是一个基于 **LangChain** + **RAG (检索增强生成)** 技术构建的垂直领域 AI 智能体。
项目旨在解决传统金融研报阅读中“非结构化数据难以提取”与“通用大模型幻觉”的痛点。通过构建本地向量知识库，实现对长文本研报的精准语义检索与自动化量化分析。

> **核心亮点**：针对金融场景优化了 Token 切分策略与 Prompt 提示词工程，支持一键提取“盈利预测”、“风险提示”等关键指标。

## Technical Architecture (技术架构)
* **LLM Core**: ZhipuAI GLM-4 (智谱 AI)
* **Framework**: LangChain (v0.2.x)
* **Embedding**: ZhipuAI Embedding-2
* **Vector Store**: FAISS
* **Frontend**: Streamlit
* **Data Cleaning**: Recursive Character Splitter + Custom Regex Filters

## Key Features (核心功能)

### 1. 智能向量知识库 (RAG Knowledge Base)
-  **批次化处理 (Batch Processing)**: 实现了自定义的分批向量化算法，解决了 API 调用限制 (Error 1210/1214) 问题，支持长文档稳定处理。
-  **数据清洗 (Data Cleaning)**: 内置强力清洗逻辑，自动过滤 PDF 中的乱码与无效空白字符。

### 2. 自动化研报分析 (Automated Analysis) 
-  **盈利预测提取**: 自动识别并结构化输出未来的营收与净利润预测（Markdown 表格）。
-  **风险因素识别**: 提取 Top-3 潜在风险并评估影响等级。
-  **市场情绪打分**: 基于 NLP 对研报措辞进行量化情感分析。

##  Quick Start (快速开始)

1. **Clone the repo**
   ```bash
   git clone [https://github.com/YourUsername/Financial-Report-Analyst.git]

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt

3. Configure API Key Create a .env file and add your ZhipuAI key:
    ```bash
    ZHIPUAI_API_KEY=your_api_key_here

4.Run the application
    ```bash
    streamlit run app.py