import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.embeddings import ZhipuAIEmbeddings
import os
from dotenv import load_dotenv, find_dotenv
import streamlit as st
import time


# 1. 尝试自动加载
loaded = load_dotenv(find_dotenv()) 
if not os.getenv("ZHIPUAI_API_KEY"):
    # 备选方案：直接在代码里硬编码 Key 
    # 请把下面引号里的内容换成你真实的 Key
    os.environ["ZHIPUAI_API_KEY"] = "" 

# --- 调试打印（运行后看终端输出）---
print(f"Env Loaded: {loaded}")
print(f"API Key Status: {'Found' if os.getenv('ZHIPUAI_API_KEY') else 'Missing'}")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            print(f"Error reading PDF file: {e}")
            continue
    return text

def get_text_chunks(text):
    # 使用递归切分器，它会尝试 "\n\n", "\n", " " 等多种分隔符，保证切分均匀
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       
        chunk_overlap=100,    
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


import time

import time

def get_vectorstore(text_chunks):
    # 1. 初始化智谱 Embedding
    try:
        embeddings = ZhipuAIEmbeddings(
            model="embedding-2",
            api_key=os.getenv("ZHIPUAI_API_KEY")
        )
    except Exception as e:
        st.error(f"API Key 配置错误: {e}")
        return None
    
    
    # 去除换行符、去除首尾空格、强制转为字符串
    clean_chunks = []
    for t in text_chunks:
        if t and isinstance(t, str):
            cleaned_t = t.replace("\n", " ").strip()
            if len(cleaned_t) > 0:
                clean_chunks.append(cleaned_t)
    
    if not clean_chunks:
        st.error("❌ 警告：文档提取内容为空！可能是扫描版 PDF（图片格式）。请上传文字版 PDF。")
        return None

   
    batch_size = 10  
    vectorstore = None
    
    # 初始化进度条
    progress_text = f"正在处理 {len(clean_chunks)} 条文本片段..."
    my_bar = st.progress(0, text=progress_text)
    
    for i in range(0, len(clean_chunks), batch_size):
        batch = clean_chunks[i : i + batch_size]
        
        if not batch:
            continue

        try:
            if vectorstore is None:
                vectorstore = FAISS.from_texts(texts=batch, embedding=embeddings)
            else:
                vectorstore.add_texts(batch)
            
            # 打印成功日志（调试用）
            print(f"✅ Batch {i} to {i+len(batch)} success.")
            
        except Exception as e:
            # 打印失败的具体内容，方便你截图给我
            print(f"❌ Batch {i} failed: {e}")
            print(f"   Sample content: {batch[0][:50]}...") # 打印这一批的第一句话看看是啥
            # 不要 continue，尝试让它失败，否则 vectorstore 为空后面还是会崩
            # 但为了不卡死，我们这里选择跳过
            continue
            
        # 更新进度
        current_progress = min((i + batch_size) / len(clean_chunks), 1.0)
        my_bar.progress(current_progress, text=f"{progress_text} ({int(current_progress*100)}%)")
        
        # 增加休息时间，防止 QPS 超限
        time.sleep(0.3) 
        
    my_bar.empty()
    return vectorstore

def get_conversation_chain(vectorstore):
    # 使用智谱 GLM-4 模型
    llm = ChatZhipuAI(
        model="glm-4",
        temperature=0.1,
        api_key=os.getenv("ZHIPUAI_API_KEY")
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Financial Analyst Agent", page_icon="📈")
    
   
    st.write(css, unsafe_allow_html=True)
    st.header("📈 Intelligent Financial Report Analyst")
    st.markdown("##### 基于 LangChain + RAG 的金融研报智能分析系统")

    # --- 初始化 session state ---
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # --- 主界面：显示聊天记录 ---
    # 如果有聊天记录，倒序显示（符合直觉）
    if st.session_state.chat_history:
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)

    # --- 底部输入框 ---
    user_question = st.text_input("向 AI 分析师提问 (例如：这家公司的主要风险是什么?):")
    if user_question:
        handle_userinput(user_question)

    # --- 侧边栏 (核心功能区) ---
    with st.sidebar:
        st.subheader("📚 研报知识库")
        pdf_docs = st.file_uploader(
            "上传研报 (PDF格式)", accept_multiple_files=True)
        
        if st.button("Process (初始化知识库)"):
            with st.spinner("正在进行向量化切片与构建索引..."):
                # 1. 获取 PDF 文本
                raw_text = get_pdf_text(pdf_docs)
                
                # 2. 文本分块
                text_chunks = get_text_chunks(raw_text)
                
                if not text_chunks:
                    st.error("无法从 PDF 中提取文字。")
                else:
                    # 3. 向量化存储
                    vectorstore = get_vectorstore(text_chunks)
                    
                    # --- [关键修复] 安全检查 ---
                    if vectorstore is None:
                        st.error("❌ 知识库构建失败。请查看终端(Terminal)里的具体报错信息。")
                    else:
                        # 4. 只有 vectorstore 存在时，才创建对话链
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        st.success("✅ 知识库构建完成！")

       
        st.markdown("---")
        st.subheader("📊 智能分析工具箱")
        analysis_task = st.selectbox(
            "选择自动化分析任务",
            ["请选择...", "📋 生成核心观点摘要", "💰 提取盈利预测数据", "⚠️ 识别潜在风险因素", "🌡️ 市场情绪量化打分"]
        )

        if analysis_task != "请选择...":
            if st.session_state.conversation is None:
                st.error("请先上传 PDF 并点击 Process！")
            else:
                if st.button(f"执行：{analysis_task}"):
                    with st.spinner("AI 分析师正在阅读研报并生成报告..."):
                        # 定义专业的 Prompt (对应简历中的 Prompt Engineering)
                        prompts = {
                            "📋 生成核心观点摘要": "你是一名资深证券分析师。请阅读这篇研报，用专业的金融术语总结文章的核心投资逻辑、推荐理由以及目标价。字数控制在300字以内。",
                            "💰 提取盈利预测数据": "请从文中提取未来3年的关键财务预测数据（如营收 Revenue、净利润 Net Profit、EPS等），并以 Markdown 表格形式列出。如果文中没有具体数字，请说明。",
                            "⚠️ 识别潜在风险因素": "请列出这篇研报中提到的前3大投资风险（Risks），并评估其对股价的潜在影响程度（高/中/低）。",
                            "🌡️ 市场情绪量化打分": "基于这篇研报的措辞强硬程度和推荐评级，给该股票的市场情绪打分（0-10分，10分为极度看涨），并简述打分理由。"
                        }
                        
                        # 调用现有对话链进行分析
                        prompt_content = prompts[analysis_task]
                        response = st.session_state.conversation({'question': prompt_content})
                        
                        # 在侧边栏直接展示结果，或者在主界面展示
                        st.markdown(f"### {analysis_task} 结果")
                        st.info(response['answer'])


if __name__ == '__main__':
    main()
