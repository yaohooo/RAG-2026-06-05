import streamlit as st
from openai import OpenAI
import re
import pandas as pd
import numpy as np
import math

# 导入自定义模块
from config import (
    DEFAULT_MODEL, SUPPORTED_MODELS, EMBEDDING_MODEL, ALIYUN_BASE_URL,
    DEFAULT_RETRIEVE_K, DEFAULT_VECTOR_WEIGHT, DEFAULT_MIN_SCORE, DEFAULT_TEMPERATURE
)
from vector_store import HybridVectorStore
from text_utils import extract_text_from_file, split_text_semantic

# ======================== 页面初始化 ========================
st.set_page_config(page_title="百炼 Qwen RAG 助手", layout="wide", page_icon="🤖")
st.title("阿里百炼 Qwen - 智能文档 RAG 系统")

# 初始化 Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = HybridVectorStore()
if "csv_dataframes" not in st.session_state:
    st.session_state.csv_dataframes = {}
if "prev_uploaded_names" not in st.session_state:
    st.session_state.prev_uploaded_names = []

# ======================== 侧边栏配置 ========================
with st.sidebar:
    st.header("⚙️ 1. 平台配置")
    api_key = st.text_input("输入阿里百炼 API Key", type="password",
                            help="请从阿里云百炼控制台获取您的 API Key")
    model_name = st.selectbox("选择千问大模型", SUPPORTED_MODELS, index=SUPPORTED_MODELS.index(DEFAULT_MODEL))

    st.write("---")
    st.header("📁 2. 知识库构建")
    uploaded_files = st.file_uploader(
        "上传参考文件 (支持 txt, md, pdf, csv, svg)",
        accept_multiple_files=True
    )

    # 检测文件列表变化，自动构建/更新知识库
    uploaded_names = [f.name for f in uploaded_files] if uploaded_files else []
    if uploaded_files and uploaded_names != st.session_state.prev_uploaded_names:
        st.session_state.prev_uploaded_names = uploaded_names
        st.session_state.csv_dataframes.clear()  # 重置CSV数据
        if not api_key:
            st.error("请先在上方填写 API Key 再上传文件！")
        else:
            with st.spinner("检测到文件变更，正在自动构建知识库..."):
                try:
                    client = OpenAI(
                        api_key=api_key,
                        base_url=ALIYUN_BASE_URL
                    )
                    # 重置向量库
                    st.session_state.vector_store = HybridVectorStore()
                    total_chunks = 0

                    for f in uploaded_files:
                        # 提取文本（兼容CSV返回的二元组）
                        extract_result = extract_text_from_file(f)
                        if isinstance(extract_result, tuple):
                            raw_text, df = extract_result
                            st.session_state.csv_dataframes[f.name] = df
                        else:
                            raw_text = extract_result

                        if not raw_text.strip():
                            continue

                        # 拆分文本块
                        chunks = split_text_semantic(raw_text)
                        total_chunks += len(chunks)

                        # 批量生成嵌入（每10个块一批）
                        for i in range(0, len(chunks), 10):
                            batch_chunks = chunks[i:i+10]
                            emb_res = client.embeddings.create(
                                model=EMBEDDING_MODEL,
                                input=batch_chunks
                            )
                            # 添加到向量库
                            for chunk, emb_data in zip(batch_chunks, emb_res.data):
                                st.session_state.vector_store.add_chunk(
                                    chunk, emb_data.embedding, source=f.name
                                )

                    st.success(f"✅ 自动构建成功！{len(uploaded_files)} 个文件，{total_chunks} 个语义块。")
                except Exception as e:
                    st.error(f"构建知识库失败：{str(e)}")

    # 显示当前知识库大小
    if st.session_state.vector_store.count > 0:
        st.info(f"📊 当前知识库：{st.session_state.vector_store.count} 个文本块")

# ======================== 聊天交互区域 ========================
# 显示历史聊天记录
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # 展示检索来源
        if "sources" in message:
            with st.expander("📄 查看检索来源"):
                for s in message["sources"]:
                    st.markdown(f"- **{s['source']}** (相关度：{s['score']:.2f})\n> {s['text'][:200]}...")
        # 展示表格查询结果
        if "csv_result" in message:
            with st.expander("📊 查看表格查询结果"):
                st.markdown(message["csv_result"])

# 用户输入问题
if query := st.chat_input("基于上传的文件向助手提问..."):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # 校验前置条件
    if not api_key:
        st.error("请在左侧边栏配置 API Key 后再进行提问！")
    elif st.session_state.vector_store.count == 0:
        st.warning("请先在左侧边栏上传文件并构建知识库！")
    else:
        # 初始化OpenAI客户端
        client = OpenAI(
            api_key=api_key,
            base_url=ALIYUN_BASE_URL
        )

        with st.chat_message("assistant"):
            with st.spinner("正在检索文档并组织回答..."):
                csv_context = ""
                csv_result_display = ""
                context_chunks = []
                full_response = ""

                try:
                    # 1. 生成查询向量
                    q_emb_res = client.embeddings.create(
                        model=EMBEDDING_MODEL,
                        input=[query]
                    )
                    query_embedding = q_emb_res.data[0].embedding

                    # 2. 混合检索（向量+BM25）
                    context_chunks = st.session_state.vector_store.similarity_search(
                        query=query,
                        query_embedding=query_embedding,
                        k=DEFAULT_RETRIEVE_K,
                        vector_weight=DEFAULT_VECTOR_WEIGHT,
                        min_score=DEFAULT_MIN_SCORE
                    )

                    # 3. Data Agent: CSV表格精确查询
                    if st.session_state.csv_dataframes:
                        try:
                            df_info = ""
                            exec_env = {"pd": pd, "np": np, "math": math}
                            # 注入所有DataFrame到执行环境
                            for name, df in st.session_state.csv_dataframes.items():
                                var_name = "df_" + re.sub(r'\W+', '_', name)
                                exec_env[var_name] = df
                                df_info += f"表名：'{name}', 变量名：`{var_name}`\n"
                                df_info += f"列名：{list(df.columns)}\n"
                                df_info += f"数据样本:\n{df.head(2).to_string()}\n\n"

                            # 生成查询代码
                            code_prompt = f"""你是一个数据分析专家。用户有以下表格数据：
{df_info}
用户的问题是："{query}"
请编写一段纯 Python 代码使用 pandas 回答该问题。
要求：
1. 使用上述提供的变量名访问 DataFrame。
2. 对于数值比较（如大于、小于），请先使用 pd.to_numeric 将对应列转换为数值类型，并处理错误值 (errors='coerce')。
3. 将最终结果赋值给变量 `result`。如果是排序，请使用 sort_values。
4. 只输出纯 Python 代码，绝对不要包含 markdown 代码块标记，不要任何解释。
5. 如果问题不需要查询上述表格数据，请直接输出 "NO_CODE"。"""

                            code_response = client.chat.completions.create(
                                model=model_name,
                                messages=[{"role": "user", "content": code_prompt}],
                                temperature=0
                            )
                            generated_code = code_response.choices[0].message.content.strip()

                            if generated_code != "NO_CODE":
                                # 清理代码格式
                                generated_code = re.sub(r'^```python\s*', '', generated_code)
                                generated_code = re.sub(r'^```\s*', '', generated_code)
                                generated_code = re.sub(r'\s*```$', '', generated_code).strip()

                                # 执行代码（隔离环境）
                                exec(generated_code, {"__builtins__": {}}, exec_env)
                                result = exec_env.get("result")

                                if result is not None:
                                    if isinstance(result, pd.DataFrame):
                                        csv_context = f"\n\n[表格精确查询结果] (共匹配 {len(result)} 条数据):\n{result.to_string()}"
                                        csv_result_display = csv_context
                                    else:
                                        csv_context = f"\n\n[表格查询结果]:\n{str(result)}"
                                        csv_result_display = csv_context
                        except Exception as e:
                            st.warning(f"表格自动分析失败，将仅使用常规文本检索：{str(e)}")

                    # 4. 构建增强Prompt
                    context_text = ""
                    # 优先添加表格查询结果
                    if csv_context:
                        context_text += csv_context + "\n"
                    # 添加文本检索结果
                    if context_chunks:
                        context_parts = []
                        for i, c in enumerate(context_chunks):
                            context_parts.append(
                                f"[资料{i+1}] (来源：{c['source']}, 相关度：{c['score']:.2f})\n{c['text']}"
                            )
                        context_text += "\n\n".join(context_parts)

                    # 组装最终Prompt
                    if context_text.strip():
                        prompt = f"""你是一个专业的文档分析助手。请根据以下参考资料回答用户问题。

## 回答规则
1. **严格依据资料**：只使用参考资料中的信息回答，不要编造。
2. **标注来源**：每个关键信息点后标注 [资料 X]。
3. **诚实告知**：如果资料中没有相关信息，明确说明"参考资料中未提及此内容"。
4. **结构化输出**：使用分点、分段的方式组织回答，便于阅读。
5. **综合分析**：如果多个资料涉及同一主题，请综合整理而非简单罗列。
6. **精确引用**：如果参考资料中包含"表格精确查询结果"，请以该结果为准进行回答，不要遗漏任何条目。

## 参考资料
{context_text}

## 用户问题
{query}

请回答："""
                    else:
                        prompt = f"""你是一个专业的文档分析助手。用户的问题在知识库中未找到相关内容。

请诚实告知用户："在上传的文档中没有找到与您的问题相关的内容，请尝试：
1. 更换更具体的关键词
2. 确认相关内容是否已包含在上传的文件中"

用户问题：{query}"""

                    # 5. 流式生成回答
                    stream = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        stream=True,
                        temperature=DEFAULT_TEMPERATURE
                    )

                    response_placeholder = st.empty()
                    full_response = ""
                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            response_placeholder.markdown(full_response + "▌")
                    response_placeholder.markdown(full_response)

                except Exception as e:
                    st.error(f"请求失败：{str(e)}")
                    if not full_response:
                        full_response = f"处理请求时发生错误：{str(e)}"

                # 6. 保存回答到会话
                msg = {"role": "assistant", "content": full_response}
                if context_chunks:
                    msg["sources"] = context_chunks
                if csv_result_display:
                    msg["csv_result"] = csv_result_display
                st.session_state.messages.append(msg)

                # 展示检索来源（折叠面板）
                if context_chunks:
                    with st.expander(f"📄 共检索到 {len(context_chunks)} 个相关片段"):
                        for i, c in enumerate(context_chunks):
                            st.markdown(
                                f"**[{i+1}] {c['source']}** "
                                f"(综合：{c['score']:.2f} | "
                                f"向量：{c['vec_score']:.2f} | "
                                f"关键词：{c['bm25_score']:.2f})"
                            )
                            st.caption(c["text"][:300])
                            if i < len(context_chunks) - 1:
                                st.divider()