"""
ui.py — 企业知识库客服 Agent 交互式 Web 界面

基于 Streamlit 构建，将 graph.py 作为黑盒引擎导入。
支持完整的多轮对话、节点流转日志展示、HITL 人工审批拦截与恢复。
"""
from dotenv import load_dotenv
load_dotenv()  # 必须在第一顺位执行！

import uuid

import streamlit as st

from graph import build_graph




# 编译 Agent 图 —— 包含 MemorySaver + interrupt_before
app = build_graph()

# ══════════════════════════════════════════════════════════════
# 1. 初始化会话与记忆 (Session State)
# ══════════════════════════════════════════════════════════════

if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid.uuid4().hex

if "messages" not in st.session_state:
    st.session_state.messages = []

if "logs" not in st.session_state:
    st.session_state.logs = []

# ══════════════════════════════════════════════════════════════
# 2. 侧边栏 —— 引擎流转轨迹
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("引擎流转轨迹")
    if st.session_state.logs:
        for log in st.session_state.logs:
            st.text(log)
    else:
        st.caption("暂无流转日志，发送一条消息开始吧")

# ══════════════════════════════════════════════════════════════
# 3. 主界面 —— 标题 + 聊天记录
# ══════════════════════════════════════════════════════════════

st.title("企业知识库客服 Agent")

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ══════════════════════════════════════════════════════════════
# 4. HITL 检查与恢复逻辑
# ══════════════════════════════════════════════════════════════

config = {"configurable": {"thread_id": st.session_state.thread_id}}
state_snapshot = app.get_state(config)

# 如果图处于挂起状态，显示警告和授权按钮
if state_snapshot.next and "human_approval_node" in state_snapshot.next:
    st.warning("检测到高危敏感操作，系统已挂起等待授权！")

    if st.button("授权管理员执行"):
        # 恢复执行：从挂起点继续运行图
        for event in app.stream(None, config, stream_mode="updates"):
            for node_name, _ in event.items():
                st.session_state.logs.append(f"节点 {node_name} 执行完毕")

        # 提取最终 AI 回复
        final = app.get_state(config)
        msgs = final.values.get("messages", [])
        if msgs:
            last = msgs[-1]
            if hasattr(last, "type") and last.type == "ai":
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": last.content,
                })

        st.rerun()

# ══════════════════════════════════════════════════════════════
# 5. 用户输入与图执行
# ══════════════════════════════════════════════════════════════

prompt = st.chat_input("输入你的问题...")

if prompt:
    # 追加用户消息并立即渲染
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 清空本轮日志
    st.session_state.logs = []

    # ── 启动图流转 ──
    with st.spinner("思考中..."):
        for event in app.stream(
            {"messages": [("user", prompt)]},
            config,
            stream_mode="updates",
        ):
            for node_name, _ in event.items():
                st.session_state.logs.append(f"节点 {node_name} 执行完毕")

    # ── 检查 HITL 挂起 ──
    snap = app.get_state(config)
    if snap.next and "human_approval_node" in snap.next:
        # 被挂起了，不取 AI 回复，让页面刷新后进入 HITL 恢复流程
        st.rerun()

    # ── 没有被挂起，获取最终回复 ──
    final_state = app.get_state(config)
    msgs = final_state.values.get("messages", [])
    if msgs:
        last = msgs[-1]
        if hasattr(last, "type") and last.type == "ai":
            st.session_state.messages.append({
                "role": "assistant",
                "content": last.content,
            })

    st.rerun()
