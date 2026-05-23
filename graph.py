"""
企业知识库客服 Agent — 图拓扑骨架 (Graph)

本文件定义了 LangGraph 的完整 DAG 拓扑，包括：
1. 5 个节点（路由 / 检索 / MCP / 生成 / 反思）—— 其中 router/generate/reflection 已接入真实 LLM
2. 2 条条件边（路由分发 + 纠错循环）
3. 图的编译与测试入口

注意：AgentState 是 Pydantic BaseModel，LangGraph 传入的是对象实例，
必须用 state.xxx 属性访问，不能用 state["xxx"] 字典语法。
"""

import os
import sys
import asyncio
import time
from pathlib import Path

# 确保 agent-demo 目录在 sys.path 中，方便直接 python graph.py 运行
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
# .env 放在 .venv/ 目录下
load_dotenv(Path(__file__).resolve().parent / ".venv" / ".env")

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from mcp import StdioServerParameters, ClientSession
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.client import load_mcp_tools

from schema import AgentState, RouterDecision, ReflectionResult, RAGSearchInput, MCPQueryInput

# ══════════════════════════════════════════════════════════════
# 0. LLM 初始化
# ══════════════════════════════════════════════════════════════
# ChatOpenAI 自动从环境变量读取 OPENAI_API_KEY 和 OPENAI_BASE_URL
# 如需指定模型名称，设置环境变量 LLM_MODEL，默认 gpt-4o-mini

base_llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
    temperature=0,
)

# 通过 with_structured_output 让 LLM 直接返回 Pydantic 对象
# 底层使用 OpenAI function calling 机制，无需手动 parse JSON
# 注意：DeepSeek 等第三方厂商不支持 json_schema 模式，必须显式指定 method="function_calling"
router_llm = base_llm.with_structured_output(RouterDecision, method="function_calling")
reflection_llm = base_llm.with_structured_output(ReflectionResult, method="function_calling")
# 工具参数提取 LLM —— 将用户自然语言转为结构化工具入参
rag_llm = base_llm.with_structured_output(RAGSearchInput, method="function_calling")
mcp_llm = base_llm.with_structured_output(MCPQueryInput, method="function_calling")

# ══════════════════════════════════════════════════════════════
# 1. Dummy 节点实现
# ══════════════════════════════════════════════════════════════
#
# 关键约束：每个节点函数的参数是 AgentState（Pydantic 对象），
# 必须用 state.xxx 读取字段，用 return dict 更新字段。
# 绝对不能写 state.xxx = yyy（Pydantic 模型默认不可变）。
# LangGraph 拿到返回的 dict 后，根据 AgentState 中每个字段的
# reducer 策略（add_messages / operator.add / 直接覆盖）自动 merge。
# ══════════════════════════════════════════════════════════════


def router_node(state: AgentState) -> dict:
    """路由节点 —— 使用 LLM 进行意图分类。

    通过 ChatOpenAI.with_structured_output(RouterDecision) 让 LLM
    直接返回符合 RouterDecision schema 的结构化 JSON。
    LangChain 底层使用 OpenAI function calling 机制完成 JSON→Pydantic 的转换。
    """
    # 提取最后一条用户消息
    user_msg = ""
    for m in reversed(state.messages):
        if hasattr(m, "type") and m.type == "human":
            user_msg = m.content
            break

    system_prompt = (
        "你是一个企业知识库客服的前台路由。"
        "请分析用户的最后一条消息，判断意图，并严格返回 JSON。\n\n"
        "意图分类规则：\n"
        "- rag_search: 用户询问企业内部制度、政策、流程等知识库类问题（如报销规定、请假流程、HR政策、IT支持等）\n"
        "- mcp_query: 用户要求查询具体工单状态、审批进度、系统数据等需要调用外部系统的请求\n"
        "- chitchat: 日常问候、闲聊、感谢等不需要业务数据的对话\n"
        "- escalate_to_human: 用户明确要求转人工服务，或表达强烈不满/投诉\n\n"
        "请在 reasoning 字段中简要说明你的判断依据。"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"用户消息: {user_msg}"),
    ]

    decision: RouterDecision = router_llm.invoke(messages)
    print(
        f"[router_node] 意图={decision.intent}, "
        f"置信度={decision.confidence:.2f}, "
        f"理由={decision.reasoning}"
    )

    # 新一轮对话开始时，重置上一轮的纠错状态，防止 retry_count 积累导致死循环
    return {
        "current_intent": decision.intent,
        "retry_count": 0,
        "reflection_feedback": None,
    }


def rag_node(state: AgentState) -> dict:
    """RAG 检索节点 —— LLM 提取搜索词 + 本地 FAISS 向量检索。

    使用与 ingest.py 完全相同的 BGE 中文 Embedding 模型，
    对 LLM 提取的搜索词做语义检索，返回最相关的知识库片段。
    """
    # ── Step 1: LLM 提取参数 → 结构化 RAGSearchInput ──
    args: RAGSearchInput = rag_llm.invoke(state.messages)
    print(f"[rag_node] [RAG 工具调用] 提取搜索词: '{args.query}', 分类: {args.category}")

    # ── Step 2: 真实 FAISS 语义检索 ──
    # 使用与 ingest.py 完全相同的本地 Embedding 模型
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

    # 加载本地持久化的 FAISS 索引
    vector_store = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )

    # 语义检索，取最相关的 2 条
    docs = vector_store.similarity_search(args.query, k=2)
    real_contexts = [doc.page_content for doc in docs]

    print(f"[rag_node] [真实 RAG] 检索到 {len(real_contexts)} 条知识块")

    return {"retrieved_context": real_contexts}


def mcp_node(state: AgentState) -> dict:
    """MCP 工具调用节点 —— 接入真实 GitHub MCP Server。

    架构：同步外壳 + 异步内核 (asyncio.run)。
    GitHub MCP Server 通过 npx 子进程启动，暴露 search_repositories /
    get_file_contents / list_issues 等工具。LLM 自行决定调用哪个工具。
    """
    # 提取最后一条用户消息
    user_msg = ""
    for m in reversed(state.messages):
        if hasattr(m, "type") and m.type == "human":
            user_msg = m.content
            break

    # ── 异步内核：MCP 生命周期 + LLM Tool Calling ──
    async def run_mcp() -> str:
        # Windows 适配：npx → npx.cmd
        npx_cmd = "npx.cmd" if sys.platform == "win32" else "npx"

        server_params = StdioServerParameters(
            command=npx_cmd,
            args=["-y", "@modelcontextprotocol/server-github"],
            env=os.environ.copy(),  # 传递 GITHUB_PERSONAL_ACCESS_TOKEN
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # 加载 GitHub MCP Server 暴露的所有工具
                tools = await load_mcp_tools(session)
                tool_names = [t.name for t in tools]
                print(f"[mcp_node] 加载 {len(tools)} 个 GitHub 工具: {tool_names}")

                # 将工具绑定到大模型，让 LLM 自行决定调用哪个
                llm_with_tools = base_llm.bind_tools(tools)

                prompt = (
                    f"用户请求: {user_msg}\n"
                    "请使用 GitHub 工具查询相关信息，返回结果给用户。"
                )

                response = await llm_with_tools.ainvoke([
                    SystemMessage(content="你是 GitHub 查询助手，根据用户需求调用合适的 GitHub 工具。"),
                    HumanMessage(content=prompt),
                ])

                # ── 检查是否发起了 tool_call ──
                if hasattr(response, "tool_calls") and response.tool_calls:
                    print(f"[mcp_node] LLM 发起 {len(response.tool_calls)} 个工具调用")

                    tool_results = []
                    for tc in response.tool_calls:
                        tool_name = tc.get("name", "")
                        tool_args = tc.get("args", {})
                        print(f"[mcp_node]   调用: {tool_name}({tool_args})")

                        # 从工具列表中找到匹配的工具并执行
                        for tool in tools:
                            if tool.name == tool_name:
                                try:
                                    raw_result = await tool.ainvoke(tool_args)
                                    # 截断保留前 1000 字符，防止 GitHub 返回过大
                                    result_str = str(raw_result)[:1000]
                                    tool_results.append(
                                        f"[工具: {tool_name}]\n{result_str}"
                                    )
                                except Exception as exc:
                                    tool_results.append(
                                        f"[工具: {tool_name}] 执行失败: {exc}"
                                    )
                                break

                    return "\n\n".join(tool_results)
                else:
                    # LLM 未发起 tool_call，直接返回文本回复
                    return response.content

    # ── 在同步函数中用 asyncio.run 阻塞执行异步内核 ──
    try:
        result = asyncio.run(run_mcp())
    except Exception as e:
        error_msg = f"MCP 调用失败: {type(e).__name__} - {e}"
        print(f"[mcp_node] {error_msg}")
        result = error_msg

    print(f"[mcp_node] 最终结果: {result[:120]}...")
    return {
        "retrieved_context": [f"【GitHub 系统响应】: {result}"],
        "requires_human_approval": False,
    }


def generate_node(state: AgentState) -> dict:
    """生成节点 —— 使用 LLM 基于知识库上下文生成回答。

    关键逻辑：
    1. 将 retrieved_context 作为背景知识注入 system prompt
    2. 如果 reflection_feedback 非空，严肃要求 LLM 修正上一轮错误
    3. 将完整对话历史传给 LLM 以保持上下文连贯

    注意：这里不调用 with_structured_output，直接生成自然语言回复。
    """
    context_list = state.retrieved_context
    feedback = state.reflection_feedback

    # ── 动态构建 System Prompt ──
    system_parts = [
        "你是一个企业知识库客服助手。请根据提供的知识库文档回答用户问题。"
    ]

    if context_list:
        context_block = "\n---\n".join(context_list)
        system_parts.append(f"\n【知识库背景资料】\n{context_block}")
        system_parts.append(
            "\n【要求】\n"
            "- 回答必须严格基于上述背景资料，不得编造信息\n"
            "- 如果背景资料不足以回答用户问题，请如实告知\n"
            "- 回答应简洁、准确、专业"
        )

    if feedback:
        # 重写轮：将上一轮质检反馈严肃注入 system prompt
        system_parts.append(
            f"\n【重要：上一轮回答被打回！】\n"
            f"质检员指出以下问题：{feedback}\n"
            f"请务必修正此错误后重新回答！"
        )
        print(f"[generate_node] 重写模式 —— 反馈: {feedback[:80]}...")
    else:
        print(f"[generate_node] 首次生成模式 —— {len(context_list)} 条上下文")

    system_msg = "\n".join(system_parts)

    # 组装完整消息列表：[SystemMessage] + 对话历史
    # 对话历史中已包含用户消息和之前的所有 AI 回复
    messages = [SystemMessage(content=system_msg)] + list(state.messages)

    response = base_llm.invoke(messages)
    print(f"[generate_node] LLM 生成完成 ({len(response.content)} 字符)")

    # response 本身就是 AIMessage，直接放入列表
    return {"messages": [response]}


def reflection_node(state: AgentState) -> dict:
    """反思节点 —— 使用 LLM 对生成的回答进行质量审查。

    通过 ChatOpenAI.with_structured_output(ReflectionResult) 获取结构化的审查结果。
    检查点包括：事实准确性、业务逻辑正确性、是否听从了之前的打回建议。
    """
    # 获取最后一条助手消息作为被审查对象
    last_ai_msg = None
    for m in reversed(state.messages):
        if hasattr(m, "type") and m.type == "ai":
            last_ai_msg = m.content
            break

    if not last_ai_msg:
        # 没有 AI 消息可审查，直接通过
        print("[reflection_node] 无助手消息可审查，自动通过")
        return {
            "reflection_feedback": None,
            "retry_count": state.retry_count + 1,
        }

    previous_feedback = state.reflection_feedback
    context_list = state.retrieved_context

    # ── 构建背景资料块 ──
    # 将检索到的知识库文档注入审查 prompt，消除质检员的信息不对称
    if context_list:
        context_block = "\n---\n".join(context_list)
        context_section = (
            "\n【底层参考资料 — 请严格依据以下内容审查事实准确性】\n"
            f"{context_block}\n"
            "以上是助手回复所依据的全部背景资料。审查时请逐条对比："
            "助手回复中的每个事实、数字、日期是否与上述资料严格一致？"
            "如果资料中没有提到的内容，助手回复中包含即视为编造。\n"
        )
    else:
        context_section = "\n【注意】本轮没有知识库背景资料，请仅基于常识和逻辑审查回复。\n"

    # ── 动态构建审查 System Prompt ──
    system_prompt = (
        "你是一个严苛的质检员。请检查助手刚刚生成的最后一条回复。\n"
        + context_section +
        "\n审查标准：\n"
        "1. 事实核查：回复中的每个事实是否在上述背景资料中有明确依据？不与资料矛盾？\n"
        "2. 业务逻辑：回复是否合理、准确、完整地解决了用户的问题？\n"
        "3. 上下文一致性：回复中的数字、日期、政策条款是否与背景资料严格一致？\n"
    )

    if previous_feedback:
        system_prompt += (
            f"4. 打回修正检查：上一轮你的打回意见是：「{previous_feedback}」"
            "请重点检查本次回复是否已按此意见修正。如果没有修正，必须再次指出。"
        )
    else:
        system_prompt += "4. 这是首次审查，请全面检查回复质量。"

    system_prompt += "\n\n请不要脱离背景资料乱报错。请在 feedback 字段中给出具体的修改建议，并设置正确的 is_valid 值。"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"【待审查的助手回复】\n{last_ai_msg}"),
    ]

    result: ReflectionResult = reflection_llm.invoke(messages)
    status = "通过" if result.is_valid else "驳回"
    print(f"[reflection_node] 第 {state.retry_count + 1} 次审查 → {status}: {result.feedback}")

    # 关键：根据 is_valid 决定是否将 feedback 写入状态
    # - is_valid=True  → feedback 设为 None，should_continue 会据此直接走 END
    # - is_valid=False → feedback 写入状态，should_continue 据此打回 generate_node 重写
    return {
        "reflection_feedback": result.feedback if not result.is_valid else None,
        "retry_count": state.retry_count + 1,
    }


def human_approval_node(state: AgentState) -> dict:
    """人工审批节点 —— 模拟人类管理员点击"同意"按钮。

    在生产环境中，这个节点会被 interrupt_before 挂起，
    等待人类在管理后台审查通过后才继续执行。
    此处模拟：直接打印授权通过，并将审批标志置为 False。
    """
    print("[human_approval_node] [系统挂起] 人类管理员已授权通过！")
    return {"requires_human_approval": False}


# ══════════════════════════════════════════════════════════════
# 2. 条件边函数
# ══════════════════════════════════════════════════════════════
#
# 条件边函数的签名: (state: AgentState) -> str
# 返回的是图中下一个节点的名字（必须是 add_node 时注册的名字）
# 或者 LangGraph 内置常量 START / END
# ══════════════════════════════════════════════════════════════


def check_approval_needed(state: AgentState) -> str:
    """MCP 后置条件边 —— 检查是否需要人工审批。

    在 mcp_node 之后被调用：
    - 如果 mcp_node 将 requires_human_approval 设为 True，
      则路由到 human_approval_node（会被 interrupt_before 挂起）。
    - 如果为 False（管理员直接放行），直接进入 generate_node。
    """
    if state.requires_human_approval:
        print("[check_approval_needed] 需要人工审批 → human_approval_node")
        return "human_approval_node"
    else:
        print("[check_approval_needed] 无需审批 → generate_node")
        return "generate_node"


def route_by_intent(state: AgentState) -> str:
    """路由条件边 —— 根据 current_intent 决定下一步走到哪个节点。

    这个函数在 router_node 执行完后被 LangGraph 调用。
    返回值必须是图中已注册的节点名字（字符串），或者是 END。
    """
    intent = state.current_intent or "chitchat"

    route_map = {
        "rag_search": "rag_node",       # 知识库问题 → 先检索再生成
        "mcp_query": "mcp_node",        # 工单查询 → 走 MCP 系统调用
        "chitchat": "generate_node",    # 闲聊 → 直接生成，无需检索
        "escalate_to_human": END,       # 转人工 → 结束，等待人工介入
    }

    next_node = route_map.get(intent, "generate_node")
    print(f"[route_by_intent] {intent} → {next_node}")
    return next_node


def should_continue(state: AgentState) -> str:
    """纠错循环条件边 —— 决定是结束还是回退到生成节点重写。

    这个函数在 reflection_node 执行完后被调用。
    判定逻辑（按优先级）：
    1. reflection_feedback 为 None → reflection_node 判定 is_valid=True，直接 END
    2. reflection_feedback 为空字符串 → 也视为通过，直接 END
    3. retry_count >= 3 → 超过最大重试次数，强制终止
    4. 以上都不满足 → 存在反馈意见且未达上限，回到 generate_node 重写
    """
    retry_count = state.retry_count
    feedback = state.reflection_feedback

    # 条件 1+2: feedback 为 None 或空 → 审查通过，直接结束
    # （reflection_node 在 is_valid=True 时会将 feedback 设为 None）
    if not feedback:
        print(f"[should_continue] 审查通过 (feedback 为空) → END")
        return END

    # 条件 3: 超过最大重试次数 (3 次)
    if retry_count >= 3:
        print(f"[should_continue] 重试已达上限 ({retry_count} 次) → END")
        return END

    # 条件 4: 存在反馈意见且未超限 → 回退重写
    print(f"[should_continue] 第 {retry_count} 次未通过 → 回到 generate_node 重写")
    return "generate_node"


# ══════════════════════════════════════════════════════════════
# 3. 构建和编译图
# ══════════════════════════════════════════════════════════════


def build_graph() -> StateGraph:
    """构建并编译企业知识库客服 Agent 的状态图。

    图拓扑概览:

                        ┌─────────────────────────────┐
                        │         START                │
                        └─────────────┬────────────────┘
                                      │
                                      ▼
                        ┌─────────────────────────────┐
                        │       router_node            │
                        │   (LLM 意图分类 →             │
                        │    current_intent)           │
                        └─────────────┬────────────────┘
                                      │ route_by_intent
                  ┌───────────────────┼───────────────────┐
                  │                   │                    │
          rag_search            mcp_query            chitchat
                  │                   │                    │
                  ▼                   ▼                    │
        ┌─────────────────┐ ┌─────────────────┐          │
        │    rag_node      │ │    mcp_node      │          │
        │ (检索知识库文档)   │ │ (调用工单系统)    │          │
        └────────┬────────┘ └────────┬────────┘          │
                 │                   │                    │
                 └─────────┬─────────┘                    │
                           │                              │
                           ▼                              │
              ┌─────────────────────────┐                 │
              │    generate_node        │◄────────────────┘
              │ (基于上下文生成回答)      │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │   reflection_node       │
              │ (自我审查 + retry_count) │
              └────────────┬────────────┘
                           │ should_continue
                  ┌────────┴────────┐
                  │                 │
             retry<3 &           通过 or
              未通过             retry>=3
                  │                 │
                  ▼                 ▼
          generate_node           END
          (回退重写)
    """
    # 初始化 StateGraph，传入 AgentState 作为全局状态类型
    graph = StateGraph(AgentState)

    # ── 注册节点 ──
    graph.add_node("router_node", router_node)
    graph.add_node("rag_node", rag_node)
    graph.add_node("mcp_node", mcp_node)
    graph.add_node("generate_node", generate_node)
    graph.add_node("reflection_node", reflection_node)
    graph.add_node("human_approval_node", human_approval_node)

    # ── 注册边 ──

    # 入口边：图启动后第一个执行的节点
    graph.add_edge(START, "router_node")

    # 条件边：路由分发
    # router_node 执行完后，调用 route_by_intent 决定下一步
    # 可能的结果: rag_node / mcp_node / generate_node / END
    graph.add_conditional_edges("router_node", route_by_intent)

    # RAG 检索完成后进入生成节点
    graph.add_edge("rag_node", "generate_node")

    # MCP 节点 → 条件边：根据 requires_human_approval 决定是否需要人工审批
    # [HITL 关键] 如果需要审批，路由到 human_approval_node，
    # 图引擎会在执行前自动挂起（由 interrupt_before 控制）
    graph.add_conditional_edges("mcp_node", check_approval_needed)

    # 人工审批通过后 → 进入生成节点
    graph.add_edge("human_approval_node", "generate_node")

    # 生成节点 → 反思节点
    graph.add_edge("generate_node", "reflection_node")

    # 条件边：纠错循环
    # reflection_node 执行完后，调用 should_continue 决定:
    # - 退回 generate_node 重写
    # - 走向 END 终止
    graph.add_conditional_edges("reflection_node", should_continue)

    # ── 编译图 ──
    # MemorySaver: 内存级检查点持久化，每次节点执行后自动保存状态快照
    # interrupt_before: 在进入 human_approval_node 前强制冻结图，
    #   等待外部调用 app.invoke(None, config) 恢复执行
    memory = MemorySaver()
    compiled = graph.compile(
        checkpointer=memory,
        interrupt_before=["human_approval_node"],
    )
    return compiled


# ══════════════════════════════════════════════════════════════
# 4. 测试入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  企业知识库客服 Agent - 沉浸式交互模式")
    print("  输入 'q' 或 'exit' 退出")
    print("=" * 60)

    # 构建图（内含 MemorySaver + interrupt_before）
    app = build_graph()

    # ── 会话配置 ──
    # 同一 thread_id 维持整轮对话的上下文记忆
    config = {"configurable": {"thread_id": "interactive_session_01"}}

    while True:
        # ── 获取用户输入 ──
        user_input = input("\n[用户]: ")
        if user_input.lower() in ("q", "exit", "quit"):
            print("再见！")
            break
        if not user_input.strip():
            continue

        # ── Stream 模式执行图，实时打印每个节点的执行 ──
        print("-" * 40)
        for event in app.stream(
            {"messages": [("user", user_input)]},
            config,
            stream_mode="updates",
        ):
            for node_name, node_state in event.items():
                print(f"  [流转] 节点 '{node_name}' 执行完毕")
        print("-" * 40)

        # ── 检查 HITL 挂起 ──
        state_snapshot = app.get_state(config)
        if state_snapshot.next and "human_approval_node" in state_snapshot.next:
            print("\n[系统告警] 发现高危操作，等待管理员授权...")
            auth = input("[?] 是否授权继续执行？(y/n): ")

            if auth.lower() == "y":
                print("-" * 40)
                for event in app.stream(None, config, stream_mode="updates"):
                    for node_name, node_state in event.items():
                        print(f"  [流转] 节点 '{node_name}' 执行完毕")
                print("-" * 40)
            else:
                print("[系统] 授权被拒绝，操作已取消")
                continue

        # ── 打字机效果打印 AI 最终回复 ──
        final_state = app.get_state(config)
        messages = final_state.values.get("messages", [])
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, "type") and last_msg.type == "ai":
                sys.stdout.write("\n[客服]: ")
                sys.stdout.flush()
                for ch in last_msg.content:
                    sys.stdout.write(ch)
                    sys.stdout.flush()
                    time.sleep(0.015)
                sys.stdout.write("\n")
                sys.stdout.flush()
