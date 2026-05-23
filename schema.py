"""
企业知识库客服 Agent — 状态与数据契约 (Schema)

本文件使用 Pydantic v2 定义了 Agent 图的所有输入/输出契约和全局状态。
LangGraph 的状态是 TypedDict，reducer 控制字段的合并策略。
"""

import operator
from typing import Annotated, List, Literal, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# 1. 路由输出契约
# ──────────────────────────────────────────────

class RouterDecision(BaseModel):
    """LLM 完成意图分类后输出的结构化决策。

    路由节点根据此决策将请求分发到不同的处理分支：
    - rag_search: 走知识库检索
    - mcp_query:  走 MCP 工具系统查询（如工单系统）
    - chitchat:   闲聊，直接生成回答
    - escalate_to_human: 转人工
    """
    intent: Literal["rag_search", "mcp_query", "chitchat", "escalate_to_human"] = Field(
        description="意图分类"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="路由置信度, 0.0~1.0"
    )
    reasoning: str = Field(
        default="", description="决策思考过程"
    )


# ──────────────────────────────────────────────
# 2. 纠错反馈契约
# ──────────────────────────────────────────────

class ReflectionResult(BaseModel):
    """反思节点对生成回答的审查结果。

    用于 Self-Reflection 模式：
    - 如果 is_valid=False，feedback 中携带修改建议，
      图会回退到 generate_node 重写。
    - 如果 is_valid=True，回答通过审查，流程结束。
    """
    is_valid: bool = Field(description="是否符合标准无幻觉")
    feedback: str = Field(description="修改建议")


# ──────────────────────────────────────────────
# 3. 工具入参契约
# ──────────────────────────────────────────────

class RAGSearchInput(BaseModel):
    """知识库检索工具的入参 schema。

    LangGraph 的 ToolNode 或 tool-calling 机制会根据此 schema
    校验 LLM 传过来的参数，确保 query 和 category 字段齐全。
    """
    query: str = Field(description="优化后的检索词")
    category: Literal["hr", "it", "finance", "general"] = Field(
        default="general", description="知识库分类"
    )


class MCPQueryInput(BaseModel):
    """MCP 工单查询工具的入参 schema。

    ticket_id 为可选——如果没有提供则根据 user_email 列出所有工单。
    user_email 用于权限校验，确保只返回该用户的工单。
    """
    ticket_id: Optional[str] = Field(default=None, description="工单 ID, 可选")
    user_email: str = Field(description="用户邮箱, 用于鉴权")


# ──────────────────────────────────────────────
# 4. 图全局状态 (AgentState)
# ──────────────────────────────────────────────

class AgentState(BaseModel):
    """LangGraph 图的全局状态。

    整个图的每个节点读取同一个 state 对象，但只能通过返回 dict
    来更新字段——LangGraph 会在每个节点返回后自动 merge。

    Reducer 说明:
    - add_messages: 将新消息追加到 messages 列表末尾（而非覆盖）
    - operator.add: 将新 chunk 追加到 retrieved_context 列表末尾
    - 其他字段无 reducer → 新值直接覆盖旧值
    """
    messages: Annotated[List[BaseMessage], add_messages] = Field(
        default_factory=list,
        description="对话历史, 使用 add_messages reducer 追加而非覆盖"
    )
    user_email: str = Field(
        default="guest@company.com",
        description="当前用户邮箱, 节点据此做权限判断"
    )
    current_intent: Optional[str] = Field(
        default=None,
        description="router_node 分类后的意图, 条件边据此分发"
    )
    retrieved_context: Annotated[List[str], operator.add] = Field(
        default_factory=list,
        description="RAG 检索到的文档片段, 使用 operator.add 追加"
    )
    reflection_feedback: Optional[str] = Field(
        default=None,
        description="reflection_node 的审查意见, generate_node 据此修改"
    )
    retry_count: int = Field(
        default=0,
        description="反思-重写循环的当前次数, 条件边用此值判断是否超限"
    )
    requires_human_approval: bool = Field(
        default=False,
        description="是否需要人工介入, mcp_node 或 escalate 触发"
    )
