"""
eval.py — Agent 自动化评测脚本

使用 LangSmith + LLM-as-a-Judge 模式，对 graph.py 中的企业知识库客服 Agent
进行批量评测。评测数据集聚焦 RAG 分支，避开需要人工审批 (HITL) 的 MCP 路径。
"""

import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".venv" / ".env")

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langsmith import Client, evaluate

from graph import build_graph

# ══════════════════════════════════════════════════════════════
# 1. 裁判契约 (SDD) —— 必须定义在 judge_llm 之前
# ══════════════════════════════════════════════════════════════

class JudgeGrade(BaseModel):
    """裁判模型输出的结构化评分。

    score=5 表示完美回答、完全匹配标准答案的核心信息。
    score=1 表示答非所问或存在严重事实错误。
    """
    score: int = Field(ge=1, le=5, description="评分，1到5分")
    reasoning: str = Field(description="给出评分的具体理由")


# ══════════════════════════════════════════════════════════════
# 2. 前置：编译图 + 裁判 LLM
# ══════════════════════════════════════════════════════════════

# 编译 Agent 图（包含 MemorySaver + interrupt_before）
app = build_graph()

# 裁判 LLM —— 与主 Agent 共用同一模型，打开结构化输出
# 注意：DeepSeek 不支持 json_schema 模式，必须显式指定 method="function_calling"
judge_llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "deepseek-chat"),
    temperature=0,
).with_structured_output(JudgeGrade, method="function_calling")


# ══════════════════════════════════════════════════════════════
# 3. 裁判函数 (LLM-as-a-Judge)
# ══════════════════════════════════════════════════════════════

def correctness_evaluator(run, example) -> dict:
    """LangSmith evaluator —— 对比标准答案与实际输出，用 LLM 打分。

    Args:
        run: LangSmith Run 对象，run.outputs["output"] 是 Agent 实际回答
        example: LangSmith Example 对象，含有 inputs/outputs（标准答案）

    Returns:
        dict: {"key": "correctness", "score": 0.0~1.0, "comment": str}
    """
    question = example.inputs["question"]
    expected = example.outputs["expected"]
    actual = run.outputs.get("output", "")

    prompt = (
        f"【用户问题】\n{question}\n\n"
        f"【标准答案】\n{expected}\n\n"
        f"【AI 实际回答】\n{actual}\n\n"
        "你是严苛的评测裁判。请对比「标准答案」和「AI 实际回答」，"
        "重点考察以下几点：\n"
        "1. 是否准确回答了用户的问题？（不答非所问）\n"
        "2. 关键事实是否与标准答案一致？（制度条款、数字、日期等）\n"
        "3. 对于标准答案为「不知道」的题目，AI 是否诚实表示不知道或未查到——"
        "如果 AI 编造了不存在的信息，请给 1 分。\n\n"
        "请给出 1-5 的评分和具体理由。"
    )

    grade: JudgeGrade = judge_llm.invoke(prompt)
    print(f"  [Judge] score={grade.score}/5, reason={grade.reasoning[:100]}...")

    # LangSmith evaluate 要求 score 归一化到 0.0~1.0
    return {
        "key": "correctness",
        "score": grade.score / 5.0,
        "comment": grade.reasoning,
    }


# ══════════════════════════════════════════════════════════════
# 4. Agent 包装器 (Target Function)
# ══════════════════════════════════════════════════════════════

def agent_target(inputs: dict) -> dict:
    """LangSmith 评测的目标函数 —— 接收 inputs，返回 Agent 最终回答。

    每次调用使用独立的 thread_id，避免 HITL 记忆跨评测用例污染。
    如果图在 human_approval_node 前挂起，自动恢复继续执行。
    """
    thread_id = uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}

    # 第一次 invoke：运行到结束或 HITL 挂起点
    res = app.invoke(
        {"messages": [("user", inputs["question"])]},
        config=config,
    )

    # 检查是否被 HITL 挂起
    state = app.get_state(config)
    if state.next and "human_approval_node" in state.next:
        print(f"  [agent_target] 触发 HITL 挂起，自动恢复执行 (thread={thread_id[:8]})")
        # 模拟人类审批通过
        res = app.invoke(None, config)

    # 提取最后一条 AI 消息
    all_messages = res.get("messages", [])
    last_ai = ""
    for m in reversed(all_messages):
        if hasattr(m, "type") and m.type == "ai":
            last_ai = m.content
            break

    return {"output": last_ai}


# ══════════════════════════════════════════════════════════════
# 5. 主执行脚本
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    client = Client()
    dataset_name = "客服知识库_RAG基础评测"

    print("=" * 60)
    print(f"  Agent 自动化评测 — {dataset_name}")
    print("=" * 60)

    # ── 创建数据集（不存在时） ──
    existing = list(client.list_datasets(dataset_name=dataset_name))
    if existing:
        ds = existing[0]
        print(f"\n[数据集已存在] {ds.name} (id={ds.id})")
    else:
        ds = client.create_dataset(dataset_name=dataset_name)
        print(f"\n[数据集已创建] {ds.name} (id={ds.id})")

        # 写入两条测试用例 —— 均为 RAG 知识库类问题，不触发 MCP/HITL
        examples_data = [
            {
                "inputs": {"question": "公司报销单多久之内提交有效？"},
                "outputs": {"expected": "在费用发生后 30 天内提交有效。"},
            },
            {
                "inputs": {"question": "年假可以休多少天？"},
                "outputs": {
                    "expected": "知识库中没有相关信息，应明确表示不知道或暂未查到。"
                },
            },
        ]

        for ex in examples_data:
            client.create_example(
                inputs=ex["inputs"],
                outputs=ex["outputs"],
                dataset_id=ds.id,
            )
        print(f"  已写入 {len(examples_data)} 条测试用例")

    # ── 执行评测 ──
    print("\n[开始评测]")
    print("-" * 60)

    results = evaluate(
        agent_target,
        data=dataset_name,
        evaluators=[correctness_evaluator],
        experiment_prefix="GPT4o_Eval",
        max_concurrency=2,
    )

    print("-" * 60)
    print("\n评测完成。可在 LangSmith 控制台查看详细报告。")
