"""
api_service.py — 模拟外部工单系统 API

生产环境中会被替换为真实的 HTTP 调用（如 requests.get / MCP 协议）。
当前用于 graph.py 中 mcp_node 的真实网络调用模拟。
"""

import json
import time


def fetch_ticket_info_api(ticket_id: str) -> str:
    """模拟查询工单详情的 HTTP API。

    使用 time.sleep(1.5) 模拟网络延迟。
    返回 JSON 字符串，调用方自行解析。

    Args:
        ticket_id: 工单 ID，如 "TKT-001"、"TKT-999"

    Returns:
        JSON 字符串，包含 status / handler / 金额 / 拒绝理由等字段
    """
    time.sleep(1.5)  # 模拟真实网络延迟

    if not ticket_id:
        return json.dumps({"error": "缺少必要参数: ticket_id"}, ensure_ascii=False)

    # ── 模拟工单数据库 ──
    mock_tickets = {
        "TKT-001": {
            "ticket_id": "TKT-001",
            "title": "差旅费报销申请",
            "status": "审批中",
            "handler": "张经理",
            "amount": 500,
            "submit_time": "2026-05-20 14:30:00",
            "category": "报销",
        },
        "TKT-999": {
            "ticket_id": "TKT-999",
            "title": "设备采购申请",
            "status": "已拒绝",
            "handler": "李总监",
            "amount": 15000,
            "submit_time": "2026-05-18 09:15:00",
            "category": "采购",
            "reject_reason": "发票抬头错误，请重新开具",
        },
    }

    ticket = mock_tickets.get(ticket_id)
    if ticket is None:
        return json.dumps(
            {"error": "工单不存在", "ticket_id": ticket_id, "code": 404},
            ensure_ascii=False,
        )

    return json.dumps(ticket, ensure_ascii=False)
