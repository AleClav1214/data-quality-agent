"""Test the agent loop without calling the real API.

FakeClient stands in for the Claude API. It follows a fixed plan of tool
calls, then reads the findings returned to it and decides which to log,
the same shape of decision a real model makes. This checks the plumbing
(tool calls, results, logging, stopping) end to end, for free.
"""

import json
from types import SimpleNamespace

from dq_agent.agent import run_agent
from dq_agent.dataset import REFERENCE_DATE, generate
from dq_agent.evaluate import score


def _tool(name, args, n):
    return SimpleNamespace(type="tool_use", id=f"call_{n}", name=name, input=args)


def _response(blocks, stop="tool_use"):
    return SimpleNamespace(content=blocks, stop_reason=stop)


class FakeClient:
    PLAN = [
        ("profile_dataset", {}),
        ("check_missing", {"group_by": "status"}),
        ("check_duplicates", {}),
        ("check_duplicate_keys", {"column": "package_id"}),
        ("check_numeric_range", {"column": "weight_lbs", "min_value": 0.1, "max_value": 150}),
        ("check_date_values", {"column": "arrival_date"}),
        ("check_date_values", {"column": "pickup_date"}),
        ("check_date_order", {"start_column": "arrival_date", "end_column": "pickup_date"}),
        ("check_category_variants", {"column": "carrier"}),
        ("check_pattern", {"column": "resident_email", "pattern": "email"}),
        ("check_missing", {"columns": ["nonexistent"]}),  # a mistake the loop must survive
    ]

    def __init__(self):
        self.messages = SimpleNamespace(create=self.create)
        self.calls = 0

    def create(self, *, messages, tools, **kwargs):
        assert tools and kwargs["system"]
        self.calls += 1
        n = self.calls
        if n <= len(self.PLAN):
            name, args = self.PLAN[n - 1]
            return _response([SimpleNamespace(type="text", text=f"Running {name}"), _tool(name, args, n)])

        if n == len(self.PLAN) + 1:  # judgment step: read every finding returned so far
            findings = []
            for m in messages:
                if m["role"] == "user" and isinstance(m["content"], list):
                    for block in m["content"]:
                        findings += json.loads(block["content"]).get("findings", [])
            blocks = []
            for f in findings:
                if f["category"] == "missing" and f["column"] == "pickup_date":
                    continue  # expected: packages still waiting have no pickup date
                blocks.append(_tool("log_issue", {"finding_id": f["finding_id"], "severity": "medium",
                                                  "why_it_matters": "x", "suggested_fix": "y"}, f"log{len(blocks)}"))
            return _response(blocks)

        return _response([_tool("finish", {"summary": "Done."}, "finish")])


def test_agent_loop_end_to_end():
    messy, truth, _ = generate(3000, seed=1)
    client = FakeClient()
    state = run_agent(messy, client=client, reference_date=REFERENCE_DATE, verbose=False)

    assert state.done and state.stop_reason == "agent finished"
    assert state.data_unchanged()
    assert any(t.get("error") for t in state.tool_log)  # bad column was reported, not fatal
    assert not any(i["column"] == "pickup_date" and i["category"] == "missing" for i in state.issues)

    total = score(state.issues, truth).set_index("category").loc["ALL"]
    assert total["recall"] == 1.0
    assert total["precision"] == 1.0


def test_step_limit_stops_runaway_loop():
    class Looper:
        messages = SimpleNamespace(create=lambda **k: _response([_tool("profile_dataset", {}, 1)]))

    messy, _, _ = generate(600, seed=2)
    state = run_agent(messy, client=Looper(), max_steps=3, verbose=False)
    assert not state.done and "step limit" in state.stop_reason
