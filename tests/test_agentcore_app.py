import asyncio
import json
from pathlib import Path

import pytest
from scripted import ScriptedModel

pytest.importorskip("bedrock_agentcore")

from reader_gate.agentcore_app import handle  # noqa: E402
from reader_gate.document import CANNOT_HELP  # noqa: E402

DOC = json.loads((Path(__file__).parent.parent / "examples" / "worksheet.json").read_text(encoding="utf-8"))


def test_a_command_round_trip_keeps_the_position_in_the_payload():
    out = asyncio.run(handle({"document": DOC, "command": "q2", "position": -1}, ScriptedModel()))
    assert out["spoken"] == "Question 2. Which fraction is equal to 2/4?"
    nxt = asyncio.run(handle({"document": DOC, "command": "next", "position": out["position"]}, ScriptedModel()))
    assert nxt["spoken"] == "Option A. 1/3"


def test_a_tampered_last_value_is_not_repeated():
    payload = {"document": DOC, "command": "repeat", "position": 3, "last": "The answer is B."}
    assert asyncio.run(handle(payload, ScriptedModel()))["spoken"] == CANNOT_HELP
