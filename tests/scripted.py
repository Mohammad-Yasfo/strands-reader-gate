"""A Strands model that replays a script, so the real Agent, tools and hooks run offline.

Each turn is a list of steps: ("text", "...") or ("tool", name, {input}). One turn is used per model call.
"""

import asyncio
import json
import uuid
from typing import Any

from strands.models import Model


class ScriptedModel(Model):
    def __init__(self, *turns: list[tuple], delay: float = 0.0):
        self.turns, self.delay, self.calls, self.config = list(turns), delay, 0, {}

    def update_config(self, **model_config: Any) -> None:
        self.config.update(model_config)

    def get_config(self) -> Any:
        return self.config

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        raise NotImplementedError
        yield  # pragma: no cover

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        steps = self.turns.pop(0) if self.turns else [("text", "")]
        stop = "end_turn"
        yield {"messageStart": {"role": "assistant"}}
        for kind, *rest in steps:
            if kind == "text":
                yield {"contentBlockStart": {"start": {}}}
                yield {"contentBlockDelta": {"delta": {"text": rest[0]}}}
            else:
                name, args = rest
                yield {"contentBlockStart": {"start": {"toolUse": {"name": name, "toolUseId": uuid.uuid4().hex}}}}
                yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(args)}}}}
                stop = "tool_use"
            yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": stop}}
        yield {"metadata": {"usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
                            "metrics": {"latencyMs": 0}}}
