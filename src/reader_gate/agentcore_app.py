"""Amazon Bedrock AgentCore Runtime entrypoint for the gated reader.

The runtime keeps no state between calls. The app sends the document, the command and the reader's position, and
gets back what to speak and the new position:

    {"document": {"passages": [...]}, "command": "read question 3", "position": -1, "last": null}

Install with `uv sync --group agentcore`. The gate also checks a `last` value sent back by the app, so a tampered
payload can't make the reader repeat text that isn't on the page.
"""

import os

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands.models import BedrockModel, Model

from reader_gate.document import Document
from reader_gate.reader import ReaderState, turn

app = BedrockAgentCoreApp()


async def handle(payload: dict, model: Model) -> dict:
    doc = Document.from_dict(payload["document"])
    state = ReaderState(position=int(payload.get("position", -1)), last=payload.get("last"))
    entry = await turn(doc, state, str(payload["command"]), model)
    return {"spoken": entry["spoken"], "position": state.position, "last": state.last, "path": entry["path"],
            "tool": entry["tool"], "violations": entry["violations"]}


@app.entrypoint
async def invoke(payload, context):
    model = BedrockModel(model_id=os.environ.get("MODEL_ID", "us.amazon.nova-lite-v1:0"),
                         region_name=os.environ.get("AWS_REGION", "us-east-1"))
    return await handle(payload, model)


if __name__ == "__main__":
    app.run()
