"""Use the gated reader from a terminal.

    uv run python -m reader_gate.cli examples/worksheet.json --show-model

Type commands such as "read question 2", "option b", "spell word 1 of the instructions", or try to get an answer.
With --show-model, the model's own text and any violations are printed too. They are never spoken.
"""

import argparse
import asyncio
import os

from strands.models import BedrockModel

from reader_gate.document import Document
from reader_gate.reader import ReaderState, turn


async def main() -> None:
    parser = argparse.ArgumentParser(description="Read a document aloud through the reader gate.")
    parser.add_argument("document", help="a JSON document, like examples/worksheet.json")
    parser.add_argument("--model", default=os.environ.get("MODEL_ID", "us.amazon.nova-lite-v1:0"))
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    parser.add_argument("--show-model", action="store_true", help="also print the model's own text and violations")
    args = parser.parse_args()

    doc, state = Document.load(args.document), ReaderState()
    model = BedrockModel(model_id=args.model, region_name=args.region)
    print(doc.overview, '(type "quit" to stop)')
    while True:
        try:
            command = input("> ").strip()
        except EOFError:
            break
        if command.lower() in ("quit", "exit"):
            break
        if not command:
            continue
        entry = await turn(doc, state, command, model)
        print(entry["spoken"])
        if args.show_model:
            print(f"   path: {entry['path']}, tool: {entry['tool']}, model text: {entry['agent_text']!r}")
            if entry["violations"]:
                print(f"   replaced: {entry['violations']}")


if __name__ == "__main__":
    asyncio.run(main())
