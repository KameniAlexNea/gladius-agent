"""
Patch claude_agent_sdk message_parser to tolerate missing 'signature' field.

Ollama and some other backends omit the 'signature' key in thinking blocks,
causing a MessageParseError. This script replaces the strict dict access with
a .get() fallback so those backends work.

Run after any SDK upgrade:
    uv run python scripts/patch_sdk.py
"""

import importlib.util
import sys
from pathlib import Path

TARGET = 'block["signature"]'
REPLACEMENT = 'block.get("signature", "")'

spec = importlib.util.find_spec("claude_agent_sdk")
if spec is None:
    print("ERROR: claude_agent_sdk not found in current environment")
    sys.exit(1)

parser = Path(spec.submodule_search_locations[0]) / "_internal" / "message_parser.py"
if not parser.exists():
    print(f"ERROR: message_parser.py not found at {parser}")
    sys.exit(1)

text = parser.read_text(encoding="utf-8")

if TARGET not in text:
    print(f"Nothing to patch — '{TARGET}' not found (already patched or changed).")
    sys.exit(0)

patched = text.replace(TARGET, REPLACEMENT)
parser.write_text(patched, encoding="utf-8")
count = text.count(TARGET)
print(f"Patched {parser}")
print(f"  {count} occurrence(s): '{TARGET}' → '{REPLACEMENT}'")