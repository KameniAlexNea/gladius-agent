#!/usr/bin/env bash
# PostToolUse hook — runs after Edit or Write tool calls.
# Compiles any modified Python file immediately so Claude sees syntax errors
# in the same turn rather than discovering them at runtime.
#
# Exit code 2 = send error back to Claude and block (Claude will fix it).
# Exit code 0 = success, continue normally.

INPUT=$(cat)

# Extract file path from tool_input (Edit sets 'path', Write sets 'path')
FILE_PATH=$(python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    ti = d.get('tool_input', {})
    print(ti.get('file_path', ''))  # SDK Write and Edit both use 'file_path'
except Exception:
    print('')
" <<< "$INPUT" 2>/dev/null || echo "")

# Block any attempt to edit this hook file itself.
if [[ "$(basename "$FILE_PATH")" == "after_edit.sh" ]]; then
    echo "Editing hook infrastructure files is not allowed." >&2
    exit 2
fi

if [[ "$FILE_PATH" == *.py ]] && [[ -f "$FILE_PATH" ]]; then
    ERRORS=$(python3 -m py_compile "$FILE_PATH" 2>&1)
    if [[ -n "$ERRORS" ]]; then
        echo "Syntax error detected in $FILE_PATH — fix before continuing:" >&2
        echo "$ERRORS" >&2
        exit 2
    fi
fi

# Validate EXPERIMENT_STATE.json: must be valid JSON and every agent value must be a dict.
if [[ "$(basename "$FILE_PATH")" == "EXPERIMENT_STATE.json" ]] && [[ -f "$FILE_PATH" ]]; then
    ERRORS=$(python3 - "$FILE_PATH" 2>&1 << 'PYEOF'
import sys, json

path = sys.argv[1]
try:
    with open(path) as f:
        data = json.load(f)
except json.JSONDecodeError as e:
    print(f"Invalid JSON: {e}")
    sys.exit(1)

if not isinstance(data, dict):
    print("EXPERIMENT_STATE.json must be a JSON object at the top level.")
    sys.exit(1)

bad = []
for key, val in data.items():
    if isinstance(val, str):
        bad.append(f"  '{key}': expected dict, got str ({repr(val)[:80]})")

if bad:
    print("EXPERIMENT_STATE.json has non-dict agent values — fix before continuing:")
    for line in bad:
        print(line)
    sys.exit(1)
PYEOF
    )
    if [[ -n "$ERRORS" ]]; then
        echo "$ERRORS" >&2
        exit 2
    fi
fi

exit 0
