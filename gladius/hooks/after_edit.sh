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

if [[ "$FILE_PATH" == *.py ]] && [[ -f "$FILE_PATH" ]]; then
    ERRORS=$(python3 -m py_compile "$FILE_PATH" 2>&1)
    if [[ -n "$ERRORS" ]]; then
        echo "Syntax error detected in $FILE_PATH — fix before continuing:" >&2
        echo "$ERRORS" >&2
        exit 2
    fi
fi

exit 0
