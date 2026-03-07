#!/usr/bin/env bash
# PreToolUse hook — runs before every Bash tool call.
# Blocks commands that could destroy data outside the project directory.
#
# Exit code 2 = block the command and send the error message to Claude.
# Exit code 0 = allow the command.

INPUT=$(cat)

COMMAND=$(python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('tool_input', {}).get('command', ''))
except Exception:
    print('')
" <<< "$INPUT" 2>/dev/null || echo "")

# Block recursive delete of absolute paths (outside project)
if echo "$COMMAND" | grep -qE 'rm[[:space:]]+-[a-zA-Z]*r[a-zA-Z]*f[[:space:]]+/'; then
    echo "Blocked: 'rm -rf /' style command is not allowed. Use relative paths only." >&2
    exit 2
fi

# Block recursive delete of home directory
if echo "$COMMAND" | grep -qE 'rm[[:space:]]+-[a-zA-Z]*r[a-zA-Z]*f[[:space:]]+~'; then
    echo "Blocked: 'rm -rf ~' is not allowed." >&2
    exit 2
fi

# Block any bash modification of CLAUDE.md (e.g. cat >> CLAUDE.md, tee CLAUDE.md)
if echo "$COMMAND" | grep -qE 'CLAUDE\.md'; then
    if echo "$COMMAND" | grep -qE '(>>|>|tee|sed -i|awk.*>|perl.*-i|patch|truncate)'; then
        echo "Blocked: modifying CLAUDE.md via Bash is not allowed. CLAUDE.md is managed by the orchestrator." >&2
        exit 2
    fi
fi

exit 0
