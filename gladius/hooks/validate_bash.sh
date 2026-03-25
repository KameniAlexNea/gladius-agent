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

# Block any attempt to modify this hook file via Bash.
if echo "$COMMAND" | grep -qE 'validate_bash\.sh'; then
    if echo "$COMMAND" | grep -qE '(>>|>|tee|sed -i|awk.*>|perl.*-i|patch|truncate|rm|mv|cp[[:space:]])'; then
        echo "Blocked: modifying hook infrastructure files is not allowed." >&2
        exit 2
    fi
fi

# Block recursive delete of absolute paths (outside project)
if echo "$COMMAND" | grep -qE 'rm[[:space:]]+-[a-zA-Z]*r[a-zA-Z]*f[[:space:]]+/'; then
    echo "Blocked: 'rm -rf /' style command is not allowed. Use relative paths only." >&2
    exit 2
fi

# Block pkill -f / killall — pattern-based kills can accidentally SIGTERM the agent itself.
# Use 'kill PID' with an explicit PID instead.
if echo "$COMMAND" | grep -qE '\bpkill[[:space:]]+(-[a-zA-Z]*f[a-zA-Z]*|.*-f)\b|\bkillall\b'; then
    echo "Blocked: 'pkill -f' and 'killall' are not allowed — they can match and kill the agent process itself. Use 'kill PID' with an explicit PID captured from nohup/launch output." >&2
    exit 2
fi

# Block recursive delete of home directory
if echo "$COMMAND" | grep -qE 'rm[[:space:]]+-[a-zA-Z]*r[a-zA-Z]*f[[:space:]]+~'; then
    echo "Blocked: 'rm -rf ~' is not allowed." >&2
    exit 2
fi

# Block deletion of log files — logs are permanent audit trails, not disposable temp files.
# Uses a single combined regex so that `rm artifacts/*.bin && > logs/train.log` does NOT
# false-positive: [^;&|>]* stops at shell separators, so rm and the log path must be in
# the same shell segment (i.e. the log path is an argument to rm, not a later redirect).
if echo "$COMMAND" | grep -qE 'rm[[:space:]][^;&|>]*logs/[a-zA-Z_.-]*\.log'; then
    echo "Blocked: deleting log files is not allowed. logs/*.log are permanent audit trails. Overwrite via redirect (> logs/train.log) is fine; rm is not." >&2
    exit 2
fi

# Block any bash modification of CLAUDE.md (e.g. cat >> CLAUDE.md, tee CLAUDE.md)
if echo "$COMMAND" | grep -qE 'CLAUDE\.md'; then
    if echo "$COMMAND" | grep -qE '(>>|>|tee|sed -i|awk.*>|perl.*-i|patch|truncate)'; then
        echo "Blocked: modifying CLAUDE.md via Bash is not allowed. CLAUDE.md is managed by the orchestrator." >&2
        exit 2
    fi
fi

# Require any script with "train" in its name to redirect output to logs/train.log.
# Matches execution via: python[3], bash, uv run, or direct ./
# Does NOT match read-only uses like: cat, tail, grep, head, py_compile, or --help invocations.
if echo "$COMMAND" | grep -qE '(python3?|bash)[[:space:]]+[^[:space:]]*train[^[:space:]]*(\.(py|sh))|uv[[:space:]]+run[[:space:]].*\btrain[^[:space:]]*(\.(py|sh))|\./[^[:space:]]*train[^[:space:]]*(\.(py|sh))'; then
    # Exclude syntax checks and help requests — these are not training runs.
    if ! echo "$COMMAND" | grep -qE '(-m[[:space:]]+py_compile|py_compile[[:space:]]|--help|--version)'; then
        if ! echo "$COMMAND" | grep -qE 'logs/train\.log'; then
            echo "Blocked: any training script must redirect output to logs/train.log. Required format: nohup uv run python train.py > logs/train.log 2>&1 &" >&2
            exit 2
        fi
    fi
fi

# Require any script with "tune" in its name to redirect output to logs/tune.log.
if echo "$COMMAND" | grep -qE '(python3?|bash)[[:space:]]+[^[:space:]]*tune[^[:space:]]*(\.(py|sh))|uv[[:space:]]+run[[:space:]].*\btune[^[:space:]]*(\.(py|sh))|\./[^[:space:]]*tune[^[:space:]]*(\.(py|sh))'; then
    if ! echo "$COMMAND" | grep -qE 'logs/tune\.log'; then
        echo "Blocked: any tune script must redirect output to logs/tune.log. Required format: nohup uv run python tune.py > logs/tune.log 2>&1 &" >&2
        exit 2
    fi
fi

exit 0
