---
name: jupyter-mcp
description: >
  Start a Jupyter Lab server and the jupyter-mcp-server process so you can
  read/write/execute notebook cells via mcp__jupyter__* tools.
  Invoke this before using any mcp__jupyter__* tool.
---

## Step 1 — install dependencies (once per project)

```bash
uv add jupyter-mcp-server jupyterlab
```

## Step 2 — start Jupyter Lab in the background

```bash
jupyter lab \
  --no-browser \
  --port 8888 \
  --IdentityProvider.token=gladius-mcp \
  --ServerApp.disable_check_xsrf=True \
  > /tmp/jupyter.log 2>&1 &
echo "Jupyter PID: $!"
```

Wait ~3 seconds for it to be ready:

```bash
sleep 3 && curl -s http://localhost:8888/api?token=gladius-mcp | grep -q version && echo "Jupyter is up" || echo "Jupyter not ready yet"
```

## Step 3 — the MCP server is already wired

`jupyter-mcp-server` is pre-registered in this session with:

| Variable      | Value                    |
| ------------- | ------------------------ |
| DOCUMENT_URL  | http://localhost:8888    |
| RUNTIME_URL   | http://localhost:8888    |
| TOKEN         | gladius-mcp              |
| DOCUMENT_ID   | solution.ipynb           |

The `mcp__jupyter__*` tools are now available — no extra config needed.

## Step 4 — create or open a notebook

If `solution.ipynb` does not exist yet, create it:

```bash
uv run python -c "
import nbformat
nb = nbformat.v4.new_notebook()
nbformat.write(nb, 'solution.ipynb')
print('solution.ipynb created')
"
```

Then use `mcp__jupyter__*` tools to add cells, execute them, and read outputs.

## Useful mcp__jupyter__* tools

| Tool | Purpose |
| ---- | ------- |
| `mcp__jupyter__notebook_read` | Read all cells and outputs |
| `mcp__jupyter__cell_append` | Append a new code or markdown cell |
| `mcp__jupyter__cell_execute` | Execute a cell by index |
| `mcp__jupyter__cell_replace` | Replace a cell's source |
| `mcp__jupyter__kernel_restart` | Restart the kernel (clears state) |

## Stopping Jupyter when done

```bash
kill $(lsof -ti:8888) 2>/dev/null && echo "Jupyter stopped"
```
