# Skill: Task Review & Quality Self-Assessment

## When to use this skill

Use this skill when you have completed a deliverable and need to:
1. Verify it meets the requirements in `README.md`
2. Self-assign a quality score (0–100) for reporting
3. Decide whether the deliverable is ready to package and submit

---

## Quality Score Guide (0–100)

| Score | Meaning |
| --- | --- |
| 90–100 | Exceeds requirements; polished, documented, tested |
| 70–89 | Meets all stated requirements; no major gaps |
| 50–69 | Meets most requirements; some gaps or rough edges |
| 30–49 | Partial implementation; core functionality works |
| 0–29 | Incomplete; significant requirements unmet |

---

## Review Checklist

Before reporting your quality score, verify each item:

### Functional Correctness
- [ ] Run the deliverable end-to-end (e.g., `uv run python app.py` or `./run.sh`)
- [ ] Outputs match what `README.md` asks for (format, content, location)
- [ ] No unhandled errors or crashes on the happy path

### Completeness
- [ ] All required features listed in `README.md` are implemented
- [ ] Any configuration/secrets described in README are handled
- [ ] Dependencies are declared (e.g., in `pyproject.toml`)

### Packaging
- [ ] All required files are present
- [ ] Deliverable can be reproduced from scratch with documented steps
- [ ] Submission artifact is saved with the path reported in `submission_file`

---

## Scoring Process

1. Read `README.md` and extract the explicit success criteria.
2. Run through the checklist above.
3. Assign a score 0–100 based on the guide.
4. Write 1–2 sentences justifying the score in your output `notes` field.

---

## Packaging the Deliverable

```bash
# Zip the output for submission
zip -r deliverable.zip output/ app.py requirements.txt README_submission.md

# Or record a URL / binary path in a text file
echo "https://..." > submission_url.txt
```

Report `submission_file` as the path to the zip / binary / URL file.
