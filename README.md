# Sign Translator

| Folder | What it holds |
|---|---|
| `ml/` | Data and training (Python) |
| `server/` | The API |
| `app/` | React Native app |

## ml/

```sh
cd ml
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # or requirements.lock.txt for exact versions
```

- `notebooks/` — Jupyter notebooks for exploring. Pick the **Python (sign-translator)** kernel.
- `scripts/` — plain `.py` files for anything you'll rerun (extraction, training, evaluation).
- `data/` — datasets and outputs; ignored by git.

Run a script with `.venv/bin/python scripts/<name>.py`. Start Jupyter with `.venv/bin/jupyter lab`.
