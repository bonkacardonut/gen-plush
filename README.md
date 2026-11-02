# gen-plush
image -> generate a plush template

## Setup (one time)
Use Python 3.12. Python 3.15 is too new: opencv / rembg may fail to install on it.

```bash
cd ~/Documents/gen-plush
rm -rf .venv
uv venv --python 3.12        # uv downloads 3.12 for you if needed
source .venv/bin/activate
uv pip install -r requirements.txt
python -m ipykernel install --user --name gen-plush --display-name "gen-plush (3.12)"
```
Then in VS Code / Jupyter, pick the **gen-plush (3.12)** kernel.

## Layout
- `src/background_removal.py`: the real background removal (fast edge flood fill, optional AI fallback)
- `notebooks/preprocessing_first_draft.ipynb`: first draft using rembg, kept for reference
- `data/test_images/2d/`: test clipart (easy / medium / hard)
- `data/2d-cleaned/2d_nobg/`: rembg output (first draft)
- `data/2d-cleaned/2d_nobg_v2/`: new method output

Batch-remove backgrounds from the terminal:
```bash
python -m src.background_removal
```
