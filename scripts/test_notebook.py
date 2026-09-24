"""
Validates end-to-end execution of the Jupyter Notebook cells.
"""

import json
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

nb_path = Path(__file__).resolve().parent.parent / "notebooks" / "01_alpha_risk_econometrics.ipynb"
with open(nb_path, encoding="utf-8") as f:
    nb = json.load(f)

print(f"[*] Validating {nb_path.name} ({len(nb['cells'])} cells)...")
globals_dict = {"__file__": str(nb_path)}

code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
for i, cell in enumerate(code_cells, 1):
    code = "".join(cell["source"])
    print(f"  [+] Executing Code Cell {i}/{len(code_cells)}...")
    exec(code, globals_dict)

print("✅ All notebook code cells executed successfully without any errors!")
