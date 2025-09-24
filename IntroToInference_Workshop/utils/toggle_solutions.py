import json
from pathlib import Path
import sys

# Paths relative to project root
config_path = Path("config.json")
source_notebook = Path("IntroToDataStructures_Workshop.ipynb")

# Safety check: ensure source notebook exists
if not source_notebook.exists():
    print(f"❌ ERROR: Source notebook not found: {source_notebook}")
    print("Make sure you're running this script from the project root, and that the source notebook exists.")
    sys.exit(1)

# Load config
if not config_path.exists():
    print(f"❌ ERROR: Config file not found: {config_path}")
    sys.exit(1)

with open(config_path, "r") as f:
    config = json.load(f)

after_workshop_mode = config.get("after_workshop_mode", False)

# Load source notebook
with open(source_notebook, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Filter or keep cells based on tags
new_cells = []
for cell in nb["cells"]:
    tags = cell.get("metadata", {}).get("tags", [])
    if "solution" in tags:
        if after_workshop_mode:
            new_cells.append(cell)  # Keep solutions
        else:
            continue  # Hide solutions
    else:
        new_cells.append(cell)

nb["cells"] = new_cells

# Decide output file name
if after_workshop_mode:
    output_path = source_notebook.with_name("IntroToDataStructures_Workshop_solutions.ipynb")
else:
    output_path = source_notebook.with_name("IntroToDataStructures_Workshop_challenge.ipynb")

# Save result
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print(f"✅ Notebook saved to {output_path}")
