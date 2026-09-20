"""Write JSON Schema files from the pydantic models.

Run as ``python -m cmtool.schema.export --out src/cmtool/schema/json``. CI runs
it and fails if the committed files differ, so the published schema can never
drift from the models that write the data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cmtool.schema.sample import EXPORTED_MODELS


def export(out_dir: str | Path) -> list[Path]:
    """Write one ``<name>.schema.json`` per exported model; return the paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, model in EXPORTED_MODELS.items():
        schema = model.model_json_schema(mode="serialization")
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["title"] = schema.get("title", name)
        path = out / f"{name}.schema.json"
        path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(path)
    return written


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="src/cmtool/schema/json", help="output directory")
    args = parser.parse_args()
    for path in export(args.out):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
