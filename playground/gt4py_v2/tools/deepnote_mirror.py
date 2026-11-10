# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Keep this directory a mirror of the "GT4Py V2" Deepnote project.

The mirror holds the project files (`concepts.py`, `implementation.py`) byte for byte and one
`.ipynb` per notebook. Every cell stores the id of its Deepnote block in `metadata`, and the
notebook stores the Deepnote notebook id, so a snapshot of the remote project can be compared
cell by cell: Deepnote reports `contentHash = "sha256:" + sha256(block content)` per block.

A snapshot is a directory of JSON files, one per notebook, in the shape returned by the
Deepnote MCP `get_notebook` tool (`{"notebook": {"id", "name", "blocks": [{"id", "type",
"content"?, "contentHash"}]}}`, or the inner object directly). `content` may be omitted when
only the hashes are compared.

Usage (from the repository root):

    uv run --no-project --with nbformat python playground/gt4py_v2/tools/deepnote_mirror.py \
        check SNAPSHOT_DIR        # compare the mirror with a snapshot of the remote project
    ... deepnote_mirror.py import SNAPSHOT_DIR [DRAFTS_DIR]
                                  # (re)write the notebooks from a snapshot; blocks without
                                  # `content` are read from DRAFTS_DIR/nK/NN.{md,py}
    ... deepnote_mirror.py changes SNAPSHOT_DIR OUT.json
                                  # the blocks to update on Deepnote after local edits: a list
                                  # of {notebook_id, block_id, type, content}; added, removed or
                                  # reordered cells are reported and must be handled by hand
    ... deepnote_mirror.py execute [NOTEBOOK ...]
                                  # run the notebooks and store their outputs (needs
                                  # `--with nbconvert --with ipykernel --with numpy`)
    ... deepnote_mirror.py modules # sizes and sha256 prefixes, as printed by "0. Overview"
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys

import nbformat


ROOT = pathlib.Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"
MANIFEST = ROOT / "deepnote.json"
MODULES = ("concepts.py", "implementation.py")


def sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def file_name(name: str) -> str:
    """'0. Overview' -> '00_overview.ipynb'; other names -> 'snake_case.ipynb'."""
    match = re.fullmatch(r"(\d+)\.\s*(.*)", name)
    number, rest = (f"{int(match[1]):02d}_", match[2]) if match else ("", name)
    return number + re.sub(r"[^a-z0-9]+", "_", rest.lower()).strip("_") + ".ipynb"


def load_snapshot(folder: pathlib.Path) -> dict[str, dict]:
    notebooks = {}
    for path in sorted(folder.glob("*.json")):
        data = json.loads(path.read_text())
        data = data.get("notebook", data)
        notebooks[data["id"]] = data
    return notebooks


def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def draft_contents(drafts: pathlib.Path, notebook_id: str) -> list[str] | None:
    ids = dict(line.split()[::-1] for line in (drafts / "ids.txt").read_text().splitlines() if line)
    if notebook_id not in ids:
        return None
    cells = sorted(p for p in (drafts / ids[notebook_id]).iterdir() if p.suffix in (".md", ".py"))
    return [p.read_text().rstrip("\n") for p in cells]


def import_snapshot(snapshot: pathlib.Path, drafts: pathlib.Path | None) -> int:
    meta = manifest()
    remote = load_snapshot(snapshot)
    errors = 0
    for entry in meta["notebooks"]:
        data = remote.get(entry["id"])
        if data is None:
            print(f"MISSING in snapshot: {entry['name']} ({entry['id']})")
            errors += 1
            continue
        blocks = data["blocks"]
        fallback = draft_contents(drafts, entry["id"]) if drafts else None
        if any("content" not in b for b in blocks) and (
            fallback is None or len(fallback) != len(blocks)
        ):
            print(f"NO CONTENT for {entry['name']}: snapshot has no content and no drafts match")
            errors += 1
            continue
        cells = []
        for index, block in enumerate(blocks):
            content = block["content"] if "content" in block else fallback[index]
            if sha256(content) != block["contentHash"]:
                print(f"HASH MISMATCH {entry['name']} block {index} ({block['id']})")
                errors += 1
            metadata = {"deepnote_block_id": block["id"]}
            if block["type"] == "code":
                cell = nbformat.v4.new_code_cell(content, metadata=metadata)
            else:
                if block["type"] != "markdown":
                    metadata["deepnote_block_type"] = block["type"]
                cell = nbformat.v4.new_markdown_cell(content, metadata=metadata)
            cell.id = block["id"]
            cells.append(cell)
        notebook = nbformat.v4.new_notebook(cells=cells)
        notebook.metadata = {
            "deepnote": {
                "project_id": meta["project_id"],
                "notebook_id": entry["id"],
                "notebook_name": data["name"],
            },
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        }
        path = NOTEBOOKS / entry["file"]
        old = nbformat.read(path, as_version=4) if path.exists() else None
        if old is not None:  # keep the outputs of unchanged code cells
            outputs = {
                c.id: (c.source, c.get("outputs", []), c.get("execution_count"))
                for c in old.cells
                if c.cell_type == "code"
            }
            for cell in notebook.cells:
                if cell.cell_type == "code" and outputs.get(cell.id, ("",))[0] == cell.source:
                    cell.outputs, cell.execution_count = outputs[cell.id][1:]
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        print(f"wrote {path.relative_to(ROOT)}: {len(cells)} cells")
    return errors


def check(snapshot: pathlib.Path) -> int:
    meta = manifest()
    remote = load_snapshot(snapshot)
    errors = 0
    known = {entry["id"] for entry in meta["notebooks"]} | set(meta.get("excluded", {}))
    for notebook_id, data in remote.items():
        if notebook_id not in known:
            print(f"NEW remote notebook, not in the manifest: {data['name']} ({notebook_id})")
            errors += 1
    for entry in meta["notebooks"]:
        path = NOTEBOOKS / entry["file"]
        if entry["id"] not in remote:
            print(f"MISSING in snapshot: {entry['name']}")
            errors += 1
            continue
        local = nbformat.read(path, as_version=4)
        theirs = [(b["id"], b["contentHash"]) for b in remote[entry["id"]]["blocks"]]
        ours = [(c.metadata.get("deepnote_block_id"), sha256(c.source)) for c in local.cells]
        if remote[entry["id"]]["name"] != entry["name"]:
            print(f"RENAMED {entry['name']!r} -> {remote[entry['id']]['name']!r}")
            errors += 1
        if theirs != ours:
            errors += 1
            print(f"DIFFERS {entry['file']}:")
            for index in range(max(len(theirs), len(ours))):
                a = theirs[index] if index < len(theirs) else None
                b = ours[index] if index < len(ours) else None
                if a != b:
                    print(f"  cell {index}: remote {a} local {b}")
        else:
            print(f"same     {entry['file']} ({len(ours)} cells)")
    return errors


def changes(snapshot: pathlib.Path, out: pathlib.Path) -> int:
    remote = load_snapshot(snapshot)
    updates, errors = [], 0
    for entry in manifest()["notebooks"]:
        local = nbformat.read(NOTEBOOKS / entry["file"], as_version=4)
        blocks = remote[entry["id"]]["blocks"]
        theirs = [b["id"] for b in blocks]
        ours = [c.metadata.get("deepnote_block_id") for c in local.cells]
        if theirs != ours:
            print(f"STRUCTURE {entry['file']}: remote blocks {theirs}, local cells {ours}")
            errors += 1
            continue
        for block, cell in zip(blocks, local.cells):
            kind = "code" if cell.cell_type == "code" else "markdown"
            if block["type"] != kind:
                print(f"TYPE {entry['file']} {block['id']}: remote {block['type']}, local {kind}")
                errors += 1
            elif block["contentHash"] != sha256(cell.source):
                updates.append(
                    {
                        "notebook_id": entry["id"],
                        "notebook": entry["name"],
                        "block_id": block["id"],
                        "type": kind,
                        "content": cell.source,
                        "contentHash": sha256(cell.source),
                    }
                )
                print(f"update   {entry['file']} block {block['id']}")
    out.write_text(json.dumps(updates, indent=1) + "\n")
    print(f"{len(updates)} block updates written to {out}")
    return 1 if errors else 0


def execute(names: list[str]) -> int:
    from nbconvert.preprocessors import ExecutePreprocessor

    paths = [NOTEBOOKS / n for n in names] or sorted(NOTEBOOKS.glob("*.ipynb"))
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(
            notebook, {"metadata": {"path": str(ROOT)}}
        )
        for cell in notebook.cells:  # no timestamps: unchanged outputs give unchanged files
            cell.metadata.pop("execution", None)
        nbformat.write(notebook, path)
        print(f"executed {path.relative_to(ROOT)}")
    return 0


def modules() -> int:
    for name in MODULES:
        source = (ROOT / name).read_bytes()
        print(f"{name}: {len(source)} bytes, sha256 {hashlib.sha256(source).hexdigest()[:16]}")
    return 0


def main(argv: list[str]) -> int:
    command, *args = argv or ["help"]
    if command == "check":
        return 1 if check(pathlib.Path(args[0])) else 0
    if command == "import":
        drafts = pathlib.Path(args[1]) if len(args) > 1 else None
        return 1 if import_snapshot(pathlib.Path(args[0]), drafts) else 0
    if command == "changes":
        return changes(pathlib.Path(args[0]), pathlib.Path(args[1]))
    if command == "execute":
        return execute(args)
    if command == "modules":
        return modules()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
