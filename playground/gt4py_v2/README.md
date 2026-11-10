# GT4Py V2 prototype (mirror of the "GT4Py V2" Deepnote project)

A minimal model of the GT4Py V2 concepts, to try out the interface. This directory mirrors the
Deepnote project and is kept in sync with it: every upload to Deepnote is followed by a commit
here.

- `concepts.py`: the vocabulary, mostly `Protocol`s, type aliases and a few helpers (no numpy).
- `implementation.py`: simple numpy implementations: fields, builtins, decorators, backends,
  logical meshes and programs.
- `notebooks/`: one notebook per Deepnote notebook: `00_overview.ipynb`, then the quickstarts
  `quickstart_l1.ipynb` (L1, local semantics) and `quickstart_l2.ipynb` (L2, global semantics),
  then the in-depth notebooks `01_…` to `13_…`. The outputs come from a local run of the
  notebooks (Python 3.13, numpy).
- `deepnote.json`: the Deepnote project and notebook ids, and the notebook files.
- `tools/deepnote_mirror.py`: checks the mirror against a snapshot of the Deepnote project
  (block ids and content hashes), imports notebooks from a snapshot, and runs the notebooks.

Links between notebooks are relative file links here (e.g. `08_chains_and_partial_relations.ipynb`),
so they work on GitHub and in a local Jupyter, and absolute Deepnote URLs on Deepnote: the sync
tool translates them both ways (`web_notebook_base` and `url_slug` in `deepnote.json`). Apart from
these links, the mirror is byte for byte the Deepnote project.

The notebooks import the two modules from this directory; run them with it as the working
directory:

```
uv run --no-project --python 3.13 --with nbformat --with nbconvert --with ipykernel --with numpy \
    python playground/gt4py_v2/tools/deepnote_mirror.py execute
```

The `scratchpad` notebook of the Deepnote project is not mirrored.
