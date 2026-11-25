# Shallow water model

The NCAR [SWM](https://github.com/NCAR/SWM) shallow water benchmark written with `gt4py.next`.
Start with the notebook; the other two files only support it.

- `swm.ipynb`: the model, a choice of backends, its validation, and an opt-in performance
  comparison across all backends.
- `swm_tpu.ipynb`: the same model restructured for TPUs, with the stencils of the linear
  stages as convolutions that lower to the MXU as implicit-im2col GEMMs, in float32
  and without a halo. Self-contained: it inlines the NumPy reference of
  `swm_numpy.py` and the NCAR reference data, so it also runs outside the repository.
  Read `swm.ipynb` first.
- `swm_numpy.py`: a NumPy implementation of the same scheme, which the notebook validates
  against.
- `swm_reference.npz`: `u`, `v` and `p` after the full 4000-step benchmark on the 16x16 grid,
  from NCAR/SWM `ref/16x16`.
