# Overcomplete dictionary Phase Stitching
### For the paper: Sparse Reconstruction of Wavefronts using an Over-Complete Phase Dictionary

Here we fit an overcomplete dictionary of modes to a set of measured wavefront derivatives, whilst using an affine transform model and L1 regularisation to find the most efficient representation.

The main evaluation class is in, `utils/models/modalevaluator.py`, and its use is in `examples/Overcomplete.ipynb`

We initialize `num_inits` sets of trainable coefficients and affine parameter pairs. Then we choose the one that minimized loss.

To adapt for other types of special mode, add the mode into the `utils/modes/` folder, and adapt the `get_modes_and_derivs` function in `utils/functions.py`. The vortex is used for the example.