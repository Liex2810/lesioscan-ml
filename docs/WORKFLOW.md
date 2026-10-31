# Execution checklist

1. Install compatible GPU PyTorch and verify CUDA.
2. Download official PAD-UFES-20; arrange images and metadata.
3. Inspect class counts and patient ID coverage.
4. Run `python scripts/prepare.py`; audit class counts, patient overlaps, image duplicates.
5. Run `pytest -q`.
6. Run ShuffleNet training and inspect validation metrics.
7. Repeat DenseNet and ViT experiments. Use different `--run` names for variants.
8. Select model using validation macro-F1 and per-class recall; do not tune on test.
9. Run the holdout evaluation only after finalizing selection.
10. Export ONNX and test consistency on real images before handing off to Flutter.

## Important research extension

Add **patient-grouped 5-fold CV within the development partition** before final research claims. The supplied CLI is a runnable single-split baseline, intentionally not masquerading as full group-stratified CV. Maintain one untouched patient-level holdout for final evaluation. Consider checking duplicate lesions across patients and collecting external smartphone photos with ground truth diagnosis.
