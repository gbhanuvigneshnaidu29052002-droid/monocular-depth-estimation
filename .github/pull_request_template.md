## Description
A concise summary of the architectural changes, loss functions, or bug fixes introduced.

## Related Issue(s)
Fixes #(issue number) or Addresses #(issue number)

## Type of Change
- [ ] Bug fix (non-breaking fix in model forward pass, loss, or dataloader)
- [ ] New feature (adding new architecture variant, loss function, or dataset)
- [ ] Performance improvement (training speedup, memory reduction, inference optimization)
- [ ] Documentation update (improving guides, docstrings, or benchmarks)
- [ ] Automated testing (adding unit tests or coverage)

## Verification Checklist
- [ ] Dependencies install cleanly via `pip install -r requirements.txt`
- [ ] Automated unit test suite passes:
  ```bash
  python3 -m unittest discover -s tests -v
  ```
- [ ] Forward pass output tensor contracts preserved across all 4 spatial heads
- [ ] Device-agnostic execution verified (CPU and CUDA)
- [ ] Evaluator metrics (Abs Rel, RMSE, Quadrant Accuracy) preserved
