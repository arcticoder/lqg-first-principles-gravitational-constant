# LQG First-Principles Gravitational Constant (Research-stage)

[![UQ Status](https://img.shields.io/badge/UQ_Status-REPORTED-orange.svg)](docs/technical-documentation.md)

## Overview

This repository documents a research-stage exploration of deriving Newton's gravitational constant G from Loop Quantum Gravity (LQG) inspired constructions. The materials include derivation notes, example computations, and an uncertainty-quantification (UQ) workflow. Numerical outputs and claims in this README reflect example runs and reported artifacts; they should be interpreted in the context of the full technical documentation and reproducibility artifacts.

### Reported Example Outcomes

- **Reported example prediction (example-run):** G ≈ 6.6743×10⁻¹¹ m³⋅kg⁻¹⋅s⁻² (see `FINAL_RESULTS.md` and `FINAL_REPORT.md` for the corresponding artifact produced by a specific run).
- **Reported agreement with experimental values (example-run):** reported percent agreement is derived from the example-run artifacts and depends on model choices and numeric tolerances.

Note: these numerical statements summarize outputs from specific computational runs included in this repository. They are not claimed here as definitive or production-grade results. Independent reproduction, sensitivity analysis, and domain review are recommended before treating any numeric output as robust.

## Key Notes on Scope and Limitations

- **Scope:** The code and documents here aim to share an approach for exploring first-principles models that relate LQG structures to an effective gravitational coupling. The repository is intended for research, reproducibility, and peer review.
- **Validation:** Example validation scripts and a UQ harness are available under `tests/` and `src/`. To reproduce reported example-run outputs, follow the steps in `docs/technical-documentation.md` and run the example scripts in `examples/` with documented seeds and environment settings.
- **Limitations:** Reported numbers are sensitive to modeling choices (e.g., vacuum selection criteria, discretization), numerical tolerances, and parameter settings. The repository's artifacts are a starting point for independent verification; they do not constitute a definitive theoretical proof or an engineering-grade measurement.

## Reproducibility & UQ Pointers

1. See `docs/technical-documentation.md` for methodological assumptions and the UQ workflow.
2. Reproduce the example run by executing `python src/vacuum_selection_uq_resolution.py` from a pinned environment; compare outputs to the included report artifacts.
3. Run parameter sweeps and multiple seeds to estimate sensitivity to vacuum selection and discretization choices.

## Repository Structure (abridged)

```
lqg-first-principles-gravitational-constant/
├── README.md
├── docs/
│   └── technical-documentation.md
├── src/
│   ├── vacuum_selection_uq_resolution.py
│   ├── scalar_tensor_extension.py
│   └── gravitational_constant.py
├── examples/
│   └── example_reduced_variables.json
└── tests/
    ├── test_enhanced.py
    └── test_uq_demo.py
```

## Recommended Next Steps for Reviewers

- Re-run the example analyses in an isolated environment and record full artifacts (raw outputs, seeds, environment, and command lines).
- Perform sensitivity sweeps for critical modeling choices and provide a short reproducibility report highlighting any unstable dependencies.
- If claiming theoretical completeness or closed-form derivations, include explicit proofs and peer-reviewed references; otherwise frame claims as research-stage, model-dependent findings.

## License

The project is released under the Unlicense (public domain dedication). The README emphasizes that the repository is research-stage; artifacts should be independently validated and peer-reviewed prior to strong claims or downstream engineering use.
