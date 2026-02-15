# SwinUNETR Self-Training Pipeline

## Architecture
Teacher-student self-training for brain tumor segmentation using SwinUNETR.
- **Teacher**: EMA-updated copy of student weights; generates pseudo-labels on unlabeled data
- **Student**: Trained on labeled data (supervised loss) + pseudo-labeled data + consistency regularization
- **Curriculum thresholding**: Confidence thresholds decrease across rounds (0.95 → 0.75)
- **Iterative rounds**: 4 cycles of pseudo-label generation → retraining

## Module Map
```
src/swinunetr_st/
├── data/           # Unlabeled dataset, pseudo-label dataset, strong augmentations
├── models/         # EMA teacher, combined loss function
├── training/       # Self-trainer loop, pseudo-labeler, curriculum scheduler
├── analysis/       # Comparison metrics, convergence analysis, visualization
├── utils/          # Config loading, path validation
└── cli.py          # Entry points: st-train, st-pseudo-label, st-compare
```

## Key Paths
| What | Path |
|------|------|
| Config | `configs/self_training_config.yaml` |
| Tests | `tests/` |
| Scripts | `scripts/` |
| Notebooks | `notebooks/` |
| Method docs | `docs/method.md` |
| Newsletter | `docs/newsletter/` |

## Commands
```bash
pytest tests/ -v                          # Run tests
ruff check src/ tests/                     # Lint
ruff format src/ tests/                    # Format
pre-commit run --all-files                 # Pre-commit hooks
python scripts/run_self_training.py --config configs/self_training_config.yaml  # Train
```

## Conventions
- `from __future__ import annotations` at top of every module
- Type hints: `X | Y` union syntax (not `Union[X, Y]`)
- Docstrings on all public functions/classes with Args/Returns
- Logging: `logging.getLogger(__name__)`
- Security: validate paths, `mode=0o700` on dirs, `torch.load(weights_only=True)`
- Line length: 120
- NEVER reference Claude, Copilot, or AI in commits or code
- GitHub username: rgbussell
