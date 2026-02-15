# SwinUNETR Self-Training Pipeline

**Python 3.10+** | **License: Apache 2.0** | **Code style: [ruff](https://github.com/astral-sh/ruff)**

Self-training with iterative pseudo-labeling for 3D brain tumor segmentation using [SwinUNETR](https://arxiv.org/abs/2201.01266) and [MONAI](https://monai.io/).

> Full README coming soon. See [docs/method.md](docs/method.md) for methodology.

## Quick Start

```bash
git clone https://github.com/rgbussell/swinunetr_self_training.git
cd swinunetr_self_training
pip install -e ".[dev]"
pytest tests/ -v
```

## License

[Apache 2.0](LICENSE)
