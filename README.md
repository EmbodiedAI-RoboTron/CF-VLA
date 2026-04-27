# CF_VLA

CF_VLA is a two-stage VLA (Vision-Language-Action) training and inference project.

## Core Information

- Core model implementation: `src/openpi/models_pytorch/cf_vla.py`
- Compatible configuration name: `pi0_2stg_pytorch`
- Training entry point: `scripts/train_pytorch.py`
- One-click launch script: `scripts/run_train_cf_vla.sh`

## Environment Setup

> Use the standalone environment file `environment.yml` to create the `cf_vla` conda environment for GPU reproduction.

```bash
# 1. Create the environment
conda env create -f environment.yml

# 2. Activate the environment
conda activate cf_vla

# 3. Install the project package
pip install -e .

# 4. Install the openpi client
pip install -e packages/openpi_client
```

## Optional Benchmark Dependencies

```bash
pip install -e ".[bench-libero]"
pip install -e ".[bench-calvin]"
```

## Training Example

```bash
bash scripts/run_train_cf_vla.sh \
  configs/pi05_2stg_pytorch_delta_actions/Cf_vla_libero.yaml
```

## Inference and Evaluation Example

```bash
# Server side
python scripts/serve_policy.py --env LIBERO

# Client side
python examples/libero/main.py
python examples/calvin/main.py
```

## License

- Code license: MIT. See `LICENSE`.
- Third-party dependency notes: `THIRD_PARTY_NOTICES.md`
- Model weights and datasets may have separate licenses. Please verify them separately before release.