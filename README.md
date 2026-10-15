<h3 align="center">OpenSim</h3>

<p align="center"><strong>Rethinking Social Simulation in Open-Population Systems:<br>From Persistent Agents to Communication Processes</strong></p>

<p align="center">
  <a href="https://github.com/AI4SS/OpenSim"><img src="https://img.shields.io/badge/GitHub-OpenSim-3d7176?style=flat-square&logo=github" alt="GitHub"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-%3E%3D3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License: MIT"></a>
</p>


OpenSim shifts social simulation from persistent individual trajectories to
observable communication processes. It models participation, collective state
transitions, and communication generation in open populations, with an explicit
system–environment boundary and future-constrained learning.

![OpenSim framework](assets/opensim-overview.png)

## Setup

Python 3.11+, CUDA, and [Git LFS](https://git-lfs.com/) are required.

```bash
git lfs install
git clone https://github.com/AI4SS/OpenSim.git
cd OpenSim
git lfs pull
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Data: [data/](data/) · [Dataset disclosure](data/DATASET_DISCLOSURE.md)

Configs: [Train](configs/train.yaml) · [Simulate](configs/simulate.yaml) · [Evaluate](configs/evaluate.yaml)

## Usage

```bash
# Preview
python scripts/run.py --dry-run

# Train
python scripts/train.py --base-model Qwen/Qwen3-1.7B --output-dir outputs

# Simulate
python scripts/simulate.py --base-model Qwen/Qwen3-1.7B --output-dir outputs --resume

# Evaluate
python scripts/evaluate.py --base-model Qwen/Qwen3-1.7B --output-dir outputs --resume
```

Simulation requires training checkpoints and a passed validation gate.
Semantic evaluation requires `OPENAI_API_KEY` (optional `OPENAI_BASE_URL`) and may incur API charges.

## License

[MIT](LICENSE)
