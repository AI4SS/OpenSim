<p align="center">
  <strong>OpenSim</strong>
</p>

<p align="center">
  <strong>
    Rethinking Social Simulation in Open-Population Systems:<br>
    From Persistent Agents to Communication Processes
  </strong>
</p>

<p align="center">
  <a href="https://github.com/AI4SS/OpenSim"><img src="https://img.shields.io/badge/GitHub-OpenSim-3d7176?style=flat-square&logo=github" alt="GitHub"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-%3E%3D3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License: MIT"></a>
</p>

OpenSim is a communication-centered framework for social simulation in
open-population systems, where participation is sparse and changes over time.
Persistent account identities do not provide continuously observable individual
states. OpenSim therefore shifts the simulation target from presumed psychological
trajectories to the observable evolution of social communication: who participates,
what they communicate, and how these communications shape subsequent dynamics.
Given a system-visible history and declared environmental inputs, it recursively
generates and updates the platform process, without requiring an LLM to stand in
for a human mind.

![OpenSim framework: communication volume, state transition, future-compatible generation, and history feedback](assets/opensim-overview.png)

## Highlights

- **From persistent agents to communication processes.** Make observable
  communication history—not unobserved individual psychological trajectories—the
  recursive object of simulation, in conceptual alignment with social systems theory.
- **Endogenous participation in an open population.** Realize participants from a
  global user pool and generate their actions and content without access to the
  realized future participant sequence.
- **An explicit system–environment boundary.** Simulate endogenous communication
  dynamics conditional on declared environmental inputs, rather than asking the
  model to invent external events or reconstruct the single future that occurred.
- **Future-constrained communication learning.** Connect locally plausible
  communicative acts with subsequent collective dynamics. Future trajectories
  provide training supervision only; inference generates and selects communications
  using information available at simulation time.

## Project Structure

```text
OpenSim-Release/
├── assets/                     # Paper overview image
├── configs/
│   ├── train.yaml              # Training and policy-validation settings
│   ├── simulate.yaml           # Simulation settings
│   └── evaluate.yaml           # Annotation and metric settings
├── data/
│   ├── DATASET_DISCLOSURE.md    # Dataset and evaluation disclosure
│   ├── index.csv               # Event metadata and relative data paths
│   ├── train/{macro,micro}/     # Training event files
│   ├── test/{macro,micro}/      # Test event files
│   └── figures/{train,test}/   # Event-level dataset visualizations
├── opensim/
│   ├── state_transition/       # Latent-state model and event datasets
│   ├── policy/                 # Policy, trajectory critic, and generation
│   ├── training/               # Data preparation, training, and validation
│   ├── simulation/             # Population sampling and closed-loop rollout
│   ├── evaluation/             # Deterministic, NLL, semantic, and metric stages
│   ├── volume.py               # Causal interaction-volume forecaster
│   ├── summarizer.py           # Communication-history summarization
│   └── pipeline.py             # Workflow orchestration
├── scripts/                    # Workflow entry points
└── tests/                      # Focused regression tests
```

## Usage

### 1. Environment and Data

Requires Python 3.11+, CUDA, and [Git LFS](https://git-lfs.com/). Default model: **Qwen/Qwen3-1.7B**.

```bash
git lfs install
git clone https://github.com/AI4SS/OpenSim.git
cd OpenSim
git lfs pull
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

See [dataset disclosure](data/DATASET_DISCLOSURE.md). Model weights are not included.

### 2. Inspect the Workflow

```bash
python scripts/run.py --dry-run
```

Configs: [Train](configs/train.yaml) · [Simulate](configs/simulate.yaml) · [Evaluate](configs/evaluate.yaml).

### 3. Train

```bash
python scripts/train.py --base-model Qwen/Qwen3-1.7B --output-dir outputs
```

### 4. Simulate

Requires trained checkpoints and a passed validation gate.

```bash
python scripts/simulate.py --base-model Qwen/Qwen3-1.7B --output-dir outputs --resume
```

### 5. Evaluate

Set `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL`. Semantic evaluation may incur API charges.

```bash
python scripts/evaluate.py --base-model Qwen/Qwen3-1.7B --output-dir outputs --resume
```

### 6. Full Runs, Individual Stages, and Resume

```bash
# Full workflow
python scripts/run.py --base-model Qwen/Qwen3-1.7B --output-dir outputs

# Preview one stage
python scripts/evaluate.py --stage metrics --output-dir outputs --resume --dry-run
```

Keep model, configuration, and data settings unchanged when using `--resume`.
Interrupted training requires a new output directory.

### Using Existing Checkpoints

Use `--checkpoint-root` with the following layout:

```text
checkpoints/
├── state_transition/
├── volume/
├── policy_future/best_checkpoint/
└── summarizer/best_checkpoint/
```

For simulation, also supply `--validation-gate` and the matching `--base-model`.

## License

[MIT](LICENSE)
