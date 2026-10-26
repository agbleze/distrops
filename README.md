# distrops

[![Python Version](https://shields.io)](https://python.org)
[![Package Version](https://shields.io)](https://github.com)
[![License](https://shields.io)](LICENSE)

**distrops** (Distributed Training Operations) is a model-agnostic framework engineered to unify, scale, and automate distributed model training and asynchronous hyperparameter optimization sweeps. The framework provides a programmatic guardrail interface for machine learning workflow.

`distrops` provides a declarative, configuration-driven workflow spanning data access, model definition, hyperparameter configuration and the necessary orchestration logic for complete model training in a distributed manner. Whether you are fine-tuning a language transformer, training a computer vision backbone, or sweeping parameters across an ensemble of tabular gradient boosters, `distrops` assembles the building blocks, handles resource orchestration and trial lifecycle management uniformly as appropriate.


---

## The Unified Value Proposition

The distributed machine learning ecosystem often forces engineering teams to choose between two structural extremes:
*   **Raw Infrastructure Primitives:** Low-level clustering utilities provide infinite flexibility but burden developers with massive boilerplate, fragile orchestration logic, and intricate network serialization traps.
*   **Opaque No-Code Engines:** Monolithic configuration engines abstract away infrastructure but hide the underlying machine learning logic behind rigid schemas, making it difficult for research teams to inject custom modeling techniques, losses, or experimental architectures.

**`distrops` bridges this gap natively by acting as a programmatic, guardrailed platform layer.** 

`distrops` lets engineers write pure, uninhibited PyTorch, Hugging Face, or traditional statistical learning code in addition to providing direct access to those provided by existing frameworks. By simply attaching a `@REGISTRY` decorator or inheriting from a baseline blueprint, developers instantly gain access to robust distributed orchestration, hyperparameter tuning, and automated early stopping - leaving `distrops` to assemble them and orchestrate them with distributed computing engineering under the hood.

---

## Core Package Architecture & State Flow

`distrops` isolates the orchestration control layer from remote data streaming and execution worker threads. This guarantees that your main process remains lightweight and eliminates runtime function closure capture bottlenecks entirely. You declare the data and model to use (in case it is Open source) and they are access wi

```text
 1. DECLARATIVE ENTRYPOINT           2. UNIFIED ORCHESTRATION ENGINE (Driver State)
┌─────────────────────────┐         ┌────────────────────────────────────────────────┐
│  distrops CLI / YAML    │────────►│  • Parses Unified Context State Manifest       │
│  Context Manifest       │         │  • Computes Dynamic Multi-Tenant Resource Map  │
└─────────────────────────┘         │  • Drives Non-Blocking Optimization Strategies  │
                                    └───────────────────────┬────────────────────────┘
                                                            │
                                        ┌───────────────────┴───────────────────┐
                                        ▼ (Asynchronous Node Serialization)      ▼
                                ┌────────────────────────┐              ┌────────────────────────┐
                                │ WORKER NODE 00001      │              │ WORKER NODE 00002      │
                                ├────────────────────────┤              ├────────────────────────┤
                                │ • Lazy Module Imports  │              │ • Lazy Module Imports  │
                                │ • Stream Data Pipeline │              │ • Stream Data Pipeline │
                                │ • Gradient Update Loop │              │ • Gradient Update Loop │
                                │ • Direct I/O Telemetry │              │ • Direct I/O Telemetry │
                                └────────────────────────┘              └────────────────────────┘
```

*   **The Orchestration Engine (Driver):** Acts as the centralized cluster control tower. It ingests your declaration manifests, provisions multi-tenant system resource boundaries, schedules execution waves, and applies automated early-stopping heuristics based on historical trial performance.
*   **The Isolated Worker Engines:** Spin up dynamically within isolated hardware contexts. Workers execute lazy importing to initialize completely self-contained training graphs, native optimization routines, and direct, asynchronous telemetry event streaming.

---

## Key Capabilities & Framework Features

*   **Model-Agnostic Interface:** Standardized runtime execution loops capable of scaling deep NLP sequence networks, convolutional vision layers, or tabular tree models without modifying core orchestration source files.
*   **Multi-Tenant Fractional Resourcing:** Provisions explicit, fractional physical or virtual hardware slices (e.g., `0.50 GPU` blocks) to run parallel high-density optimization waves concurrently without triggering memory allocation collisions.
*   **Streaming Data Backpressure Mitigation:** Implements asynchronous, object-oriented collation data-loaders that stream raw data partitions lazily via zero-copy structures, enforcing hard system memory boundaries.
*   **Storage Boundary Hardening:** Enforces strict execution checkpoint limits to prune sub-optimal intermediate models in real-time, automatically tracking the absolute top-performing parameters while protecting drive space.
*   **Uniform Metric Telemetry Contracts:** Standardizes evaluation tracking flags across all distributed worker nodes, providing robust, non-blocking telemetry streaming that pipes logs straight to shared external visualization drives.

---

## Getting Started

### Installation
```bash
# Clone the core framework
git clone https://github.com.git
cd distrops

# Install standard base package
pip install .

# Package installation with developer dependency (Includes unit test, linting, and formatting tools)
pip install -e ".[dev]"

# Package installation with Tensorboard visualization
pip install -e ".[dashboard]"

# Complete Package with all optional dependency Installation
pip install -e ".[all]"
```
*Note: The outer quotation marks are required in advanced shells like Zsh to prevent character-globbing evaluation errors.*

---

# Usage

## Command-Line Interface (CLI) Specification

`distrops` exposes a unified, brand-encapsulated CLI tool namespace. Trailing subcommands route execution dynamically to decoupled backend engines via an abstract subparsers routing layer.

### 1. Execute an Asynchronous Hyperparameter Sweep
Orchestrates a comprehensive distributed hyperparameter search sweep, evaluating parameter bounds and applying early-stopping rules based on a declarative YAML config manifest:
```bash
distrops hpo --config_filepath config/config.yml
```

### 2. Run a Single Baseline Trial
Trains a single model checkpoint against fixed, explicit parameters to establish baseline performance metrics:
```bash
distrops train --config_filepath config/config.yml
```

### 3. Trigger Post-Training Performance Optimizations
Traces, quantizes, or packages trained model artifacts to produce highly optimized, production-ready inference outputs:
```bash
distrops optimize --config_filepath config/config.yml
```

---

## 📊 Observability & Performance Monitoring

All `distrops` workflows natively output non-blocking scalar event files to a decoupled storage mount, enabling comprehensive experiment tracing without adding overhead inside the active model training loops.

1. Launch your visual telemetry background server pointing to your shared results path:
```bash
tensorboard --logdir_spec=experiment:/mnt/d/distributed_work/cluster_storage/ray-results/ --host 0.0.0.0 --reload_interval 5
```
2. Open your web browser and navigate to `http://localhost:6006/` to monitor model training performance metrics 


---

## 🔬 Repository Structure

```text
distrops/
├── config/
│   └── config.yml              # Configuration for model training, resource and workflow configuration 
├── src/
│   └── distrops/               # package namespace
│       ├── __init__.py         # Package initialiazation & version flags
│       ├── runner.py           # CLI Subcommand traffic controller
│       ├── train_worker.py     # Self-contained worker execution engine
│       ├── data_loader.py      # Automated data streaming & collation layer
│       └── utils.py            # Package-wide YAML and system utilities
├── tests/
│   └── test_distrops.py        # Automated validation & unit testing pipelines
├── pyproject.toml              # Package build configuration & dependency matrix
├── monitor.sh                  # Bash script to launch tensorboard
├── run.sh                      # Bash script to run package
└── README.md
```

---

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`distrops` was created by Linus Agbleze. It is licensed under the terms of the MIT license.

## Credits

`distrops` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
