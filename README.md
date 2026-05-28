# SwarmGPT

![swarm_gpt_banner](/docs/img/swarm_gpt_banner.png)

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![ROS Noetic](https://img.shields.io/badge/ROS2-Kilted-blue.svg)](https://docs.ros.org/en/kilted/index.html)
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![Ruff](https://github.com/learnsyslab/swarmGPT/actions/workflows/ruff.yaml/badge.svg)](https://github.com/learnsyslab/swarmGPT/actions/workflows/ruff.yaml)
[![Tests](https://github.com/learnsyslab/swarmGPT/actions/workflows/tests.yaml/badge.svg)](https://github.com/learnsyslab/swarmGPT/actions/workflows/tests.yaml)
[![Docs](https://github.com/learnsyslab/swarmGPT/actions/workflows/docs.yaml/badge.svg)](https://github.com/learnsyslab/swarmGPT/actions/workflows/docs.yaml)

SwarmGPT integrates large language models (LLMs) with safe swarm motion planning, providing an automated and novel approach to deployable drone swarm choreography. Users can automatically generate synchronized drone performances through natural language instructions. Emphasizing safety and creativity, the system combines the creative power of generative models with the effectiveness and safety of model-based planning algorithms. For more information, visit the [project website](https://utiasdsl.github.io/swarmGPT/) or read our [paper](https://ieeexplore.ieee.org/document/11197931/).

- [SwarmGPT](#swarmgpt)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setting up SwarmGPT](#setting-up-swarmgpt)
  - [How to run SwarmGPT](#how-to-run-swarmgpt)
    - [Prerequisites](#prerequisites-1)
    - [Launching the Interface](#launching-the-interface)
    - [Using the Interface](#using-the-interface)
    - [Ready for Deployment](#ready-for-deployment)
  - [Deployment](#deployment)
  - [Citing](#citing)

## Installation

SwarmGPT uses [Pixi](https://pixi.sh) for dependency management and environment setup. Pixi provides a fast, reliable package manager that handles both conda and PyPI dependencies seamlessly.

### Prerequisites

- Linux x64 or macOS Apple Silicon (see `platforms` in `pyproject.toml`)
- [Pixi package manager](https://pixi.sh) - see [installation instructions](https://pixi.sh/latest/installation/)

### Setting up SwarmGPT

Clone the repository and activate the environment:
```bash
git clone git@github.com:utiasDSL/swarmGPT.git
cd swarmGPT
pixi install
```

Lastly, we rely on the VLC media player to play the music. In case you don't have it installed, run:
```bash
sudo apt install vlc
```
On macOS, install [VLC](https://www.videolan.org/vlc/) separately if needed.
Your setup is ready now. 

<!-- ### Documentation Environment

To work with documentation, use the docs environment:

```bash
# Serve documentation locally
pixi run -e docs docs-serve

# Build documentation
pixi run -e docs docs-build
``` -->
## How to run SwarmGPT

### Prerequisites

Before running SwarmGPT, ensure you have:

1. **LLM access** (pick one):
   - **OpenAI (default)** — set your API key:
     ```bash
     export OPENAI_API_KEY="your-api-key-here"
     ```
     For convenience, create `openai_api_key.sh` in the repo root with that line; Pixi sources it on `pixi shell` (the file is gitignored).
   - **Ollama (local)** — no OpenAI key required; install the CLI following the instructions [here](https://ollama.com/download) and run the server as in step 2 below.

2. **Configuration** — edit swarm and safety settings at:
   - `swarm_gpt/data/drones.toml` — drone URIs and home positions
   - `swarm_gpt/data/settings.yaml` — environment and safety filter settings

### Launching the Interface

Typical first-time setup:

1. **Pixi shell** (main terminal):
   ```bash
   pixi shell
   ```
   This activates the environment and installs or refreshes browser UI dependencies.

2. **Ollama server** (only if using Ollama; in a separate terminal, this command blocks):
   ```bash
   ollama serve
   ```
   Pull a model in another terminal if needed: `ollama pull <model_name>`. We have had the best results with `gemma4:latest` ([model library](https://ollama.com/library)).

3. **API**:
   ```bash
   pixi run api
   ```
   This builds the browser UI first, then serves the API and frontend.

4. **Open** `http://127.0.0.1:8000` and choose **ChatGPT / OpenAI** or **Ollama (local)** in the UI.

   Optional parameters:
   ```bash
   # Use different LLM model
   python swarm_gpt/launch.py --model_id="gpt-4o-mini"
   
   # Disable motion primitives (use raw waypoints)
   python swarm_gpt/launch.py --use_motion_primitives=False
   ```

For frontend development, run the API and Vite dev server in separate terminals (keep `ollama-serve` running if you use Ollama):
```bash
pixi run api
pixi run web-dev
```
Then open `http://127.0.0.1:5173`.

### Using the Interface

1. **Preview and select a song** from the available music library
2. **Generate choreography** - SwarmGPT will create a first synchronized drone performance automatically
3. **Wait for the automatic safety filter**
4. **Preview the result** in the browser playback view
5. **Refine as needed** by providing additional prompts or modifications
6. **Deploy when satisfied** with the generated choreography

The system will automatically:
- Analyze the selected music for beats, rhythm, and musical features
- Generate safe, collision-free trajectories for your drone swarm
- Ensure all movements stay within the configured flight boundaries
- Synchronize drone movements with the musical timeline

### Ready for Deployment

Once you're happy with your generated choreography, you can proceed to deploy it on your physical drone swarm.

## Deployment

Use the deploy environment (`pixi shell -e deploy`) to run the following code. You need to start two terminals.

1. **Start the motion_capture_tracking lib**:
   ```bash
   ros2 launch motion_capture_tracking launch.py
   ```
2. **Launch SwarmGPT** as described in the [Launching the Interface](#launching-the-interface) section.
3. **Generate and preview choreography** using the web interface.
4. **Deploy to drones**: Once satisfied with the choreography, click the "Let the Crazyflies dance" button in the web interface to execute the performance on your physical drone swarm.


## Citing
If you find this work useful, compare it with other approaches or use some components, please cite
us as follows:

```bibtex
@article{schuck2025swarmgpt,
  title={SwarmGPT: Combining Large Language Models with Safe Motion Planning for Drone Swarm Choreography},
  author={Schuck, Martin and Dahanaggamaarachchi, Dinushka Orrin and Sprenger, Ben and Vyas, Vedant and Zhou, Siqi and Schoellig, Angela P.},
  journal={IEEE Robotics and Automation Letters},
  year={2025},
  publisher={IEEE}
}
```
