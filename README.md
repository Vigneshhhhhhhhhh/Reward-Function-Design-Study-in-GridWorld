# 🤖 Reward Function Design Study in GridWorld  

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![RL Environment: GridWorld](https://img.shields.io/badge/RL-GridWorld-green.svg)](https://en.wikipedia.org/wiki/Grid_world)

A comprehensive Reinforcement Learning study investigating how various **Reward Function Designs** impact the convergence speed and policy optimality of Value Iteration agents in stochastic environments.

---

## 🌍 Layman's Explanation: "The Robot & The Treats"
Imagine teaching a robot vacuum to find its charger in a messy room. The "GridWorld" is the room, and the "Brain" is the algorithm. This project explores how different **"Treats" (Rewards)** change how the robot learns:

*   **Sparse Reward**: "Good job" only when it hits the charger. (Hardest to learn).
*   **Dense Reward**: "Warmer... warmer" as it moves closer. (Fastest to learn).
*   **Negative Step**: "You lose a point for every wasted second." (Encourages speed).
*   **Deceptive Reward**: A fake treasure that tricks the robot away from the charger.

---

## 🚀 Key Features
*   **5 Distinct Reward Structures**: Sparse, Dense, Shaped, Deceptive, and NegativeStep.
*   **Stochastic MDP**: Implements realistic state-transition probabilities.
*   **Dynamic Programming**: Optimized Value Iteration and Policy Iteration solvers.
*   **Automated Visualization**: Generates high-DPI Heatmaps, Policy Maps, and Convergence Plots.
*   **Colab-Ready**: Interactive notebooks designed for instant cloud execution.

---

## 📂 Project Structure
- `environment/`: Core GridWorld MDP implementation.
- `algorithms/`: Value Iteration and Policy Iteration solvers.
- `rewards/`: The 5 mathematical reward designs.
- `experiments/`: Metric tracking and telemetry logs.
- `visualization/`: Automated graph and GIF generation.
- `analysis/`: Statistical analysis and LaTeX exporting.

---

## 🛠️ Installation & Quick Start

```bash
# Clone the repository
git clone https://github.com/Vigneshhhhhhhhhh/Reward-Function-Design-Study-in-GridWorld.git
cd Reward-Function-Design-Study-in-GridWorld

# Install dependencies
pip install -r requirements.txt

# Run the full experiment
python main.py --grid_size all --seeds 30 --save_figs
```

---

## 📓 Interactive Presentation
For a guided walkthrough of the results, use our Jupyter Notebooks:
*   [**Local Presentation**](Presentation.ipynb): Best for VS Code or local Jupyter Lab.
*   [**Colab Presentation**](Colab_Presentation.ipynb): Optimized for Google Colab (clones repo automatically).

---

## 👥 Contributors
This project was developed by:
*   **Vicky** ([@Vigneshhhhhhhhhh](https://github.com/Vigneshhhhhhhhhh))
*   **Arun**
*   **Shiva085000** ([@Shiva085000](https://github.com/Shiva085000))
*   **Sarvesh Raam** ([@sarvesh-raam](https://github.com/sarvesh-raam))

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
