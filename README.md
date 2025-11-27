# Quantum-Enhanced Multi-Agent Reinforcement Learning for Autonomous Satellite Task Scheduling

## Abstract
Cooperative multi-agent systems are becoming increasingly vital for complex, large-scale applications such as autonomous satellite constellation management. While Multi-Agent Reinforcement Learning (MARL) offers a powerful framework for learning decentralized policies, standard methods often struggle to efficiently handle the combinatorial nature of joint action selection. This paper explores a novel hybrid architecture, Q-MAPPO, that decouples representation learning from combinatorial reasoning. Q-MAPPO integrates a quantum-inspired optimization module, which solves a QUBO formulation of the agent-task assignment problem, into the decision-making loop of a strong MARL agent. This allows a neural network to learn task utilities while offloading the complex coordination logic to a dedicated solver. We conduct an empirical study comparing Q-MAPPO to the MAPPO baseline in a simulated satellite scheduling environment. Our findings reveal that Q-MAPPO exhibits significantly superior sample efficiency, achieving high performance much earlier in training. Although both methods converge to a similar level of final performance, we provide a detailed analysis of the trade-offs between learning speed, policy stability, and computational cost. We conclude that this hybrid approach is a promising direction and identify key challenges, particularly in developing more direct training methods for the utility-generating component, for future research.

## Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/ailabteam/qe-marl-asts.git
    cd qe-marl-asts
    ```

2.  Create and activate the Conda environment:
    ```bash
    conda env create -f environment.yml
    conda activate qemarl
    ```

