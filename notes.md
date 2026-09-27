### Lewis et al., 2025

<p align="center">
  <img src="figures/emu-man.png" alt="emu-man" width="100" />
</p>

**Boltzmann distribution**: Probabilty function that gives likelihood that a system will occupy a specific state based on that state's energy and the termperature of the system.

**Boltzmann generator**: Generative model designed to sample molecular configurations from an approzimation to the Boltzmann distribution. 

For a molecular systems with configuration $x$ and potential energy $U(x)$:
$$p(x) \propto e^{-\beta U(x)} $$

where $\beta = \frac{1}{k_b T}$

This same equilibrium distribution MD / Monte Carlo methods try to sample over time. The generator though can generate equilibrium -like structures and can potentially jump between distant conformational states (in constrast to MD which evolve dynamics step by step).

**BioEmu** is designed to $\sim$ sample from the quilibrium distribution of protein conformations. 

`Input Sequence --> Single/Pair Representations (AlphaFold2 Evoformer) --> Denoising Diffusion Model --> Protein Structures` 

**Feynman-Kac steering** Inference time framework for steering diffusion models with reward functions. It samples a system of multiple interacting diffusion processes (particles) and resamples particles at intermediate steps based on scores computed using functions called *potentials*. Potentials are defined by using rewards for intermediate states and selected so that high value = particle will tield a high reward sample. 