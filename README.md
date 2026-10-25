# invertible-architectures-benchmark
Replication and extension of Kruse et al. (2021), "Benchmarking Invertible Architectures on Inverse Problems" (https://arxiv.org/abs/2101.10763). We reproduce INN, cINN and autoencoder baselines on the inverse kinematics and ballistics benchmarks, then explore newer generative methods (Flow Matching, Diffusion, and others).

Implemented: full-covariance MDN, standard autoencoder, conditional VAE, and
tied-transpose invertible autoencoder. INN/cINN and newer generative models are
future work.

For a shared Docker environment in which you select individual training and
evaluation commands, see [Docker usage](docs/docker.md). Checkpoints, evaluation
results and caches stay in the host project directory through a bind mount.
