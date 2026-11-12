# FR3 FIF/CFMPE implementation status

Prepared the existing FIF and prior-conditioned FMPE implementations for
the professor's FR3 6-D pose benchmark. The 5-D task is supported by the
same dimension-aware paths. Protocol details and commands are in
[FIF/CFMPE on FR3](../fr3_fif_cfmpe.md).

The requested headline setup is ~3M parameters, 1M training samples, and
50 epochs, following Emine's FR3 FMPE work. FIF uses a one-dimensional
latent for the six independent pose constraints. CFMPE uses a proposed
family of contracted uniform joint priors with a 25% nominal-prior anchor;
benchmark evaluation is restricted to the original full-limits prior.

Added validation/test routing, experiment plans that require explicit
execution, learning/posterior plots, descriptive parameter-scaling analysis,
and regression tests. Reviewed Emine's FR3 and CFMPE commits and Umut's
ILNN experiment automation.

FR3 adaptations are isolated in an experiment-local adapter and calibration
entry point. The shared data/prior registry, robot task classes, ground-truth
module, and ConditionalFMPE class retain their original contents. No runtime
monkey-patching or shared registry mutation is used.

**No experiments were run.** No trained FR3 FIF/CFMPE checkpoint, generated
reference cache, measured metric, scaling exponent, or comparison claim is
provided. Runtime tests are also unexecuted. Existing colleagues' results
were left intact.
