# Vendored SemTrans core for the PyTorch→Paddle demo

The source snapshot is SemTrans revision
`431bd23c9773ec4a3c2d9d924e433e93d5a5577b`. `provenance.json` records the
copied core file hashes and distinguishes unchanged files from local adapters.
The source tree's full experimental tests and unrelated backend frontends are
excluded. The retained runtime includes PyTorch/Paddle frontends, their shared
abstract IR, the NumPy backend, both framework definition trees, the JIT,
dynamic region tracer, differential validation, initial conversion prompts,
and repair prompt construction.

The four PyTorch/Paddle `nn_ops.yaml` and `tensors.yaml` files were copied
byte-for-byte. The local adapter changes are: bounded input-search parameters
and telemetry; fail-closed handling when no paired cases are generated;
route-specific prompt guidance for the explicit AvgPool teaching case;
correction of three `process_model_response` keyword mismatches; capture of
actual `SemanticTracer` region/first-mismatch records; and per-output
structure, shape, dtype, finite-value, elementwise-tolerance and error metrics.
The OpenAI-compatible adapter explicitly selects non-streaming transport and
parses non-streaming Chat Completions by response shape, avoiding brittle
Qwen/Kimi/Claude model-name protocol guesses.

The Paddle demo uses real `create_symbolnet`, `auto_repair_pipeline`,
`SemanticTracer`, and the vendored SemTrans-JIT formulas. Search budgets are
deliberately limited for an interactive demonstration. A passed run is
empirical evidence on the recorded sampled inputs/tolerance, not a formal proof
or reproduction of the paper's benchmark results.

SemTrans's source README states that its original code has no public license.
The frontend attribution to NNSmith and the associated Apache-2.0 text are
retained in `THIRD_PARTY_NOTICES.md` and `LICENSES/Apache-2.0.txt`. The copied
SemTrans source should not be pushed to a public remote until its redistribution
rights are confirmed.

Install the framework-independent SemTrans dependencies listed in
`requirements.txt` into the backend or a dedicated Python 3.9–3.12 environment.
Set `SEMTRANS_PYTHON` to that interpreter when it differs from the backend.
PyTorch and Paddle must also be installed in the selected worker environment.
