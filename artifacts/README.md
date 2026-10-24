# Paddle verification artifacts

Current regression evidence:

- `paddle-demo-validation.json`: all nine configured models, real
  conversion (`pre_generated=false`), generated-file syntax checks, CPU dummy
  forwards, finite outputs, expected shapes and source/artifact fingerprints.
- `paddle-browser-validation.json`: real browser conversions for DenseNet-121
  and Inception-v3, three completed evaluation stages, helper-code tabs, and
  Paddle-rule search (including an empty result). Browser and console errors
  must both be empty.
- `paddle-densenet-library-validation.png` and
  `paddle-inception-library-validation.png`: screenshots from that browser run.

`paddle-evaluation-running.png`, `paddle-evaluation-complete.png` and
`paddle-shufflenet-validation.json` are earlier integration snapshots, retained
for reference. Use the full demo/browser reports for the final regression.

The tested environment is Windows, Python 3.9 and Paddle 2.6.0. These checks
verify executability and static conversion structure, not dataset accuracy,
PyTorch/Paddle numerical equivalence or an inference-performance benchmark.

Reproduce (with the required Python environment activated):

```text
python -m unittest discover -s server/tests -p test_paddle_artifacts.py -v
python server/tools/verify_paddle_demos.py
```

For browser verification, start the backend on port 8200 and the production
frontend on port 3200, then open a dedicated headless Chrome CDP endpoint on
port 9234 and run:

```text
node app/scripts/verify-paddle-demos.cjs http://127.0.0.1:9234
```

The browser script uses Node's built-in WebSocket client (tested with Node 24).
