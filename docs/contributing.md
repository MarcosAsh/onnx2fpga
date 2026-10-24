# Writing the docs


Say what the user types and what happens. Every code example has to run as
written. Use real numbers from real runs. Plain ASCII, short and long sentences
mixed, no marketing, no hedging. A spec states the behaviour, the API, the
errors, and how we will know it is done. Anything undecided goes under "Open
questions" so somebody answers it.

To build these pages locally:

```
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

They publish to GitHub Pages on every push to `main` that touches `docs/` or
`compiler/`.
