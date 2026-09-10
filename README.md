# Hybrid code search

## Index a codebase

```bash
python -m hybrid_search.cli index path/to/codebase --out artifacts/index
```

This scans `.py` and `.java` files, extracts functions, and stores the index in `artifacts/index`.

## Search

```bash
python -m hybrid_search.cli search --index artifacts/index --query "login function" --encoder unixcoder
```

The command prints ranked matches with lexical, semantic, and blended scores. Uses zero-anchored scaling by default; `--fusion rrf` enables rank based sorting.

Example output:

```text
1. auth.login_user [score=0.8421, lexical=0.7910, semantic=0.8823] src/auth.py:18
2. UserService.login [score=0.7014, lexical=0.7444, semantic=0.6712] src/UserService.java:42
```

## Benchmark

Create a JSON file like this:

```json
[
  {
    "query": "login function",
    "relevant": ["python:src/auth.py:18:login_user"]
  }
]
```

Then run with:

```bash
python -m hybrid_search.cli benchmark --index artifacts/index --queries queries.json --models codebert,unixcoder
```

The benchmark reports hit rate, MRR, recall@K, and precision@K for each model.
