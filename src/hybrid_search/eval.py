import json
from dataclasses import dataclass
from pathlib import Path

from .search import HybridSearcher


@dataclass(frozen=True)
class QueryItem:
    query: str
    relevant: list[str]


def load_queries(path: str | Path) -> list[QueryItem]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        QueryItem(query=item["query"], relevant=list(item["relevant"])) for item in raw
    ]


def evaluate(
    searcher: HybridSearcher, queries: list[QueryItem], top_k: int = 10
) -> dict[str, float]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    mrr = 0.0
    recall_at_k = 0.0
    precision_at_k = 0.0
    hits = 0
    for item in queries:
        results = searcher.search(item.query, top_k=top_k)
        ids = [result.function_id for result in results]
        rank = next(
            (
                index + 1
                for index, function_id in enumerate(ids)
                if function_id in item.relevant
            ),
            None,
        )
        if rank is not None:
            hits += 1
            mrr += 1.0 / rank
            recall_at_k += sum(
                1 for function_id in item.relevant if function_id in ids
            ) / max(len(item.relevant), 1)
        precision_at_k += (
            sum(1 for function_id in ids if function_id in item.relevant) / top_k
        )
    total = max(len(queries), 1)
    return {
        "queries": float(len(queries)),
        "hit_rate": hits / total,
        "mrr": mrr / total,
        "recall_at_k": recall_at_k / total,
        "precision_at_k": precision_at_k / total,
    }


def evaluate_at_cutoffs(
    searcher: HybridSearcher,
    queries: list[QueryItem],
    cutoffs: tuple[int, ...] = (1, 3),
) -> dict[str, float]:
    if not cutoffs or any(cutoff <= 0 for cutoff in cutoffs):
        raise ValueError("cutoffs must contain only positive values")
    max_k = max(cutoffs)
    totals = {cutoff: 0.0 for cutoff in cutoffs}
    precision_at_3 = 0.0
    mrr = 0.0
    for item in queries:
        results = searcher.search(item.query, top_k=max_k)
        ids = [result.function_id for result in results]
        rank = next(
            (
                i + 1
                for i, function_id in enumerate(ids)
                if function_id in item.relevant
            ),
            None,
        )
        if rank is not None:
            mrr += 1.0 / rank
        for cutoff in cutoffs:
            selected = ids[:cutoff]
            totals[cutoff] += sum(
                1 for function_id in item.relevant if function_id in selected
            ) / max(len(item.relevant), 1)
        selected = ids[:3]
        precision_at_3 += (
            sum(1 for function_id in selected if function_id in item.relevant) / 3
        )
    total = max(len(queries), 1)
    report = {
        "queries": float(len(queries)),
        "mrr": mrr / total,
        "precision_at_3": precision_at_3 / total,
    }
    report.update(
        {f"recall_at_{cutoff}": value / total for cutoff, value in totals.items()}
    )
    return report


def run_benchmark_grid(
    index,
    queries: list[QueryItem],
    model_names: list[str],
    alphas: tuple[float, ...] = (0.0, 0.5, 1.0),
    cutoffs: tuple[int, ...] = (1, 3),
    fusion: str = "zero_anchored",
    rrf_k: int = 60,
) -> dict[str, dict[str, dict[str, float | str]]]:
    from .embeddings import make_encoder

    reports: dict[str, dict[str, dict[str, float | str]]] = {}
    configurations = (
        [("rrf", None)]
        if fusion == "rrf"
        else [(f"alpha_{alpha:g}", alpha) for alpha in alphas]
    )
    for alpha_key, alpha in configurations:
        reports[alpha_key] = {}
        for model_name in model_names:
            try:
                encoder = make_encoder(model_name)
            except RuntimeError as exc:
                reports[alpha_key][model_name] = {
                    "status": "unavailable",
                    "error": str(exc),
                }
                continue
            options = {} if alpha is None else {"alpha": alpha}
            searcher = HybridSearcher(
                index, encoder, fusion=fusion, rrf_k=rrf_k, **options
            )
            reports[alpha_key][model_name] = {
                "status": "ok",
                **evaluate_at_cutoffs(searcher, queries, cutoffs),
            }
    return reports


def run_benchmark(
    index,
    queries: list[QueryItem],
    model_names: list[str],
    alpha: float = 0.5,
    top_k: int = 10,
    fusion: str = "zero_anchored",
    rrf_k: int = 60,
) -> dict[str, dict[str, float | str]]:
    from .embeddings import make_encoder
    from .search import HybridSearcher

    report: dict[str, dict[str, float | str]] = {}
    for model_name in model_names:
        try:
            encoder = make_encoder(model_name)
        except RuntimeError as exc:
            report[model_name] = {"status": "unavailable", "error": str(exc)}
            continue
        searcher = HybridSearcher(
            index, encoder, alpha=alpha, fusion=fusion, rrf_k=rrf_k
        )
        report[model_name] = {
            "status": "ok",
            **evaluate(searcher, queries, top_k=top_k),
        }
    return report
