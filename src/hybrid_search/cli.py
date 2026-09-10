import argparse
import json

from .embeddings import make_encoder
from .eval import load_queries, run_benchmark, run_benchmark_grid
from .extractors import CodebaseIndexer
from .index import HybridIndex
from .search import HybridSearcher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    index_cmd = sub.add_parser("index")
    index_cmd.add_argument("path")
    index_cmd.add_argument("--out", default="artifacts/index")
    index_cmd.add_argument("--encoder", default="unixcoder")

    search_cmd = sub.add_parser("search")
    search_cmd.add_argument("--index", required=True)
    search_cmd.add_argument("--query", required=True)
    search_cmd.add_argument("--encoder", default="unixcoder")
    search_cmd.add_argument(
        "--alpha",
        type=float,
        default=0.5,
    )
    search_cmd.add_argument(
        "--fusion",
        choices=["zero_anchored", "rrf"],
        default="zero_anchored",
    )
    search_cmd.add_argument("--rrf-k", type=int, default=60)
    search_cmd.add_argument("--top-k", type=int, default=10)

    bench_cmd = sub.add_parser("benchmark")
    bench_cmd.add_argument("--index", required=True)
    bench_cmd.add_argument("--queries", required=True)
    bench_cmd.add_argument("--models", default="codebert,unixcoder")
    bench_cmd.add_argument(
        "--alpha",
        type=float,
        default=0.5,
    )
    bench_cmd.add_argument(
        "--fusion",
        choices=["zero_anchored", "rrf"],
        default="zero_anchored",
    )
    bench_cmd.add_argument("--rrf-k", type=int, default=60)
    bench_cmd.add_argument("--alphas")
    bench_cmd.add_argument("--top-k", type=int, default=10)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "index":
        indexer = CodebaseIndexer()
        functions = indexer.index_path(args.path)
        encoder = make_encoder(args.encoder)
        index = HybridIndex.build(functions, encoder)
        index.save(args.out)
        print(json.dumps({"functions": len(functions), "out": args.out}, indent=2))
        return 0

    if args.command == "search":
        index = HybridIndex.load(args.index)
        encoder = make_encoder(args.encoder)
        searcher = HybridSearcher(
            index, encoder, alpha=args.alpha, fusion=args.fusion, rrf_k=args.rrf_k
        )
        results = searcher.search(args.query, top_k=args.top_k)
        for result in results:
            print(
                f"{result.rank}. {result.qualified_name} "
                f"[score={result.score:.4f}, lexical={result.lexical_score:.4f}, semantic={result.semantic_score:.4f}] "
                f"{result.file_path}:{result.start_line}"
            )
        return 0

    if args.command == "benchmark":
        index = HybridIndex.load(args.index)
        queries = load_queries(args.queries)
        models = [item.strip() for item in args.models.split(",") if item.strip()]
        if args.fusion == "rrf" or args.alphas:
            alphas = (
                ()
                if args.fusion == "rrf"
                else tuple(
                    float(item.strip())
                    for item in args.alphas.split(",")
                    if item.strip()
                )
            )
            report = run_benchmark_grid(
                index,
                queries,
                models,
                alphas=alphas,
                cutoffs=(1, args.top_k),
                fusion=args.fusion,
                rrf_k=args.rrf_k,
            )
        else:
            report = run_benchmark(
                index,
                queries,
                models,
                alpha=args.alpha,
                top_k=args.top_k,
                fusion=args.fusion,
                rrf_k=args.rrf_k,
            )
        print(json.dumps(report, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    main()
