import argparse
import sys
from .answer import Answer, AnswerError, answer
from .eval import EvalReport, load_eval_set, run_eval

EXIT_SUCCESS = 0
EXIT_FAILURE = 1

def _print_answer(result: Answer) -> None:
    print(result.answer)
    for chunk_id in result.citations:
        print(f"  - {chunk_id}")
    if result.attempts:
        print(f"(LLM attempts: {result.attempts})")
    else:
        print("(no LLM call: top retrieval score below MIN_SCORE_TO_ANSWER)")

def _print_report(report: EvalReport) -> None:
    for result in report.retrievals:
        print(f"rank={result.rank} gated={result.gated}  {result.case.question}")
    for result in report.refusals:
        print(f"refused={result.refused}  {result.case.question}")
    print(f"\nchunk_max_chars={report.chunk_max_chars} top_k={report.top_k}")
    print(
        f"recall@{report.top_k}={report.recall_at_k:.2f} mrr={report.mrr:.2f} "
        f"gated={report.gated_count}/{len(report.retrievals)} refusal_rate={report.refusal_rate:.2f}"
    )

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rag", description="BM25 RAG over SecondBrain notes")
    commands = parser.add_subparsers(dest="command", required=True)
    ask = commands.add_parser("ask", help="answer a question with chunk citations")
    ask.add_argument("question")
    commands.add_parser("eval", help="report recall@k, MRR and refusal rate on the eval set")
    return parser

def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "ask":
            _print_answer(answer(args.question))
        else:
            _print_report(run_eval(load_eval_set()))
    except (AnswerError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_FAILURE
    return EXIT_SUCCESS

if __name__ == "__main__":
    sys.exit(main())
