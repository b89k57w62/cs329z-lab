import sys
from extract import SCHEMA, ExtractionError, extract, load_samples

MODES = {"feedback": True, "no feedback": False}

def run_sample(sample: dict, feedback: bool) -> dict:
    try:
        _, attempts = extract(sample["text"], SCHEMA, feedback=feedback)
        return {"id": sample["id"], "success": True, "attempts": attempts, "errors": []}
    except ExtractionError as error:
        return {"id": sample["id"], "success": False, "attempts": error.attempts, "errors": error.errors}

def run_mode(name: str, samples: list[dict], feedback: bool) -> list[dict]:
    runs = []
    for sample in samples:
        run = run_sample(sample, feedback)
        print(f"[{name}] #{run['id']} success={run['success']} attempts={run['attempts']}", file=sys.stderr)
        runs = [*runs, run]
    return runs

def summarize(runs: list[dict]) -> dict:
    total = len(runs)
    return {
        "first_try": sum(run["success"] and run["attempts"] == 1 for run in runs) / total,
        "final": sum(run["success"] for run in runs) / total,
        "avg_attempts": sum(run["attempts"] for run in runs) / total,
    }

def format_retries(name: str, runs: list[dict]) -> list[str]:
    retried = [run for run in runs if run["attempts"] > 1 or not run["success"]]
    if not retried:
        return [f"- {name}: none"]
    return [
        f"- {name}: #{run['id']} attempts={run['attempts']}"
        + ("" if run["success"] else f" FAILED {run['errors']}")
        for run in retried
    ]

def format_report(results: dict[str, list[dict]]) -> str:
    header = [
        "| Mode | First-try success | Final success | Avg attempts |",
        "|---|---|---|---|",
    ]
    rows = [
        f"| {name} | {stats['first_try']:.1%} | {stats['final']:.1%} | {stats['avg_attempts']:.2f} |"
        for name, stats in ((name, summarize(runs)) for name, runs in results.items())
    ]
    retries = [line for name, runs in results.items() for line in format_retries(name, runs)]
    return "\n".join([*header, *rows, "", "Retried or failed cases:", *retries])

def main() -> None:
    samples = load_samples()
    results = {name: run_mode(name, samples, feedback) for name, feedback in MODES.items()}
    print(format_report(results))

if __name__ == "__main__":
    main()
