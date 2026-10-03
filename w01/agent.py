import ast, json, operator, re, subprocess, sys
from dataclasses import dataclass
from pathlib import Path

LLM_TIMEOUT_SECONDS = 60
MAX_STEPS = 8
DATA_DIR = (Path(__file__).parent / "data").resolve()
BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
OBSERVATION_PATTERN = re.compile(r"^Observation:", re.MULTILINE)
ACTION_PATTERN = re.compile(r"^Action:\s*(\w+)\[(.*)\]\s*$", re.MULTILINE)
FINAL_PATTERN = re.compile(r"^Final:\s*(.+)", re.MULTILINE | re.DOTALL)

def call_llm(system: str, transcript: str) -> str:
    result = subprocess.run(
        ["claude", "-p", transcript,
         "--system-prompt", system,
         "--model", "haiku",
         "--tools", "",
         "--setting-sources", "",
         "--no-session-persistence",
         "--output-format", "json"],
        stdin=subprocess.DEVNULL, capture_output=True, text=True,
        timeout=LLM_TIMEOUT_SECONDS, check=True,
    )
    return json.loads(result.stdout)["result"]

def _evaluate(node: ast.AST) -> int | float:
    match node:
        case ast.Expression(body=body):
            return _evaluate(body)
        case ast.Constant(value=value) if type(value) in (int, float):
            return value
        case ast.BinOp(left=left, op=op, right=right) if type(op) in BINARY_OPERATORS:
            return BINARY_OPERATORS[type(op)](_evaluate(left), _evaluate(right))
        case ast.UnaryOp(op=op, operand=operand) if type(op) in UNARY_OPERATORS:
            return UNARY_OPERATORS[type(op)](_evaluate(operand))
        case _:
            raise ValueError(f"Unsupported expression: {ast.dump(node)}")

def calculate(expr: str) -> str:
    return str(_evaluate(ast.parse(expr, mode="eval")))

def read_file(path: str) -> str:
    target = (DATA_DIR / path).resolve()
    if not target.is_relative_to(DATA_DIR):
        raise PermissionError(f"Access denied: {path} is outside the data directory")
    return target.read_text(encoding="utf-8")

TOOLS = {"calculate": calculate, "read_file": read_file}

SYSTEM_PROMPT = """You are a ReAct agent. Answer the question by interleaving Thought, Action, and Observation steps.

Tools:
- calculate[expr]: evaluate an arithmetic expression using + - * / and parentheses. Example: calculate[(12+30)*2]
- read_file[path]: read a text file from the data directory. Monthly spending files are named spending-YYYY-MM.txt. Example: read_file[spending-2026-01.txt]

Each turn, output exactly one of these two forms and nothing else:

Thought: <your reasoning about what to do next>
Action: <tool>[<argument>]

Thought: <your reasoning>
Final: <the final answer>

Rules:
- Output one Action per turn, then stop. Never write an Observation yourself; the system provides it.
- Use calculate for all arithmetic instead of computing in your head.
- If an Observation is an error, reason about the cause and try a different Action, or give a Final answer explaining what is missing.

Example:
Question: What is the total weight listed in boxes.txt?
Thought: I need to read boxes.txt first.
Action: read_file[boxes.txt]
Observation: box A 12kg
box B 30kg
Thought: I will add the two weights.
Action: calculate[12+30]
Observation: 42
Thought: The total weight is 42 kg.
Final: 42 kg
"""

@dataclass(frozen=True)
class Action:
    text: str
    tool: str
    arg: str

@dataclass(frozen=True)
class Final:
    text: str
    answer: str

def truncate_at_observation(reply: str) -> str:
    return OBSERVATION_PATTERN.split(reply, maxsplit=1)[0].strip()

def parse_reply(reply: str) -> Action | Final:
    text = truncate_at_observation(reply)
    if action := ACTION_PATTERN.search(text):
        return Action(text[:action.end()], action.group(1), action.group(2))
    if final := FINAL_PATTERN.search(text):
        return Final(text, final.group(1).strip())
    raise ValueError("Invalid format: reply must contain 'Action: tool[arg]' or 'Final: answer'")

def run_tool(action: Action) -> str:
    tool = TOOLS.get(action.tool)
    if tool is None:
        raise ValueError(f"Unknown tool: {action.tool}. Available tools: {', '.join(TOOLS)}")
    return tool(action.arg)

def run_agent(question: str) -> str | None:
    transcript = f"Question: {question}"
    for step in range(1, MAX_STEPS + 1):
        reply = call_llm(SYSTEM_PROMPT, transcript)
        print(f"--- step {step} ---")
        try:
            parsed = parse_reply(reply)
        except ValueError as error:
            observation = f"Error: {error}"
            print(f"{reply.strip()}\nObservation: {observation}")
            transcript = f"{transcript}\n{truncate_at_observation(reply)}\nObservation: {observation}"
            continue
        match parsed:
            case Final(answer=answer):
                print(parsed.text)
                return answer
            case Action():
                try:
                    observation = run_tool(parsed)
                except Exception as error:
                    observation = f"Error: {error}"
                print(f"{parsed.text}\nObservation: {observation}")
                transcript = f"{transcript}\n{parsed.text}\nObservation: {observation}"
    return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit('Usage: python3 agent.py "<question>"')
    answer = run_agent(sys.argv[1])
    print(f"=== Final answer: {answer}" if answer is not None else f"=== No answer within {MAX_STEPS} steps")
