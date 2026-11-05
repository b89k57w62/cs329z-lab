from __future__ import annotations

from jsonschema import Draft202012Validator

from cs329z_hw1.types import ToolResult, ToolSpec


class ToolRegistry:
    def __init__(self, tools: list[ToolSpec]) -> None:
        for tool in tools:
            Draft202012Validator.check_schema(tool.parameters)
        self._tools = {tool.name: tool for tool in tools}
        self._validators = {tool.name: Draft202012Validator(tool.parameters) for tool in tools}

    def schemas(self) -> list[dict]:
        return [
            {"name": tool.name, "description": tool.description, "parameters": tool.parameters}
            for tool in self._tools.values()
        ]

    def validate(self, name: str, args: dict) -> ToolResult | None:
        if not isinstance(name, str) or name not in self._tools:
            return ToolResult("unknown_tool", f"Unknown tool {name!r}. Available tools: {list(self._tools)}.")
        problems = [error.message for error in self._validators[name].iter_errors(args)]
        if problems:
            return ToolResult("invalid_args", f"Invalid arguments for {name}: {'; '.join(problems)}.")
        return None

    def execute(self, name: str, args: dict) -> ToolResult:
        rejection = self.validate(name, args)
        if rejection is not None:
            return rejection
        try:
            return ToolResult("ok", str(self._tools[name].fn(**args)))
        except Exception as error:
            return ToolResult("tool_error", f"{name} raised {type(error).__name__}: {error}")
