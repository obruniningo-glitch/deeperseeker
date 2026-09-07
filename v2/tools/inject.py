"""Tool injection — converts tool schema to prompt sections (inverse of parsing)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class ToolInjection:
    """Injected tool schema for prompt construction.

    The pair (inject, parse) is the contract — every corpus entry
    is tested through BOTH directions.
    """
    tools: list[dict]  # OpenAI-style function definitions
    format_hint: str = "json"  # "json" | "dsml" | "freetext"

    def to_prompt_section(self) -> str:
        """Generate the tool-calling instruction section for the system prompt."""
        if not self.tools:
            return ""
        lines = ["## Available Tools", ""]
        for t in self.tools:
            fn = t.get("function", t)
            name = fn.get("name", "")
            desc = fn.get("description", "")
            params = fn.get("parameters", {})
            lines.append(f"### {name}")
            if desc:
                lines.append(desc)
            lines.append(f"```json")
            lines.append(f'{{"name": "{name}", "arguments": {json.dumps(params)}}}')
            lines.append(f"```")
            lines.append("")
        return "\n".join(lines)

    def to_dsml_prompt(self) -> str:
        """DeepSeek-family format with fullwidth-pipe variants."""
        if not self.tools:
            return ""
        lines = ["## 工具调用格式", ""]
        for t in self.tools:
            fn = t.get("function", t)
            name = fn.get("name", "")
            params = fn.get("parameters", {})
            lines.append(f"工具: {name}")
            lines.append(f"参数: {json.dumps(params, ensure_ascii=False)}")
            lines.append("调用示例:")
            lines.append("```json")
            lines.append(f'{{"name": "{name}", "arguments": {json.dumps(params)}}}')
            lines.append("```")
            lines.append("")
        return "\n".join(lines)


def build_tool_injection(tools: list[dict] | None, format_hint: str = "json") -> ToolInjection:
    """Build ToolInjection from OpenAI-style tool definitions."""
    if not tools:
        return ToolInjection(tools=[], format_hint=format_hint)
    normalized = []
    for t in tools:
        if "function" in t:
            normalized.append(t)
        else:
            # Assume it's already a function definition
            normalized.append({"type": "function", "function": t})
    return ToolInjection(tools=normalized, format_hint=format_hint)


import json