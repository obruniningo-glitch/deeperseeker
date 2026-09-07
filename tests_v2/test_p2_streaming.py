"""P2 tests: emitters, grammars, invariant checker."""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v2.streaming.emitter_anthropic import AnthropicEmitter
from v2.streaming.emitter_openai import OpenAIEmitter
from v2.streaming.validate import validate_anthropic, validate_openai
from v2.ir.events import (
    MessageStart, ThinkingDelta, TextDelta, ToolCallStart,
    ToolCallArgsDelta, ToolCallEnd, MessageEnd, SinkError, StopReason
)
from v2.tools.grammar import DSMLGrammar, FreeTextGrammar, GrammarContext, ToolInjection
from v2.tools.inject import build_tool_injection


async def test_anthropic_emitter_basic():
    """Anthropic emitter produces valid lifecycle for text-only."""
    emitter = AnthropicEmitter()
    events = [
        MessageStart(model="test-model"),
        TextDelta(text="Hello"),
        TextDelta(text=" world"),
        MessageEnd(stop_reason=StopReason.END_TURN, usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}),
    ]
    frames = []
    for ev in events:
        for frame in emitter.emit(ev):
            frames.append(frame)
    for frame in emitter.close():
        frames.append(frame)

    # Validate
    errors = validate_anthropic(iter(frames))
    assert not errors, f"Validation errors: {errors}"

    # Check frames contain expected events
    frame_types = []
    for f in frames:
        for line in f.decode().split('\n'):
            if line.startswith('event: '):
                frame_types.append(line[7:].strip())
                break

    expected = ["message_start", "content_block_start", "content_block_delta", "content_block_delta",
                "content_block_stop", "message_delta", "message_stop"]
    assert frame_types == expected, f"Expected {expected}, got {frame_types}"
    print("PASS test_anthropic_emitter_basic")


async def test_anthropic_emitter_thinking_then_text():
    """ThinkingDelta then TextDelta closes thinking block correctly."""
    emitter = AnthropicEmitter()
    events = [
        MessageStart(model="test-model"),
        ThinkingDelta(text="Let me think..."),
        TextDelta(text="The answer is 42."),
        MessageEnd(stop_reason=StopReason.END_TURN, usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}),
    ]
    frames = []
    for ev in events:
        for frame in emitter.emit(ev):
            frames.append(frame)
    for frame in emitter.close():
        frames.append(frame)

    errors = validate_anthropic(iter(frames))
    assert not errors, f"Validation errors: {errors}"
    print("PASS test_anthropic_emitter_thinking_then_text")


async def test_anthropic_emitter_tool_call():
    """Tool call lifecycle: start -> args -> end -> text."""
    emitter = AnthropicEmitter()
    events = [
        MessageStart(model="test-model"),
        ToolCallStart(index=0, id="call_0", name="get_weather"),
        ToolCallArgsDelta(index=0, partial_json='{"city": "'),
        ToolCallArgsDelta(index=0, partial_json='NYC"}'),
        ToolCallEnd(index=0),
        TextDelta(text="The weather is sunny."),
        MessageEnd(stop_reason=StopReason.TOOL_CALLS, usage={"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25}),
    ]
    frames = []
    for ev in events:
        for frame in emitter.emit(ev):
            frames.append(frame)
    for frame in emitter.close():
        frames.append(frame)

    errors = validate_anthropic(iter(frames))
    assert not errors, f"Validation errors: {errors}"
    print("PASS test_anthropic_emitter_tool_call")


async def test_openai_emitter_basic():
    """OpenAI emitter produces valid lifecycle for text-only."""
    emitter = OpenAIEmitter()
    events = [
        MessageStart(model="test-model"),
        TextDelta(text="Hello"),
        TextDelta(text=" world"),
        MessageEnd(stop_reason=StopReason.END_TURN, usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}),
    ]
    frames = []
    for ev in events:
        for frame in emitter.emit(ev):
            frames.append(frame)
    for frame in emitter.close():
        frames.append(frame)

    errors = validate_openai(iter(frames))
    assert not errors, f"Validation errors: {errors}"

    # Check for role emission and [DONE]
    has_role = False
    has_done = False
    for f in frames:
        if b'"role"' in f and b'assistant' in f:
            has_role = True
        if b"[DONE]" in f:
            has_done = True
    assert has_role, "Missing role emission"
    assert has_done, "Missing [DONE] sentinel"
    print("PASS test_openai_emitter_basic")


async def test_openai_emitter_reasoning_then_text():
    """Reasoning then text closes reasoning stream correctly."""
    emitter = OpenAIEmitter()
    events = [
        MessageStart(model="test-model"),
        ThinkingDelta(text="Let me think..."),
        TextDelta(text="The answer is 42."),
        MessageEnd(stop_reason=StopReason.END_TURN, usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}),
    ]
    frames = []
    for ev in events:
        for frame in emitter.emit(ev):
            frames.append(frame)
    for frame in emitter.close():
        frames.append(frame)

    errors = validate_openai(iter(frames))
    assert not errors, f"Validation errors: {errors}"
    print("PASS test_openai_emitter_reasoning_then_text")


async def test_openai_emitter_tool_call():
    """OpenAI tool call lifecycle."""
    emitter = OpenAIEmitter()
    events = [
        MessageStart(model="test-model"),
        ToolCallStart(index=0, id="call_0", name="get_weather"),
        ToolCallArgsDelta(index=0, partial_json='{"city": "'),
        ToolCallArgsDelta(index=0, partial_json='NYC"}'),
        ToolCallEnd(index=0),
        MessageEnd(stop_reason=StopReason.TOOL_CALLS, usage={"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25}),
    ]
    frames = []
    for ev in events:
        for frame in emitter.emit(ev):
            frames.append(frame)
    for frame in emitter.close():
        frames.append(frame)

    errors = validate_openai(iter(frames))
    assert not errors, f"Validation errors: {errors}"

    # Check tool_calls in output
    has_tool_calls = False
    for f in frames:
        if b'"tool_calls"' in f:
            has_tool_calls = True
            break
    assert has_tool_calls, "Missing tool_calls emission"
    print("PASS test_openai_emitter_tool_call")


async def test_dsmL_grammar_basic():
    """DSML grammar parses fenced JSON tool calls."""
    ctx = GrammarContext(tool_injection=ToolInjection(tools=[]))
    grammar = DSMLGrammar(ctx)

    # Test streaming
    events = [
        {"type": "text", "text": "I'll call the tool ```json {\"name\": \"get_weather\", \"arguments\": {\"city\": \"NYC\"}} ``` now."}
    ]

    # Use the grammar's transform
    async def event_stream():
        for e in events:
            yield type('obj', (object,), {'text': e['text']})()

    # Just test parse_final
    tool_calls, remaining = grammar.parse_final(
        'I will call ```json {"name": "get_weather", "arguments": {"city": "NYC"}} ``` done.'
    )
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "get_weather"
    assert '"city": "NYC"' in tool_calls[0]["arguments"]
    print("PASS test_dsmL_grammar_basic")


async def test_freetext_grammar_brace_balance():
    """FreeTextGrammar handles brace-balanced JSON."""
    ctx = GrammarContext(tool_injection=ToolInjection(tools=[]))
    grammar = FreeTextGrammar(ctx)

    tool_calls, remaining = grammar.parse_final(
        'Before { "name": "tool1", "arguments": {"x": 1} } after { "name": "tool2", "arguments": {"y": 2} } end.'
    )
    assert len(tool_calls) == 2
    assert tool_calls[0]["name"] == "tool1"
    assert tool_calls[1]["name"] == "tool2"
    print("PASS test_freetext_grammar_brace_balance")


async def test_tool_injection_prompt():
    """ToolInjection generates prompt sections."""
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        }
    }]
    injection = build_tool_injection(tools, "json")
    prompt = injection.to_prompt_section()
    assert "get_weather" in prompt
    assert "city" in prompt
    assert "required" in prompt
    print("PASS test_tool_injection_prompt")


async def main():
    await test_anthropic_emitter_basic()
    await test_anthropic_emitter_thinking_then_text()
    await test_anthropic_emitter_tool_call()
    await test_openai_emitter_basic()
    await test_openai_emitter_reasoning_then_text()
    await test_openai_emitter_tool_call()
    await test_dsmL_grammar_basic()
    await test_freetext_grammar_brace_balance()
    await test_tool_injection_prompt()
    print("\nAll P2 tests passed!")


if __name__ == "__main__":
    asyncio.run(main())