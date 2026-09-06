"""v2 P0 tests: IR model invariants + adapter round-trips.

Run: deeperseeker_env/Scripts/python.exe tests_v2/run_all.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from v2 import adapters
from v2.ir import (
    canonicalize_conversation,
    Conversation,
    ImageBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UnknownBlock,
)

# ---------- targeted regression tests (v1's historical bugs, at the IR) ----------


def test_anthropic_tool_result_becomes_tool_role():
    body = {
        "messages": [
            {"role": "user", "content": "run it"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu1", "name": "Bash", "input": {"command": "ls"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu1", "content": "file.txt"},
                {"type": "text", "text": "now list hidden files"},
            ]},
        ]
    }
    conv = adapters.to_ir(body, "anthropic")
    roles = [m.role for m in conv.messages]
    assert roles == ["user", "assistant", "user", "tool"], \
        "tool_result must land in its own role=tool message, never user text"
    tool_msg = conv.messages[-1]
    assert isinstance(tool_msg.blocks[0], ToolResultBlock)
    assert tool_msg.blocks[0].text() == "file.txt"
    # the sibling text stays a user message
    assert conv.messages[2].text() == "now list hidden files"


def test_openai_tool_result_roundtrip():
    body = {"messages": [
        {"role": "user", "content": "run"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "Bash", "arguments": '{"command": "ls"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "Bash", "content": "out"},
    ]}
    conv = adapters.to_ir(body, "openai")
    tu = [b for b in conv.messages[1].blocks if isinstance(b, ToolUseBlock)][0]
    assert tu.arguments == {"command": "ls"}, "JSON-string arguments must parse to dict"
    wire = adapters.from_ir(conv, "openai")
    tool_msg = [m for m in wire["messages"] if m["role"] == "tool"][0]
    assert tool_msg["tool_call_id"] == "c1" and tool_msg["content"] == "out"
    assert tool_msg["name"] == "Bash"


def test_anthropic_structured_tool_result_canonicalizes():
    """Canonical IR merges adjacent TextBlocks in tool_result content."""
    # Test 1: adjacent text blocks get merged
    body = {"messages": [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t", "content": [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]},
    ]}]}
    conv = adapters.to_ir(body, "anthropic")
    tr = conv.messages[0].blocks[0]
    assert [type(b).__name__ for b in tr.content] == ["TextBlock"]
    assert tr.content[0].text == "first\nsecond"
    wire = adapters.from_ir(conv, "anthropic")
    inner = wire["messages"][0]["content"][0]["content"]
    assert [c["type"] for c in inner] == ["text"]
    assert inner[0]["text"] == "first\nsecond"
    
    # Test 2: text separated by image preserves structure
    body2 = {"messages": [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t", "content": [
            {"type": "text", "text": "first"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}},
            {"type": "text", "text": "second"},
        ]},
    ]}]}
    conv2 = adapters.to_ir(body2, "anthropic")
    tr2 = conv2.messages[0].blocks[0]
    assert [type(b).__name__ for b in tr2.content] == ["TextBlock", "ImageBlock", "TextBlock"]
    wire2 = adapters.from_ir(conv2, "anthropic")
    inner2 = wire2["messages"][0]["content"][0]["content"]
    assert [c["type"] for c in inner2] == ["text", "image", "text"]


def test_unknown_wire_types_roundtrip_losslessly():
    body = {"messages": [{"role": "user", "content": [
        {"type": "web_search_result", "url": "https://x", "title": "X"}]}]}
    conv = adapters.to_ir(body, "anthropic")
    ub = conv.messages[0].blocks[0]
    assert isinstance(ub, UnknownBlock) and ub.type_name == "web_search_result"
    wire = adapters.from_ir(conv, "anthropic")
    assert wire["messages"][0]["content"][0]["type"] == "web_search_result"


def test_ir_version_and_metadata():
    conv = adapters.to_ir({"messages": [{"role": "user", "content": "hi"}],
                           "temperature": 0.3, "seed": 7}, "openai")
    assert conv.ir_version == 1
    assert conv.metadata == {"temperature": 0.3, "seed": 7}


def test_anthropic_block_array_system_prompt():
    body = {"system": [{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}],
            "messages": [{"role": "user", "content": "hi"}]}
    conv = adapters.to_ir(body, "anthropic")
    assert conv.system_text() == "part one\npart two"


def test_redacted_thinking_roundtrip():
    body = {"messages": [{"role": "assistant", "content": [
        {"type": "redacted_thinking"},
        {"type": "text", "text": "answer"}]}]}
    conv = adapters.to_ir(body, "anthropic")
    # Canonical sort order: TextBlock (1) before ThinkingBlock (3)
    blocks = conv.messages[0].blocks
    assert len(blocks) == 2
    assert isinstance(blocks[0], TextBlock) and blocks[0].text == "answer"
    assert isinstance(blocks[1], ThinkingBlock) and blocks[1].is_redacted
    wire = adapters.from_ir(conv, "anthropic")
    # Wire format uses canonical order
    assert wire["messages"][0]["content"][0]["type"] == "text"
    assert wire["messages"][0]["content"][1]["type"] == "redacted_thinking"


def test_legacy_function_call_maps_to_tool_use():
    body = {"messages": [{"role": "assistant", "content": None,
                          "function_call": {"name": "get_weather", "arguments": '{"city": "X"}'}}]}
    conv = adapters.to_ir(body, "openai")
    tu = [b for b in conv.messages[0].blocks if isinstance(b, ToolUseBlock)]
    assert tu and tu[0].name == "get_weather" and tu[0].arguments == {"city": "X"}


def test_image_detail_preserved():
    body = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://x/img.png", "detail": "high"}}]}]}
    conv = adapters.to_ir(body, "openai")
    assert conv.messages[0].blocks[0].detail == "high"
    wire = adapters.from_ir(conv, "openai")
    assert wire["messages"][0]["content"][0]["image_url"]["detail"] == "high"


# ---------- property tests: round-trips over generated conversations ----------

# Canonical IR never contains empty TextBlocks (wire "" vs absent is indistinguishable), so generated text is non-empty.
block_str = st.text(min_size=1, max_size=80)
tool_ids = st.sampled_from(["a", "b", "c"])
tool_names = st.sampled_from(["Bash", "Read", "Edit", "WebSearch"])
args_dicts = st.dictionaries(st.sampled_from(["command", "path", "query"]),
                             st.text(min_size=0, max_size=40), max_size=3)

representable_blocks = st.one_of(
    st.builds(TextBlock, text=block_str),
    st.builds(ToolUseBlock, id=tool_ids, name=tool_names, arguments=args_dicts),
    st.builds(ThinkingBlock, text=block_str),
)

# Assistant message: at most one TextBlock, multiple ToolUseBlocks, optional one ThinkingBlock
# Build as: (tool_calls...) + (text?) + (thinking?)
assistant_tool_calls = st.lists(
    st.builds(ToolUseBlock, id=tool_ids, name=tool_names, arguments=args_dicts),
    min_size=1, max_size=3,
).map(tuple)

assistant_text = st.one_of(
    st.just(()),
    st.builds(TextBlock, text=block_str).map(lambda b: (b,)),
)

assistant_thinking = st.one_of(
    st.just(()),
    st.builds(ThinkingBlock, text=block_str).map(lambda b: (b,)),
)

assistant_blocks = st.tuples(assistant_tool_calls, assistant_text, assistant_thinking).map(
    lambda x: x[0] + x[1] + x[2]
)

user_blocks = st.one_of(
    st.builds(TextBlock, text=block_str),
    st.builds(ImageBlock, source=st.just("url"), url=st.text(min_size=1, max_size=60),
              detail=st.just(None)),
)

# User messages: exactly one TextBlock (optional) + multiple ImageBlocks
# Since canonicalize merges adjacent TextBlocks, we only generate one
user_text = st.one_of(
    st.just(()),
    st.builds(TextBlock, text=block_str).map(lambda b: (b,)),
)

user_images = st.lists(
    st.builds(ImageBlock, source=st.just("url"), url=st.text(min_size=1, max_size=60), detail=st.just(None)),
    max_size=2,
).map(tuple)

user_blocks_tuple = st.tuples(user_text, user_images).map(lambda x: x[0] + x[1]).filter(lambda b: len(b) > 0)

user_msg = st.builds(Message, role=st.just("user"), blocks=user_blocks_tuple)


def _conv_strategy():
    tool_result_msg = st.builds(
        Message, role=st.just("tool"),
        blocks=st.tuples(st.builds(ToolResultBlock, tool_use_id=tool_ids,
                                   content=st.tuples(st.builds(TextBlock, text=block_str)))),
    )
    # Generate valid conversation sequences: alternate user/assistant, with optional tool messages
    def build_valid_sequence():
        # Use a composite strategy to build valid sequences
        pass
    
    # Simpler approach: use a list but filter to valid sequences
    msgs = st.lists(
        st.one_of(
            user_msg,
            st.builds(Message, role=st.just("assistant"), blocks=assistant_blocks),
            tool_result_msg,
        ),
        min_size=0, max_size=8,
    ).filter(lambda msgs: all(
        msgs[i].role != msgs[i+1].role for i in range(len(msgs)-1)
    ) if msgs else True)
    
    # Build Conversation, then apply canonicalize_conversation so generated
    # conversations are in canonical form (matching wire format round-trip output)
    base = st.builds(Conversation, system=st.one_of(st.none(), st.text(min_size=1, max_size=40)),
                     messages=msgs.map(tuple), metadata=st.fixed_dictionaries({}))
    return base.map(canonicalize_conversation)


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(_conv_strategy())
def test_openai_roundtrip(conv):
    back = adapters.to_ir(adapters.from_ir(conv, "openai"), "openai")
    assert back == conv, f"OpenAI round-trip lost information:\n{conv}\n-->\n{back}"


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(_conv_strategy())
def test_anthropic_roundtrip(conv):
    back = adapters.to_ir(adapters.from_ir(conv, "anthropic"), "anthropic")
    assert back == conv, f"Anthropic round-trip lost information:\n{conv}\n-->\n{back}"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {str(e)[:400]}")
        except Exception as e:  # hypothesis errors etc.
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {str(e)[:400]}")
    if failed:
        sys.exit(1)
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
