"""离线测试：Read 读图片，以及发给模型时附上或省略。

不需要 API key：
    uv run pytest
"""
from __future__ import annotations

import base64
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from mini_agent.client import LLM, _TOOL_IMAGE_OMITTED  # noqa: E402
from mini_agent.compaction import serialize_for_summary  # noqa: E402
from mini_agent.config import Config, load_config, model_accepts_images  # noqa: E402
from mini_agent.context import Context  # noqa: E402
from mini_agent.loop import Agent  # noqa: E402
from mini_agent.tokens import message_tokens  # noqa: E402
from mini_agent.tools import build_tools  # noqa: E402
from mini_agent.tools.image import detect_image_mime  # noqa: E402

SANDBOX = ROOT / "sandbox" / "image_test"
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)



def check(name: str, cond: bool, detail: str = "") -> None:
    assert cond, f"{name}  {detail}"



def _chunk(kind: bytes, data: bytes) -> bytes:
    return len(data).to_bytes(4, "big") + kind + data + b"\x00\x00\x00\x00"


def test_image_suite() -> None:
    shutil.rmtree(SANDBOX, ignore_errors=True)
    SANDBOX.mkdir(parents=True)
    (SANDBOX / "dot.png").write_bytes(PNG_1X1)

    print("[魔数]")
    check("1x1 png", detect_image_mime(PNG_1X1) == "image/png")
    check("jpeg-ls 不当图片", detect_image_mime(b"\xff\xd8\xff\xf7" + b"\x00" * 8) is None)
    check("普通 jpeg", detect_image_mime(b"\xff\xd8\xff\xe0" + b"\x00" * 8) == "image/jpeg")
    sig = bytes((0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A))
    ihdr = _chunk(b"IHDR", b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00")
    actl = _chunk(b"acTL", b"\x00\x00\x00\x02\x00\x00\x00\x00")
    check("动画 png 拒绝", detect_image_mime(sig + ihdr + actl) is None)
    check("静止 png 头", detect_image_mime(sig + ihdr) == "image/png")

    print("[Read 图片]")
    reg = build_tools(Config(workdir=SANDBOX, enable_shell=False, enable_subagent=False))
    done = reg.execute("Read", {"file_path": "dot.png"})
    check("说明不含 base64", "Read image file [image/png]" in done.text and "iVBOR" not in done.text, done.text)
    check("带回一张图", len(done.images) == 1 and done.images[0]["mimeType"] == "image/png")
    check("dispatch 仍是纯文本", isinstance(reg.dispatch("Read", {"file_path": "dot.png"}), str))
    again = reg.execute("Read", {"file_path": "dot.png"})
    check("重复读图片不走 file_unchanged", "File unchanged" not in again.text and again.images)
    edit = reg.dispatch("Edit", {
        "file_path": "dot.png",
        "old_string": "x",
        "new_string": "y",
    })
    check("图片不满足先读再改", "必须先 Read" in edit, edit)

    broken = SANDBOX / "broken.png"
    broken.write_bytes(sig + ihdr)
    bad = reg.execute("Read", {"file_path": "broken.png"})
    check("坏图只回说明", not bad.images and "Image omitted" in bad.text, bad.text)

    from PIL import Image

    wide = SANDBOX / "wide.png"
    Image.new("RGB", (2400, 8), "red").save(wide)
    wide_out = reg.execute("Read", {"file_path": "wide.png"})
    check("超长边有缩放说明", "displayed at" in wide_out.text and "Multiply coordinates" in wide_out.text, wide_out.text)
    check("缩放后仍是一张图", len(wide_out.images) == 1)

    bmp = SANDBOX / "tiny.bmp"
    Image.new("RGB", (2, 2), "blue").save(bmp, format="BMP")
    bmp_out = reg.execute("Read", {"file_path": "tiny.bmp"})
    check("bmp 转 png", "image/bmp" in bmp_out.text and "image/png" in bmp_out.text, bmp_out.text)
    check("bmp 结果是 png", bmp_out.images and bmp_out.images[0]["mimeType"] == "image/png")

    gif = SANDBOX / "tiny.gif"
    Image.new("P", (2, 2), 1).save(gif, format="GIF")
    gif_out = reg.execute("Read", {"file_path": "tiny.gif"})
    check("小 gif 原样", gif_out.images and gif_out.images[0]["mimeType"] == "image/gif", gif_out.text)
    check("小 gif 不写转换说明", "converted" not in gif_out.text, gif_out.text)

    print("[请求格式]")
    stored = {
        "role": "tool",
        "tool_call_id": "c1",
        "content": [
            {"type": "text", "text": "Read image file [image/png]"},
            {"type": "image", "data": "AAAA", "mimeType": "image/png"},
        ],
    }
    history = [
        {"role": "user", "content": "看图"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "Read", "arguments": "{}"},
        }]},
        stored,
        {"role": "tool", "tool_call_id": "c2", "content": "纯文本"},
    ]
    blind = LLM(Config(api_key="test", model="deepseek-v4-pro"))._wire_messages(history)
    blob = json.dumps(blind)
    check("纯文本 deepseek 不附 data URL", "image_url" not in blob and "AAAA" not in blob, blob[:300])
    check("不支持时换成说明", any(
        m.get("role") == "tool" and _TOOL_IMAGE_OMITTED in str(m.get("content"))
        for m in blind
    ))
    check("会话里的图片还在", stored["content"][1]["data"] == "AAAA")
    check("同批的文本工具原样", any(
        m.get("tool_call_id") == "c2" and m.get("content") == "纯文本" for m in blind
    ))

    seen = LLM(Config(api_key="test", model="gpt-4o"))._wire_messages(history)
    tool_msgs = [m for m in seen if m.get("role") == "tool"]
    user_imgs = [m for m in seen if m.get("role") == "user" and isinstance(m.get("content"), list)]
    check("视觉模型的 tool 只留文字", tool_msgs[0]["content"] == "Read image file [image/png]", tool_msgs[0]["content"])
    check("图片跟在整批 tool 后面", len(user_imgs) == 1 and seen[-1] is user_imgs[0])
    check("data URL", user_imgs[0]["content"][1]["image_url"]["url"] == "data:image/png;base64,AAAA")
    economy = LLM(Config(api_key="test", model="deepseek-v4.1-flash-economy"))._wire_messages(history)
    economy_imgs = [m for m in economy if m.get("role") == "user" and isinstance(m.get("content"), list)]
    check("经济版 Flash 附图", economy_imgs and "AAAA" in json.dumps(economy_imgs))
    check("强制关闭盖过模型名", not Config(model="gpt-4o", vision=False).accepts_images)
    check("强制打开盖过模型名", Config(model="deepseek-v4-pro", vision=True).accepts_images)
    check("经济版 Flash 默认可看图", model_accepts_images("deepseek-v4.1-flash-economy"))
    check("火山方舟带日期的 id 也可看图", model_accepts_images("deepseek-v4-1-flash-economy-261017"))
    check("V4-Pro 不看图", not model_accepts_images("deepseek-v4-pro"))
    check("deepseek-chat 不看图", not model_accepts_images("deepseek-chat"))
    check("vl 模型默认可看图", model_accepts_images("qwen2.5-vl-72b"))

    big = "A" * 100_000
    image_msg = {"role": "tool", "content": [
        {"type": "text", "text": "note"},
        {"type": "image", "data": big, "mimeType": "image/png"},
    ]}
    check("token 不按 base64 长度计", message_tokens(image_msg) < 5000, message_tokens(image_msg))
    summary = serialize_for_summary([image_msg])
    check("摘要只记张数", "[1 image(s)]" in summary and "AAAA" not in summary and big[:20] not in summary, summary[:200])

    ctx = Context(Config(context_window=4000, compact_ratio=0.5))
    ctx.messages = [
        {"role": "system", "content": "s"},
        image_msg,
    ]
    class _NoLLM:
        def summarize(self, prompt: str) -> str:
            return "nope"

    ctx.compact(_NoLLM(), force=True)
    shrunk = ctx.messages[1]["content"]
    check("超限时丢掉图片块", isinstance(shrunk, str) and "已省略 1 张图片" in shrunk and big[:20] not in shrunk, shrunk[:180])

    print("[主循环把图片留在 tool 消息里]")
    events: list[dict] = []

    class ScriptLLM:
        def __init__(self):
            self.n = 0

        def chat_stream(self, messages, tools=None, on_delta=None, on_notice=None):
            self.n += 1
            if self.n == 1:
                return {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "r1",
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "arguments": json.dumps({"file_path": "dot.png"}),
                        },
                    }],
                    "_finish_reason": "stop",
                }
            return {"role": "assistant", "content": "看到了", "_finish_reason": "stop"}

        def summarize(self, prompt):
            return "摘要"

    agent = Agent(
        Config(workdir=SANDBOX, enable_shell=False, enable_subagent=False),
        reg,
        llm=ScriptLLM(),
        on_event=lambda kind, payload: events.append({"kind": kind, **payload}),
    )
    answer = agent.run("读这张图")
    tool_rows = [m for m in agent.ctx.messages if m.get("role") == "tool"]
    content = tool_rows[0]["content"]
    check("循环存的是图片块", isinstance(content, list) and content[-1]["type"] == "image", content if isinstance(content, str) else content[0])
    preview = next(e["result"] for e in events if e["kind"] == "tool_result")
    raw_b64 = content[-1]["data"]
    check("终端预览不含 base64", raw_b64[:20] not in preview and "1 张图片" in preview, preview)
    check("模型最终答复", answer == "看到了", answer)

    print("[环境变量开关]")
    old = os.environ.get("MINI_AGENT_VISION")
    try:
        os.environ["MINI_AGENT_VISION"] = "0"
        forced_off = load_config()
        check("VISION=0", forced_off.vision is False and not forced_off.accepts_images)
        os.environ["MINI_AGENT_VISION"] = "1"
        forced_on = load_config()
        check("VISION=1", forced_on.vision is True and forced_on.accepts_images)
    finally:
        if old is None:
            os.environ.pop("MINI_AGENT_VISION", None)
        else:
            os.environ["MINI_AGENT_VISION"] = old


