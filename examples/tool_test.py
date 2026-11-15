"""离线测试：移植的 Read/Edit/Write/Glob/Grep 五件套 + 工具框架。

不需要 API key，直接跑：
    uv run python examples/tool_test.py
覆盖：read-before-write、staleness 闸（含 mtime 假阳性内容兜底）、
CRLF/UTF-16 保真、弯引号容错、file_unchanged 去重、分页、二进制/UNC/
越界拒绝、Grep 三种模式与翻页、Glob 排序与截断、semantic_bool 容错。
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_agent.config import Config  # noqa: E402
from mini_agent.tools import build_tools  # noqa: E402

SANDBOX = ROOT / "sandbox" / "tool_test"

PASSED = FAILED = 0
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = ""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ok  {name}")
    else:
        FAILED += 1
        FAILURES.append(name)
        print(f"  FAIL {name}  {detail}")


def make_registry():
    cfg = Config(workdir=SANDBOX, enable_shell=False)
    return build_tools(cfg)


def d(reg, name, **kw) -> str:
    return reg.dispatch(name, json.dumps(kw, ensure_ascii=False))


def main() -> bool:
    shutil.rmtree(SANDBOX, ignore_errors=True)
    SANDBOX.mkdir(parents=True)
    reg = make_registry()

    # ---------- Write ----------
    print("[Write]")
    r = d(reg, "Write", file_path="hello.py", content="def greet():\n    return 'hi'\n")
    check("新建成功", "已新建" in r and "hello.py" in r, r)
    check("新建内容正确", (SANDBOX / "hello.py").read_text(encoding="utf-8")
          == "def greet():\n    return 'hi'\n")

    sub = d(reg, "Write", file_path="deep/nested/x.txt", content="nested\n")
    check("父目录自动创建", "已新建" in sub and (SANDBOX / "deep/nested/x.txt").exists())

    # 覆盖未读过的文件 → read-before-write 拒绝
    (SANDBOX / "unread.py").write_text("original\n", encoding="utf-8")
    r = d(reg, "Write", file_path="unread.py", content="hacked\n")
    check("覆盖未读文件被拒绝", r.startswith("ERROR") and "必须先 Read" in r, r)

    # ---------- Read ----------
    print("[Read]")
    r = d(reg, "Read", file_path="hello.py")
    check("cat -n 行号 + 内容", "     1\tdef greet():" in r and "     2\t    return 'hi'" in r, r)
    check("footer 行数", "[共 2 行，已显示 1-2]" in r, r)

    big = SANDBOX / "big.py"
    with open(big, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("".join(f"line {i}\n" for i in range(1, 501)))
    r = d(reg, "Read", file_path="big.py", offset=11, limit=5)
    check("分页 offset/limit", "    11\tline 11" in r and "    15\tline 15" in r
          and "line 16" not in r, r)
    check("分页 footer + 续读提示", "[共 500 行，已显示 11-15]" in r and "offset=16" in r, r)

    r = d(reg, "Read", file_path="big.py", offset=9999)
    check("offset 越界警告", "<system-reminder>" in r and "共 500 行" in r, r)

    (SANDBOX / "empty.txt").write_text("", encoding="utf-8")
    r = d(reg, "Read", file_path="empty.txt")
    check("空文件警告", "内容为空" in r, r)

    r = d(reg, "Read", file_path="no_such.py")
    check("不存在报错", r.startswith("ERROR") and "不存在" in r, r)

    (SANDBOX / "real_name.py").write_text("x\n", encoding="utf-8")
    r = d(reg, "Read", file_path="real_name.pyc")
    check("Did-you-mean 建议", "real_name.py" in r, r)

    # 读/搜索工具允许任意路径（范围限制只作用于写工具）
    outside = SANDBOX.parent / "outside_area"
    outside.mkdir(exist_ok=True)
    (outside / "out.txt").write_text("外面的文件\n", encoding="utf-8")
    (outside / "out2.txt").write_text("第二个外面文件\n", encoding="utf-8")
    r = d(reg, "Read", file_path=str(outside / "out.txt"))
    check("绝对路径读工作目录外", "外面的文件" in r, r)
    r = d(reg, "Read", file_path="../outside_area/out2.txt")
    check("相对 .. 读出工作目录", "第二个外面文件" in r, r)
    r = d(reg, "Glob", pattern="*.txt", path=str(outside))
    check("Glob 搜工作目录外", "out.txt" in r, r)
    r = d(reg, "Grep", pattern="外面", path=str(outside), output_mode="content")
    check("Grep 搜工作目录外", "out.txt:1:外面的文件" in r, r)

    # 写工具仍受 workdir 约束
    r = d(reg, "Write", file_path="../outside_area/hack.txt", content="x\n")
    check("Write 越界拒绝", r.startswith("ERROR") and "越界" in r, r)
    (outside / "edit_me.txt").write_text("v\n", encoding="utf-8")
    r = d(reg, "Edit", file_path=str(outside / "edit_me.txt"),
          old_string="v", new_string="w")
    check("Edit 越界拒绝", r.startswith("ERROR") and "越界" in r, r)

    r = d(reg, "Read", file_path="\\\\server\\share\\x.txt")
    check("UNC 拒绝", r.startswith("ERROR") and "UNC" in r, r)

    (SANDBOX / "bin.dat").write_bytes(b"\x00\x01\x02binary")
    r = d(reg, "Read", file_path="bin.dat")
    check("二进制拒绝", r.startswith("ERROR") and "二进制" in r, r)

    huge = SANDBOX / "huge.txt"
    with open(huge, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("".join(f"{i:08x} data\n" for i in range(12000)))  # ≈ 300KB 多行
    r = d(reg, "Read", file_path="huge.txt")
    check("整读字节帽拒绝", r.startswith("ERROR") and "offset/limit" in r, r)
    r = d(reg, "Read", file_path="huge.txt", offset=1, limit=10)
    check("范围读绕过字节帽", "     1\t" in r and "已显示 1-10" in r, r)

    # ---------- file_unchanged 去重 ----------
    print("[去重]")
    d(reg, "Read", file_path="hello.py")
    r = d(reg, "Read", file_path="hello.py")
    check("重复读取回 stub", "File unchanged since last read" in r, r)
    d(reg, "Edit", file_path="hello.py", old_string="return 'hi'",
       new_string="return 'hello'")
    r = d(reg, "Read", file_path="hello.py")
    check("Edit 后不去重（拿到新内容）", "return 'hello'" in r, r)

    # ---------- Edit ----------
    print("[Edit]")
    r = d(reg, "Edit", file_path="hello.py", old_string="return 'hello'",
          new_string="return 'yo'")
    check("替换成功", "已编辑" in r and "-1/+1" in r, r)
    check("替换落盘", "return 'yo'" in (SANDBOX / "hello.py").read_text(encoding="utf-8"))

    (SANDBOX / "multi.py").write_text("a = 1\nb = 1\n", encoding="utf-8")
    d(reg, "Read", file_path="multi.py")
    r = d(reg, "Edit", file_path="multi.py", old_string="= 1", new_string="= 2")
    check("多命中报错并提示", "命中 2 处" in r and "replace_all" in r, r)
    r = d(reg, "Edit", file_path="multi.py", old_string="= 1", new_string="= 2",
          replace_all=True)
    check("replace_all 生效", "已编辑" in r and "2 处" in r, r)
    check("replace_all 落盘", (SANDBOX / "multi.py").read_text(encoding="utf-8") == "a = 2\nb = 2\n")

    r = d(reg, "Edit", file_path="hello.py", old_string="不存在的文本",
          new_string="x")
    check("未命中报错", r.startswith("ERROR") and "未找到 old_string" in r, r)

    r = d(reg, "Edit", file_path="hello.py", old_string="x", new_string="x")
    check("old==new 早退", r.startswith("ERROR") and "相同" in r, r)

    # 没读过的新文件不能直接 Edit
    (SANDBOX / "fresh.py").write_text("v = 1\n", encoding="utf-8")
    r = d(reg, "Edit", file_path="fresh.py", old_string="v = 1", new_string="v = 2")
    check("未读文件 Edit 拒绝", r.startswith("ERROR") and "先 Read" in r, r)

    # semantic_bool：模型把布尔写成字符串
    (SANDBOX / "sem.py").write_text("f = 1\nf = 1\n", encoding="utf-8")
    d(reg, "Read", file_path="sem.py")
    r = d(reg, "Edit", file_path="sem.py", old_string="f = 1", new_string="f = 9",
          replace_all="true")  # 字符串布尔
    check("字符串布尔容错", "2 处" in r, r)

    # ---------- CRLF / UTF-16 保真 ----------
    print("[编码/行尾保真]")
    crlf = SANDBOX / "crlf.py"
    crlf.write_bytes(b"a = 1\r\nb = 1\r\n")
    d(reg, "Read", file_path="crlf.py")
    r = d(reg, "Edit", file_path="crlf.py", old_string="b = 1", new_string="b = 2")
    check("CRLF 编辑成功", "已编辑" in r, r)
    check("CRLF 写回保真", crlf.read_bytes() == b"a = 1\r\nb = 2\r\n",
          repr(crlf.read_bytes()))

    u16 = SANDBOX / "u16.txt"
    u16.write_bytes(b"\xff\xfe" + "hello\nworld\n".encode("utf-16-le"))
    r = d(reg, "Read", file_path="u16.txt")
    check("UTF-16 读取", "hello" in r and "world" in r, r)
    r = d(reg, "Edit", file_path="u16.txt", old_string="world", new_string="earth")
    check("UTF-16 编辑成功", "已编辑" in r, r)
    raw = u16.read_bytes()
    check("UTF-16 写回保真（BOM + 编码）",
          raw.startswith(b"\xff\xfe") and "earth".encode("utf-16-le") in raw,
          repr(raw[:20]))

    # ---------- 弯引号容错 ----------
    print("[弯引号]")
    curly = SANDBOX / "curly.txt"
    curly.write_text("他说“你好”\n", encoding="utf-8")
    d(reg, "Read", file_path="curly.txt")
    r = d(reg, "Edit", file_path="curly.txt", old_string='他说"你好"',
          new_string='她说"再见"')
    check("直引号匹配弯引号", "已编辑" in r, r)
    check("弯引号还原", curly.read_text(encoding="utf-8") == "她说“再见”\n",
          repr(curly.read_text(encoding="utf-8")))

    # ---------- new_string 行尾空白 ----------
    print("[行尾空白]")
    ws = SANDBOX / "ws.py"
    ws.write_text("x = 1\n", encoding="utf-8")
    d(reg, "Read", file_path="ws.py")
    d(reg, "Edit", file_path="ws.py", old_string="x = 1", new_string="x = 2   ")
    check(".py 剥行尾空白", ws.read_text(encoding="utf-8") == "x = 2\n",
          repr(ws.read_text(encoding="utf-8")))
    md = SANDBOX / "note.md"
    md.write_text("old\n", encoding="utf-8")
    d(reg, "Read", file_path="note.md")
    d(reg, "Edit", file_path="note.md", old_string="old", new_string="硬换行  ")
    check(".md 保留行尾两空格（硬换行）", md.read_text(encoding="utf-8") == "硬换行  \n",
          repr(md.read_text(encoding="utf-8")))

    # ---------- staleness ----------
    print("[staleness]")
    stale = SANDBOX / "stale.py"
    stale.write_text("v = 1\n", encoding="utf-8")
    d(reg, "Read", file_path="stale.py")
    time.sleep(0.02)
    stale.write_text("EXTERNAL CHANGE\n", encoding="utf-8")  # 外部改内容
    r = d(reg, "Edit", file_path="stale.py", old_string="v = 1", new_string="v = 2")
    check("外部修改后 Edit 拒绝", r.startswith("ERROR") and "被修改" in r, r)

    fake = SANDBOX / "fake.py"
    fake.write_text("same\n", encoding="utf-8")
    d(reg, "Read", file_path="fake.py")
    time.sleep(0.02)
    fake.write_text("same\n", encoding="utf-8")  # 同内容重写 → mtime 假阳性
    r = d(reg, "Edit", file_path="fake.py", old_string="same", new_string="changed")
    check("mtime 假阳性内容兜底放行", "已编辑" in r, r)

    # ---------- Glob ----------
    print("[Glob]")
    (SANDBOX / "pkg").mkdir(exist_ok=True)
    for i in range(120):
        (SANDBOX / "pkg" / f"m{i:03}.py").write_text(f"# {i}\n", encoding="utf-8")
    time.sleep(0.02)
    (SANDBOX / "pkg" / "newest.py").write_text("# newest\n", encoding="utf-8")
    r = d(reg, "Glob", pattern="**/*.py", path="pkg")
    lines = [ln for ln in r.split("\n") if ln.strip()]
    path_lines = [ln for ln in lines if not ln.startswith("（")]
    check("Glob 截断到 100 条", len(path_lines) == 100 and "只显示最新 100 个" in r,
          f"paths={len(path_lines)}")
    check("Glob 按 mtime 排序（最新在前）", path_lines[0] == "pkg/newest.py",
          path_lines[0])
    r = d(reg, "Glob", pattern="**/*.nomatch")
    check("Glob 无匹配", r.startswith("No files found"), r)
    r = d(reg, "Glob", pattern="*.py", path="no_such_dir")
    check("Glob 目录不存在报错", r.startswith("ERROR"), r)

    # ---------- Grep ----------
    print("[Grep]")
    r = d(reg, "Grep", pattern="def greet")
    check("fwm 默认模式", r.strip() == "hello.py", r)
    r = d(reg, "Grep", pattern="def greet", output_mode="content")
    check("content 模式带行号", "hello.py:1:def greet():" in r, r)
    r = d(reg, "Grep", pattern="def greet", output_mode="count")
    check("count 模式", "hello.py:1" in r, r)
    r = d(reg, "Grep", pattern="DEF GREET", **{"-i": True})
    check("-i 忽略大小写", "hello.py" in r, r)
    r = d(reg, "Grep", pattern="greet", **{"-A": 1}, output_mode="content")
    check("-A 上下文行（path-lineno- 形态）", "hello.py-2-    return 'yo'" in r, r)
    r = d(reg, "Grep", pattern="^line 1", path="big.py", output_mode="content",
          head_limit=3)
    check("head_limit 截断提示", "head_limit=3" in r and "offset" in r, r)
    match_lines = [ln for ln in r.split("\n")
                   if ln.strip() and not ln.startswith("[")]
    check("head_limit 条数正确", len(match_lines) == 3, match_lines)
    r = d(reg, "Grep", pattern="zzz_no_such_thing")
    check("无匹配 No matches found", r.startswith("No matches found"), r)
    r = d(reg, "Grep", pattern="def greet", glob="*.md")
    check("glob 过滤无匹配", r.startswith("No matches found"), r)
    r = d(reg, "Grep", pattern="def greet", glob="*.py,*.txt")
    check("多 glob 模式", "hello.py" in r, r)
    r = d(reg, "Grep", pattern="return.*\n", multiline=True, output_mode="count")
    check("multiline 不炸", not r.startswith("ERROR"), r)

    # ---------- 端到端闭环 ----------
    print("[端到端]")
    flow = make_registry()  # 新 registry = 新会话
    d(flow, "Write", file_path="app.py", content="def main():\n    pass\n")
    d(flow, "Read", file_path="app.py")
    d(flow, "Edit", file_path="app.py", old_string="    pass",
       new_string="    print('ok')")
    r = d(flow, "Read", file_path="app.py")
    check("写→读→改→读 闭环", "print('ok')" in r, r)

    print("=" * 60)
    print(f"PASS {PASSED}  FAIL {FAILED}")
    if FAILURES:
        print("失败用例:", ", ".join(FAILURES))
    return FAILED == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
