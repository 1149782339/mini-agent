# mini-agent

一个最小但能真跑的 Hermes 风格 agent：**工具调用闭环 + 上下文自动压缩**。
核心代码约 3100 行，仅依赖 openai / python-dotenv，uv 管理（`uv run mini-agent`）。

## 快速开始

```bash
cd mini-agent
uv sync                                   # 首次：建 .venv 并装依赖

# 配置（可选：留空会自动复用 ~/.hermes/.env 里的 ARK_API_KEY）
cp .env.example .env

# 启动 REPL（新建会话，会打印会话 ID）
uv run mini-agent

# 会话管理
uv run mini-agent --list              # 列出所有会话
uv run mini-agent --resume <ID>       # 按 ID 恢复（latest = 最近一个）
uv run mini-agent --continue          # 继续最近一次会话
```

## 用法

```
你 › 帮我在 demo/a.py 里写个快排，然后读一遍确认
  → Write {"file_path": "demo/a.py", ...}
    ← 已新建 ...
  → Read {"file_path": "demo/a.py"}
    ←      1\tdef quick_sort...
已完成 ...

你 › /tokens
1284 tokens（窗口 65536）
```

命令：`/help` `/clear` `/compact` `/tokens` `/log` `/quit`

## 工具

框架移植自 Claude-Code（`Tool.ts` 的四段生命周期）：每个工具严格分为
`validate_input`（纯校验、零 I/O）→ `check_permissions`（默认 allow，扩展点）
→ `call`（结构化输出）→ `render`（序列化给模型）；默认值一律 fail-closed。
错误信息即自愈指引——每个失败都告诉模型下一步该做什么，并带 errorCode 编号。

文件五件套同样移植自 Claude-Code，共享一个读状态缓存（readFileState 机制）。
**读写不对称：Read/Glob/Grep 允许访问机器上任意路径**（相对路径基于工作目录
解析，绝对路径任意位置放行）；**Edit/Write 仍限定在工作目录内**，防止误写系统。
UNC 路径（`\\server\share`）一律拒绝且不做任何文件系统操作——Windows 的
stat 会触发 SMB 连接造成 NTLM 凭据泄漏（Claude-Code 同款细节）。

- **Read**：任意路径可读。cat -n 行号视图；offset/limit 分页；整读 256KB 字节帽（超限报错
  提示分段，报错比截断省 token）；同范围重复读回 `file_unchanged` stub；
  双路径读取（小文件全文、大文件流式只留范围内行）；BOM 自动识别。
- **Edit**：精确替换。read-before-write（必须先 Read）+ mtime staleness 闸
  （mtime 假阳性有内容比对兜底）；old_string 不唯一报错并提示加上下文或
  `replace_all`；弯引号容错（直引号能匹配弯引号，写回还原风格）；CRLF/UTF-16
  编码行尾写回保真；`.md` 行尾两空格豁免。
- **Write**：新建或整文件覆盖。覆盖已存在文件前必须先 Read；内容原样写入。
- **Glob**：任意目录可搜。`rg --files` 按文件名找文件，修改时间最新在前，截断 100 条。
- **Grep**：ripgrep 全功能映射（content/files_with_matches/count、-i/-A/-B/-C、
  glob/type/multiline、head_limit+offset 翻页），超时与无匹配严格区分。

另有 `run_command`（shell 执行，见下）。语义差异一处：Claude-Code 用权限层
deny 规则做文件排除，mini-agent 无权限层，Glob/Grep 默认用 rg 原生语义
（跳过隐藏文件、尊重 .gitignore）；设 `MINI_AGENT_RG_ALL=1` 可反转。

## 结构

```
src/mini_agent/
├── config.py       # 环境变量 / .env / ARK_API_KEY 回退
├── tokens.py       # 无依赖 token 估算（ASCII 3.5 字符/token，中文 1 字/token）
├── client.py       # OpenAI 兼容客户端；丢弃 reasoning_content；usage 锚点
├── compaction.py   # 压缩纯函数：切点 / 摘要序列化 / 文件清单 / 模板（移植自 Pi）
├── context.py      # 历史管理 + 压缩（token 预算切分 / 结构化摘要 / 质量闸 / 兜底）
├── store.py        # SQLite 会话持久化（sessions + messages 两表）
├── loop.py         # 主循环：请求 → tool_calls → 回填 → 再请求
├── cli.py          # REPL
└── tools/
    ├── base.py        # 工具框架：四段生命周期 + 注册表 + 输出落盘截断
    ├── filestate.py   # 读状态 LRU 缓存（read-before-write 的核心状态机）
    ├── read.py        # Read
    ├── edit.py        # Edit（弯引号容错 / 编码行尾保真）
    ├── write.py       # Write
    ├── glob_tool.py   # Glob（rg --files）
    ├── grep_tool.py   # Grep（rg 全功能映射）
    ├── rg.py          # ripgrep 子进程封装（超时/退出码语义）
    └── shell.py       # run_command：继承环境变量、超时杀进程组、cwd 受限
```

## 压缩怎么做的（移植自 Pi 的上下文管理）

1. **记账**：token 估算优先用 provider 真实 usage 做锚点——最后一条 assistant
   消息的 `_usage.total`（流式 `include_usage` 带回）≈ 当时上下文规模，之后的
   消息按启发式增量估算；锚点之后发生过压缩则锚点作废、全量重估。
2. **触发**：请求估算（消息 + 工具 schema）超过 `context_window × compact_ratio`。
3. **切分**：token 预算制（`keep_recent_tokens`，默认 20000，小窗口自适应收缩）——
   从最新往回累加估算到预算即停；切点**绝不切断 toolCall/toolResult 配对**
   （宁可多保留往前挪）。
4. **摘要**：头部先序列化成 `[User]/[Assistant]/[Tool result]` 标签文本
   （防模型续写对话；工具结果限 2000 字符、思维链丢弃），用固定 6 节结构
   （Goal / Constraints / Progress / Key Decisions / Next Steps /
   Critical Context）；有旧摘要时走"更新合并"模式而非重写。
5. **文件清单**：`<read-files>/<modified-files>` 从工具调用参数**机械提取**
   （不靠 LLM），确定性附加在摘要尾，跨次压缩随摘要文本累积。
6. **质量闸**：摘要为空 / finish_reason=length（被截断）/ 像续写或报错 →
   拒收，走机械裁剪兜底——坏摘要永久污染会话比不摘要更糟。
7. **校验**：压缩后检查 system 在首位、每条 tool 都能找到 tool_call、总量不超
   窗口；失败回滚 + 机械裁剪。

源头截断（Pi 的"三层收缩"第一层）：Read 单次输出 50000 字符封顶（不在行中间
切断，提示 offset 继续）、单行超 2000 字符截断；run_command 超 40KB 保留**末尾**
（错误在尾部）并把完整输出落盘 `.mini_agent/spill/`、结果里带路径引用。
未移植：split-turn 双摘要、分支摘要、append-only 事件树投影、溢出自动恢复。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `MINI_AGENT_API_KEY` | 空 → 回退 `ARK_API_KEY` | 密钥 |
| `MINI_AGENT_BASE_URL` | `https://ark.cn-beijing.volces.com/api/plan/v3` | OpenAI 兼容端点 |
| `MINI_AGENT_MODEL` | `deepseek-v4.1-flash-economy` | 模型名 |
| `MINI_AGENT_CONTEXT_WINDOW` | 1000000 | 窗口 token（V4.1-Flash 原生 1M 输入） |
| `MINI_AGENT_COMPACT_RATIO` | 0.7 | 压缩触发比例（即约 70 万 token 时触发） |
| `MINI_AGENT_KEEP_RECENT_TOKENS` | 20000 | 压缩时尾部保留的 token 预算（Pi keepRecentTokens，小窗口自适应收缩） |
| `MINI_AGENT_WORKDIR` | `.` | 文件工具可达范围 |
| `MINI_AGENT_ENABLE_SHELL` | `1` | 设为 `0` 关闭 run_command 工具 |
| `MINI_AGENT_RG_ALL` | 未设 | 设 `1` 后 Glob/Grep 搜索隐藏文件且忽略 .gitignore |
| `MINI_AGENT_SHOW_REASONING` | `1` | 是否在终端实时打印思维链 |
| `MINI_AGENT_DB` | `<项目根>/.mini_agent/sessions.db` | 会话库路径 |

数据默认落在**项目内**：`<项目根>/.mini_agent/`（`sessions.db` + `spill/`），
不污染家目录，项目挪到哪数据就跟到哪。该目录不可写时（只读检出、受限环境），
退回 `~/.mini-agent/sessions.db` 并打一行提示。启动脚本的当前目录不影响落点。

## 编码约定：全链路 UTF-8

- 文件读写、SQLite、`.env`、工具返回值一律 UTF-8，不再按系统码页猜；
- 子进程输出统一按 UTF-8 解码（不抛异常，非法字节替换成 `�`）；
- `run_command` 强制给子进程注入 `PYTHONUTF8=1` / `PYTHONIOENCODING=utf-8`，
  Python 子进程不会因为本地 locale 是 cp936 而吐出 GBK；
- 终端输出在启动时把 stdout/stderr 切到 UTF-8，管道里也不会再被劈成乱码。

副作用（刻意取舍）：Windows 原生工具（cmd 内建命令、部分本地化程序）输出 GBK 时
会显示成 `�`。这类场景优先用 `git` / `node` / `python` / `rg` 这些天然 UTF-8 的工具。

## 思维链回传（must-echo）

DeepSeek/Kimi/MiniMax 这类 provider 要求 assistant 历史必须带 `reasoning_content`，
否则可能 400。实现与 Hermes 一致：

- 思维链**存进 SQLite**（`messages.payload` 里的 `reasoning_content`），resume 时一起恢复；
- 发给 API 前，每条 assistant 消息都带该字段：有真实文本传文本，缺失补**一个空格** `" "`
  （DeepSeek V4 拒绝空字符串）；
- 压缩摘要后的历史同样补 pad，不会因为 reasoning 被清掉而报错。

## 流式输出

所有模型调用走 `stream=True`：思维链灰色逐字输出，正文随后逐字输出，
工具调用分片按 `index` 聚合后再派发。首字通常 1–2 秒到达，不会再有
"整轮卡住几十秒后突然出结果"的感觉。

实测：首字 1.33s（reasoning），整段完成 3.29s。

## 测试

```bash
uv run python examples/tool_test.py       # 离线，60 项：五件套 + 框架行为
uv run python examples/context_test.py    # 离线，33 项：压缩管线（切点/摘要/锚点记账，假 LLM）
uv run python examples/smoke_test.py      # 需要 API key
uv run python examples/timeout_check.py   # 离线，验证超时清理
```

`context_test` 不需要模型（LLM 用假实现），覆盖：切点绝不拆 toolCall/toolResult
配对、摘要输入序列化（思维链丢弃/2000 字符截断）、文件清单机械提取、6 节模板
与更新模式、坏摘要拒收、机械裁剪兜底、usage 锚点记账与压缩后作废、小窗口预算
自适应。

`tool_test` 不需要模型，覆盖：read-before-write、staleness 闸（含 mtime
假阳性内容兜底）、CRLF/UTF-16/BOM 写回保真、弯引号容错、file_unchanged
去重、分页、二进制/UNC/越界拒绝、Grep 三模式与翻页、Glob 排序截断、
字符串布尔容错。

`smoke_test` 覆盖两项：
- 工具闭环（Write → Edit → Read，真实模型驱动）；
- 压缩（构造超长历史，验证摘要、配对完整、压缩后继续对话）。

`timeout_check` 不需要模型：让子进程睡 5 秒后写标记文件、超时设 1 秒，
断言超时被识别且进程树真的被杀（标记文件不会出现）。

## 安全边界

- **读写不对称**：Edit/Write 的路径 `resolve()` 后必须落在 `workdir` 内，
  `../` 越界直接拒绝；Read/Glob/Grep 只读，允许访问任意路径，不做范围限制。
- **read-before-write**：Edit/Write 修改已存在文件前，该会话必须 Read 过它，
  且磁盘未被外部修改（mtime staleness 闸，全文读有内容比对兜底误报）。
- UNC 路径（`\\server\share`）一律拒绝且不做任何文件系统操作——Windows 的
  stat 会触发 SMB 连接造成 NTLM 凭据泄漏（Claude-Code 同款细节）。
- `/dev/zero` 等设备文件黑名单，防读取挂死。
- Edit 按原编码/行尾写回（CRLF 文件不会被写成 LF）；Write 内容原样写入。
- `run_command` 的 `cwd` 同样受 `workdir` 约束；命令本身不设白名单（要能跑测试/编译/git）。
- `run_command` 继承当前进程完整环境变量，子进程能看到 `PATH`、`HOME`、API key——这是刻意设计，便于复现用户的 shell 行为。
- 超时默认 60s（上限 300s），到点杀整棵进程树，不留孤儿：
  POSIX 对进程组发 `SIGKILL`，Windows 用 Job Object（`taskkill` 只在
  Job 创建失败时兜底，因为受限环境里它可能被拒绝）。
- `stdin` 关闭，交互式命令（vim/top）会直接退出而不是挂死主循环。
- 工具异常统一转成 `ERROR: ...` 字符串返回给模型，不会让主循环崩溃。

## 已知限制

- token 估算是启发式（±15%），不是 tiktoken 精确值。
- 压缩摘要会丢失细节，只保关键信息；极长会话需要多轮压缩。
- 会话已持久化（`.mini_agent/sessions.db`），但压缩后的原文不保留，只留摘要。
- 未实现并行工具执行、审批机制（框架的 `check_permissions` 阶段是预留扩展点）。
- Read 只支持文本（无图片/PDF/ipynb 多模态）；没有 list_dir 工具，
  看目录用 `run_command` 的 ls。
- Glob/Grep 的排除规则来自 rg 原生语义（隐藏文件/.gitignore），
  没有 Claude-Code 那种按路径的权限规则体系。
