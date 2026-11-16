# mini-agent

一个能真跑的编码代理：工具调用闭环、上下文自动压缩、SQLite 会话。
核心代码在 `src/mini_agent/`，约 5100 行。依赖 `openai`、`pillow`、`python-dotenv`、`rich`，
需要 Python ≥ 3.14，用 uv 管理（`uv run mini-agent`）。

## 快速开始

在项目根目录：

```bash
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

单次非交互（子代理也是这个形态）：

```bash
uv run mini-agent -p "任务" --json --tools "Read,Write"
```

`--json` 时 stdout 每行一个 JSON 事件（`thinking` / `message_end` / `result` / `error`）。
`--no-session` 不写会话库。`--model`、`--max-steps`、`--append-system-prompt FILE`
覆盖当次运行。`--tools` 是逗号分隔的白名单；print 模式永不注册 Task。

## 用法

```
你 › 帮我在 demo/a.py 里写个快排，然后读一遍确认
  → Write {"file_path": "demo/a.py", ...}
    ← 已新建 ...
  → Read {"file_path": "demo/a.py"}
    ←      1	def quick_sort...
已完成 ...

你 › /tokens
1284 tokens（窗口 1000000）
```

REPL 命令：`/help` `/clear` `/compact` `/tokens` `/log` `/session` `/quit`
（`/exit`、`/q` 等同于退出）。`/clear` 和 `/compact` 会同步写回会话库。
退出时打印恢复命令：`mini-agent --resume <会话 ID>`。

## 工具

每个工具分为
`validate_input`（纯校验、零 I/O）→ `check_permissions`（默认 allow，扩展点）
→ `call`（结构化输出）→ `render`（序列化给模型）。`is_read_only` /
`is_concurrency_safe` 默认 false。失败信息即自愈指引，并带 errorCode。

文件工具共享一个读状态缓存。读写不对称：**Read / Ls / Glob / Grep 可以读机器上任意路径**
（相对路径基于工作目录解析）；**Edit / Write 必须落在工作目录内**。
UNC 路径（`\\server\share`）在任何文件系统调用之前拒绝——Windows 对 UNC 做
stat 会触发 SMB 连接，造成 NTLM 凭据泄漏。

- **Read**：cat -n 行号视图。默认最多 2000 行，`offset` / `limit` 分页。
  整读有 256KB 字节帽，超限报错并提示分段（报错比截断省 token）。
  单次正文再封顶 50000 字符，单行超过 2000 字符截断。同范围重复读返回
  `file_unchanged`。小文件全文读，大文件流式只留范围内行。自动识别 BOM。
  也读图片：按文件头识别 jpg、png、gif、webp、bmp，不看扩展名。能直接放进请求的
  原样附上，bmp 转成 png；长边超过 2000 或 base64 超过约 4.5MB 时缩小。图片不写入
  读状态缓存，不能用来满足 Edit 的先读要求。
- **Edit**：精确替换。修改前必须先 Read。mtime 变了先当过期；全文读过的文件
  再用内容比对排除 mtime 假阳性。`old_string` 不唯一时报错，提示加上下文或
  `replace_all`。直引号可以匹配弯引号，写回时还原原风格。按原编码和行尾写回
  （CRLF / UTF-16 不会被改成 LF）。`.md` 行尾的两个空格不参与匹配。
- **Write**：新建或整文件覆盖。覆盖已存在的文件前必须先 Read。内容原样写入。
- **Ls**：列一层目录。目录名以 `/` 结尾，目录在前。默认跳过以 `.` 开头的项，`all=true` 可显示。最多 200 项。
- **Glob**：`rg --files` 按文件名找文件，修改时间新的在前，最多 100 条。
- **Grep**：ripgrep 的 content / files_with_matches / count，以及
  `-i` / `-A` / `-B` / `-C`、glob、type、multiline、`head_limit` + `offset`。
  超时和无匹配是两种结果。
- **Task**：子代理委派。子代理是独立进程（`python -m mini_agent -p`），
  不继承父会话。交接物只有子代理 system prompt 和父模型写好的任务文本。
  子代理自己读写文件、运行命令，返回最后一条 assistant 文本（50000 字符截断）。
  默认步数 150，超时 900 秒（参数可调，上限 3600）。print 模式不注册 Task，
  不能递归派生。同一轮里多个 Task 会并行执行（最多 4 个线程），结果按
  `tool_calls` 原顺序回填。执行期间终端底部用 rich 面板显示每个子代理的
  步数、token、一行摘要；长思考期间有「思考中 N 字」心跳。
- **run_command**：见下方安全边界。Windows 上优先 Git Bash（`bash.exe -lc`，
  找不到则退回 cmd），POSIX 上是 `sh -c`。按 bash 语法写命令。

没有权限层。Glob / Grep 使用 rg 默认语义
（跳过隐藏文件、尊重 `.gitignore`）。`MINI_AGENT_RG_ALL=1` 改为搜索隐藏文件并忽略 gitignore。

同一轮按 `tool_calls` 的顺序执行。连续的 `is_concurrency_safe()` 工具
（Read / Ls / Glob / Grep / Task）并行，最多 4 个线程。Edit、Write、`run_command`
是屏障，在主线程按原顺序执行，所以「先改文件再跑测试」不会把测试跑到修改前面。
同一路径的写入还有一把互斥锁。`finish_reason=length` 时工具参数视为被截断，
不执行，回一条合成错误让模型把输出拆小。Ctrl+C 会给还没回填的 `tool_call`
补上「已中断」，避免历史里留下没有结果的工具调用。

工具返回值还有一层统一截断：超过 80000 字符时保留前半和末尾四分之一，
全文落到 `.mini_agent/spill/`。这个上限高于 Read 的 50000 和 `run_command` 的
40000，正常读取不会被悄悄掐成头尾两段。`run_command` 仍先把超过 40000 字符的
输出收成末尾并单独落盘（错误通常在尾部）。

## 目录

```
src/mini_agent/
├── config.py       # 环境变量 / .env / ARK_API_KEY 回退
├── tokens.py       # 无依赖 token 估算（ASCII 约 3.5 字符/token，中文 1 字/token）
├── client.py       # OpenAI 兼容客户端；流式；usage 锚点；重试
├── compaction.py   # 压缩纯函数：切点 / 摘要序列化 / 文件清单 / 模板
├── context.py      # 历史管理 + 压缩（token 预算切分 / 结构化摘要 / 质量闸 / 兜底）
├── store.py        # SQLite 会话持久化（sessions + messages 两表）
├── loop.py         # 主循环：请求 → tool_calls → 回填 → 再请求
├── cli.py          # REPL、print 模式、子代理进度面板
└── tools/
    ├── base.py        # 四段生命周期 + 注册表 + 输出落盘截断
    ├── filestate.py   # 读状态 LRU 缓存（read-before-write 的状态机）
    ├── read.py        # Read（文本 + 图片）
    ├── image.py       # 图片魔数、缩放、bmp 转 png
    ├── edit.py        # Edit
    ├── write.py       # Write
    ├── glob_tool.py   # Glob（rg --files）
    ├── grep_tool.py   # Grep
    ├── ls.py          # Ls
    ├── rg.py          # ripgrep 子进程封装
    ├── shell.py       # run_command
    └── subagent.py    # Task
```

## 上下文压缩

在每次请求前检查。

1. **记账**。优先用最后一条 assistant 的 `_usage.total`（流式 `include_usage`）
   当作当时的上下文规模，之后的消息按启发式增量加。锚点之后如果发生过压缩，
   锚点作废，改为全量重估。没有锚点时，启发式大约 ±15%。
2. **触发**。消息加工具 schema 的估算超过 `context_window × compact_ratio`
   （默认 100 万 × 0.7）。
3. **切分**。从最新消息往回累加，凑满 `keep_recent_tokens`（默认 20000；
   小窗口会收缩，且不超过压缩阈值的一半）即停。切点不拆开 tool call 和 tool result。
4. **摘要**。被切掉的头部先变成 `[User]` / `[Assistant]` / `[Tool result]` 标签文本
   （思维链丢弃，单条工具结果限 2000 字符），再让模型按六节写：
   Goal、Constraints、Progress、Key Decisions、Next Steps、Critical Context。
   已有旧摘要时走更新合并。
5. **文件清单**。`<read-files>` / `<modified-files>` 从工具参数机械提取，
   附在摘要尾部，跨次压缩随摘要文本累积。
6. **质量闸**。摘要为空、`finish_reason=length`，或看起来像续写 / 报错，就拒收，
   改走机械裁剪。坏摘要会留在会话里，比不摘要更糟。
7. **校验**。压缩后检查 system 在首位、每条 tool 消息能对上 tool_call、
   总量不超过窗口。失败则只回滚这一次尝试，再机械裁一次。
   单条消息超过窗口四分之一时，先把其中的思维链收成空格并截断正文，
   避免一条超长消息让裁剪无从下手。

provider 返回上下文过长时，主循环强制压缩并重试这一轮恰好一次。

还没有 split-turn 双摘要、分支摘要、append-only 事件树。
压缩后原文不保留，只留摘要。极长会话会多次压缩，细节会丢。

## 配置

优先级：环境变量 > 项目 `.env` > `~/.hermes/.env` 里的 `ARK_API_KEY`。
模板见 `.env.example`。

| 变量 | 默认 | 说明 |
|---|---|---|
| `MINI_AGENT_API_KEY` | 空，回退 `ARK_API_KEY` | 密钥 |
| `MINI_AGENT_BASE_URL` | `https://ark.cn-beijing.volces.com/api/plan/v3` | OpenAI 兼容端点 |
| `MINI_AGENT_MODEL` | `deepseek-v4.1-flash-economy` | 模型名 |
| `MINI_AGENT_CONTEXT_WINDOW` | `1000000` | 窗口 token |
| `MINI_AGENT_COMPACT_RATIO` | `0.7` | 压缩触发比例 |
| `MINI_AGENT_MAX_OUTPUT_TOKENS` | `32768` | 单次输出预算（思维链、正文、工具参数共享）。太小会被 `length` 截断 |
| `MINI_AGENT_MAX_STEPS` | `40` | 单轮最大工具步数。子代理进程内被设为 150 |
| `MINI_AGENT_KEEP_RECENT_TOKENS` | `20000` | 压缩时尾部保留的 token |
| `MINI_AGENT_WORKDIR` | `.` | Edit/Write 与 `run_command` 的 cwd 限制在此目录。Read/Glob/Grep 可读任意路径 |
| `MINI_AGENT_ENABLE_SHELL` | `1` | `0` / `false` / `no` / `off` 关闭 `run_command` |
| `MINI_AGENT_ENABLE_SUBAGENT` | `1` | 同上，关闭 Task。print 模式始终不注册 Task |
| `MINI_AGENT_SHELL` | `auto` | `auto` / `gitbash` / `cmd` |
| `MINI_AGENT_BASH` | 空 | `bash.exe` 路径。也认 `GIT_BASH`。其次从 PATH 上的 `git.exe` 推导，再扫常见安装位置 |
| `MINI_AGENT_RG_ALL` | 未设 | `1` / `true` / `yes` 时 Glob/Grep 搜索隐藏文件并忽略 `.gitignore` |
| `MINI_AGENT_SHOW_REASONING` | `1` | `0` 关闭终端上的思维链 |
| `MINI_AGENT_VISION` | 按模型名 | `1` 强制附图；`0` 强制不附。未设置时，DeepSeek-V4.1-Flash（含 `deepseek-v4.1-flash-economy`）以及名字里含 `vl` / `vision` / `gpt-4o` / `gpt-4.1` / `gpt-5` / `gemini` / `claude` / `pixtral` / `grok-4` 的模型附图。`deepseek-v4-pro`、`deepseek-chat` 不附 |
| `MINI_AGENT_DB` | `<项目根>/.mini_agent/sessions.db` | 会话库路径。设置后不再做可写回退 |
| `MINI_AGENT_DEBUG` | 未设 | `1` 或 `true` 时，子代理往 stderr 打计时 |

数据默认在项目内 `<项目根>/.mini_agent/`（`sessions.db` 和 `spill/`）。
该目录不可写时退回 `~/.mini-agent/sessions.db` 并提示一行。
启动时的当前目录不影响落点：项目根按 `pyproject.toml` 向上找。

单条工具结果的 80000 字符上限写在 `Config.max_tool_output`，没有对应环境变量。

## 编码：全链路 UTF-8

文件读写、SQLite、`.env`、工具返回值一律 UTF-8。子进程输出按 UTF-8 解码，
非法字节换成 `�`，不按系统码页猜。`run_command` 给子进程注入
`PYTHONUTF8=1` 和 `PYTHONIOENCODING=utf-8`。启动时把 stdout / stderr 切到 UTF-8。

Windows 上输出 GBK 的程序（cmd 内建命令、部分本地化工具）会显示成 `�`。
优先用 `git`、`node`、`python`、`rg`。

## 思维链回传

DeepSeek、Kimi、MiniMax 要求历史里的 assistant 消息带 `reasoning_content`，
否则可能 400。思维链存在 SQLite 的 `messages.payload` 里，resume 时一起恢复。
发给 API 前，有真实文本就传文本，缺失则补一个空格（DeepSeek V4 拒绝空字符串）。
只有最后一条 assistant 保留完整思维链，更早的收成一个空格。全文仍在 SQLite 里。
压缩后的历史同样补这一位。模型名里不含这些关键字时不补，也不改已有思维链。

## 流式输出

模型调用都走 `stream=True`。思维链灰色逐字输出，正文接着逐字输出，
工具参数按 `index` 拼好再派发。大段工具参数生成期间，同一行刷新
「正在生成 … 的参数」。流式 `include_usage` 不被端点接受时，去掉该参数重试一次。

网络抖动、429、5xx、流中途断开，这一层最多再试 3 次（SDK 自己还有 `max_retries=2`），
并尊重 `Retry-After`。认证错误和参数类 400 直接失败。

## 测试

```bash
uv run python examples/tool_test.py       # 离线：文件五件套 + 框架行为
uv run python examples/image_test.py      # 离线：Read 图片、缩放、请求里附图或省略
uv run python examples/context_test.py    # 离线：压缩管线（假 LLM）
uv run python examples/retry_test.py      # 离线：重试分类、退避、溢出恢复（假 API）
uv run python examples/parallel_test.py   # 离线：并行派发、按序回填、截断不派发
uv run python examples/subagent_test.py   # 离线：print 模式 argv、JSON 事件、截断、注册开关
uv run python examples/smoke_test.py      # 需要 API key：工具闭环 + 真实压缩
uv run python examples/timeout_check.py   # 离线：超时后进程树被杀掉
uv run python examples/shell_test.py      # 手工脚本：继承环境、退出码、超时、cwd、管道
```

`context_test` 覆盖切点不拆配对、摘要序列化、文件清单、六节模板与更新模式、
坏摘要拒收、机械裁剪、usage 锚点作废、小窗口预算收缩。

`tool_test` 覆盖 read-before-write、staleness 闸、CRLF / UTF-16 / BOM、
弯引号、`file_unchanged`、分页、二进制 / UNC / 越界拒绝、Grep 三模式、
Glob 排序截断、字符串布尔容错。

`timeout_check` 让子进程睡 5 秒后写标记文件，超时设 1 秒，断言超时被识别且标记文件不会出现。

## 安全边界

- Edit / Write 的路径 `resolve()` 之后必须在 `workdir` 内，`../` 越界直接拒绝。
  Read / Glob / Grep 只读，不设范围。
- Edit / Write 修改已存在文件前，本会话必须 Read 过它，且磁盘未被外部改过。
- UNC 路径不做任何文件系统操作。设备文件（如 `/dev/zero`）在黑名单里。
- `run_command` 的 cwd 受 `workdir` 约束。命令本身没有白名单。
- `run_command` 继承当前进程的完整环境，子进程能看到 `PATH`、`HOME` 和 API key。
- 超时默认 60 秒，上限 300 秒。到点杀掉整棵进程树：POSIX 对进程组发 `SIGKILL`，
  Windows 用 Job Object（Job 创建失败时才用 `taskkill`）。
- stdin 关闭。vim / top 这类交互命令会自己退出。
- 工具异常变成 `ERROR: ...` 返回给模型，不让主循环崩溃。

## 已知限制

- 审批没有实现。`check_permissions` 只是扩展点。
- Task 只有 single 模式：没有 parallel / chain，没有 `agents/*.md` 自定义定义。
  父终端看得到步骤级进度。Ctrl+C 会杀掉子进程。
- Read 能读 jpg、png、gif、webp、bmp。PDF 和 notebook 仍不读。列目录用 Ls，它只列一层。
  默认的 DeepSeek-V4.1-Flash（含经济版）会看图：这一轮工具结果后面多一条
  user 消息，内容是 `image_url` 的 data URL。纯文本模型（如 `deepseek-v4-pro`）
  只留文字，并换成一句不支持图片的说明。`MINI_AGENT_VISION=0` 强制不附，`1` 强制附。
- Glob / Grep 的排除规则来自 rg（隐藏文件、`.gitignore`），没有按路径的权限规则。
- API 重试覆盖网络抖动、限流、5xx 和流中途断开。上下文溢出自动压缩后重试一次。
