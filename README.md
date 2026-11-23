# mini-agent

一个能真跑的编码代理：工具调用闭环、上下文自动压缩、SQLite 会话。
核心代码在 `src/mini_agent/`，约 6000 行。依赖 `openai`、`pillow`、`rich`，
需要 Python ≥ 3.13，用 uv 管理（`uv run mini-agent`）。

## 快速开始

在项目根目录：

```bash
uv sync                                   # 首次：建 .venv 并装依赖

# 配置（复制模板到用户目录并填写 api_key）
mkdir -p ~/.config/mini-agent
cp config.toml.example ~/.config/mini-agent/config.toml

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
（`/exit`、`/q` 等同于退出）。`/compact` 把摘要写入会话库，原文仍保留；`/clear` 只重置当前视图，原文同样保留。
退出时打印恢复命令：`mini-agent --resume <会话 ID>`。

## 工具

每个工具分为
`validate_input`（纯校验、零 I/O）→ `check_permissions`（默认 allow，扩展点）
→ `call`（结构化输出）→ `render`（序列化给模型）。`is_read_only` /
`is_concurrency_safe` 默认 false。失败信息即自愈指引，并带 errorCode。

文件工具共享一个读状态缓存。读写不对称：**Read / Ls / Glob / Grep 可以读机器上任意路径**
（相对路径基于工作目录解析）；**Edit / Write / RM / MV 必须落在工作目录内**。RM / MV 还要求路径是字面绝对路径。
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
- **RM**：删除工作目录内的文件或目录。路径必须是字面绝对路径，相对路径和 `~/xxx` 一律拒绝，不会先展开再判断。解析软链接后仍须落在工作目录内，也禁止删除工作目录本身。默认不递归，删除目录必须显式 `recursive=true`。
- **MV**：移动或重命名工作目录内的文件或目录。`source` 和 `destination` 都必须是字面绝对路径，且 resolve 后仍在工作目录内，禁止移动工作目录本身。`destination` 必须是最终路径；若已存在则拒绝覆盖。
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
  长时间任务可设 `background=true`，立刻返回 `task_id`，不受 300 秒硬杀限制；
  再加 `notify_on_complete=true` 时，任务结束后把退出码和输出汇报给用户并写入会话，然后停下，不自动开始下一步。
  前台命令触达超时时，交互模式会升级为后台并打开完成通知，而不是杀掉。
- **process_manage**：管理后台命令，动作是 `list` / `poll` / `log` / `wait` / `kill`。
  同一任务连续两次 poll 或 wait 且没有新输出时，自动改为完成通知。轮询对防空转护栏豁免提示和硬停。

没有权限层。Glob / Grep 使用 rg 默认语义
（跳过隐藏文件、尊重 `.gitignore`）。`MINI_AGENT_RG_ALL=1` 改为搜索隐藏文件并忽略 gitignore。

同一轮按 `tool_calls` 的顺序执行。连续的 `is_concurrency_safe()` 工具
（Read / Ls / Glob / Grep / Task）并行，最多 4 个线程。Edit、Write、RM、MV、`run_command`、
`process_manage` 是屏障，在主线程按原顺序执行，所以「先改文件再跑测试」不会把测试跑到修改前面。
同一路径的写入还有一把互斥锁。`finish_reason=length` 时工具参数视为被截断，
不执行，回一条合成错误让模型把输出拆小。Ctrl+C 会给还没回填的 `tool_call`
补上「已中断」，避免历史里留下没有结果的工具调用。

主循环里还有一层防空转护栏（`guardrails.py`，思路来自 Hermes，只做瘦身版）。
同一轮用户输入里，连续用相同参数打出相同结果：第 2 次且结果超过 512 字符时，
工具仍执行，但写入上下文的改成短引用；第 3 次给模型追加提示；另外会抓周期
2 到 4 的循环（例如 Read A、Grep B 交替）。`process_manage` 视为轮询类工具，豁免提示和硬停，短引用照做。同一工具同一参数失败，第 2 次警告；
`-p` / 脚本 / Task 子代理默认开硬停，第 5 次在执行前拦截，第 5 次相同成功调用
或第 8 次「同一工具换参数失败」会结束本轮（`run_command` 对后一种不硬停，
因为测试和编译失败是正常诊断）。交互式 CLI 默认只警告、不硬停。`Edit` /
`Write` / `run_command` / `Task` / `RM` / `MV` 成功会清空失败计数，避免
「改代码再跑测试」被误杀。这层与 `max_steps`、溢出压缩重试并存，不写入会话库。

工具返回值还有一层统一截断：超过 80000 字符时保留前半和末尾四分之一，
全文落到 `.mini_agent/spill/`。这个上限高于 Read 的 50000 和 `run_command` 的
40000，正常读取不会被悄悄掐成头尾两段。`run_command` 仍先把超过 40000 字符的
输出收成末尾并单独落盘（错误通常在尾部）。

## 目录

```
src/mini_agent/
├── config.py       # 用户 config.toml 加载，环境变量覆盖
├── tokens.py       # 无依赖 token 估算（ASCII 约 3.5 字符/token，中文 1 字/token）
├── client.py       # OpenAI 兼容客户端；流式；usage 锚点；重试
├── compaction.py   # 压缩纯函数：切点 / 摘要序列化 / 文件清单 / 模板
├── context.py      # 历史管理 + 压缩（token 预算切分 / 结构化摘要 / 质量闸 / 兜底）
├── store.py        # SQLite 会话持久化（sessions + messages + session_events）
├── envutil.py      # 子进程环境清理（去掉 API key 等密钥）
├── guardrails.py   # 防空转：相同调用盯梢、周期循环、失败拦截
├── jobs.py         # 后台命令注册表：完成通知、查询、退出清理
├── loop.py         # 主循环：请求 → tool_calls → 回填 → 再请求
├── cli.py          # REPL（输入线程 + 完成通知汇报）、print 模式、子代理进度面板
└── tools/
    ├── base.py        # 四段生命周期 + 注册表 + 输出落盘截断
    ├── filestate.py   # 读状态 LRU 缓存（read-before-write 的状态机）
    ├── read.py        # Read（文本 + 图片）
    ├── image.py       # 图片魔数、缩放、bmp 转 png
    ├── edit.py        # Edit
    ├── write.py       # Write
    ├── rm_mv.py       # RM / MV
    ├── glob_tool.py   # Glob（rg --files）
    ├── grep_tool.py   # Grep
    ├── ls.py          # Ls
    ├── rg.py          # ripgrep 子进程封装
    ├── shell.py       # run_command + process_manage
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
   若切点落在一轮中间（某条 user 之后的 assistant / tool 上），不会把这一轮的前半段
   塞进同一份总摘要。
4. **摘要**。切在轮与轮的边界时，被切掉的头部变成 `[User]` / `[Assistant]` /
   `[Tool result]` 标签文本（思维链丢弃，单条工具结果限 2000 字符），再让模型按六节写：
   Goal、Constraints、Progress、Key Decisions、Next Steps、Critical Context。
   已有旧摘要时走更新合并。切在一轮中间时，先写本轮之前的历史摘要，再单独写被切开的
   这一轮前缀摘要（原请求、已有进展、理解后缀还需要的上下文），合并到
   `Turn Context (split turn)` 小节下；这一轮的后缀仍保留原文。任一份摘要被质量闸拒收
   或调用失败，都整段改走机械裁剪，不会只留下一半摘要。
5. **文件清单**。`<read-files>` / `<modified-files>` 从工具参数机械提取，
   附在摘要尾部，跨次压缩随摘要文本累积。
6. **质量闸**。摘要为空、`finish_reason=length`，或看起来像续写 / 报错，就拒收，
   改走机械裁剪。坏摘要会留在会话里，比不摘要更糟。
7. **校验**。压缩后检查 system 在首位、每条 tool 消息能对上 tool_call、
   总量不超过窗口。失败则只回滚这一次尝试，再机械裁一次。
   单条消息超过窗口四分之一时，先把其中的思维链收成空格并截断正文，
   避免一条超长消息让裁剪无从下手。

provider 返回上下文过长时，主循环强制压缩并重试这一轮恰好一次。

还没有分支摘要、分支事件树。
压缩和 `/clear` 都只改发给模型的视图，SQLite 里的原文仍在，resume 时按最近一次压缩/清空重建视图。极长会话会多次压缩，模型侧细节会丢，但磁盘上可以回看原文。

## 配置

默认从 `~/.config/mini-agent/config.toml` 读取，Windows 也使用这个路径。
完整模板见 `config.toml.example`，复制后填写 `api_key` 即可。
优先级：命令行参数（`--config` / `--model` / `--max-steps`）> 环境变量 > 用户 `config.toml` > 默认值。
`.env` 不再参与配置加载。

```toml
# ~/.config/mini-agent/config.toml
api_key = "你的密钥"
model = "deepseek-v4.1-flash-economy"
base_url = "https://ark.cn-beijing.volces.com/api/coding/v3"
context_window = 1000000
show_reasoning = true
```

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `MINI_AGENT_API_KEY` | 空 | 覆盖 TOML 的 `api_key`；通常在 TOML 中配置 |
| `MINI_AGENT_CONFIG` | `~/.config/mini-agent/config.toml` | 配置文件路径，也可用 `--config` 指定 |
| `MINI_AGENT_BASE_URL` | `https://ark.cn-beijing.volces.com/api/coding/v3` | 覆盖 TOML 的 `base_url`，OpenAI 兼容端点 |
| `MINI_AGENT_MODEL` | `deepseek-v4.1-flash-economy` | 覆盖 TOML 的 `model`；`--model` 优先级更高 |
| `MINI_AGENT_CONTEXT_WINDOW` | `1000000` | 窗口 token |
| `MINI_AGENT_COMPACT_RATIO` | `0.7` | 压缩触发比例 |
| `MINI_AGENT_MAX_OUTPUT_TOKENS` | `32768` | 单次输出预算（思维链、正文、工具参数共享）。太小会被 `length` 截断 |
| `MINI_AGENT_MAX_STEPS` | `40` | 单轮最大工具步数。子代理进程内被设为 150 |
| `MINI_AGENT_FOREGROUND_BLOCK_BUDGET` | `15` | 前台命令最多阻塞秒数，到点自动转后台继续跑；`timeout` 仍负责最终终止 |
| `MINI_AGENT_KEEP_RECENT_TOKENS` | `20000` | 压缩时尾部保留的 token |
| `MINI_AGENT_WORKDIR` | `.` | Edit/Write/RM/MV 与 `run_command` 的 cwd 限制在此目录。Read/Glob/Grep 可读任意路径 |
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

文件读写、SQLite、`config.toml`、工具返回值一律 UTF-8。子进程输出按 UTF-8 解码，
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

离线测试已收成 pytest 套件，不需要 API key：

```bash
uv sync --group dev
uv run pytest
```

覆盖文件工具、图片、压缩管线、会话库（压缩后原文仍在、视图是摘要）、
API 重试与溢出恢复、并行派发、子代理 argv/事件、`run_command` 行为，
以及子进程环境会去掉 API key。

需要真实 API key 的冒烟测试不在默认 pytest 里，请单独跑：

```bash
uv run python examples/smoke_test.py      # 工具闭环 + 真实压缩
```

## 安全边界

- Edit / Write 的路径 `resolve()` 之后必须在 `workdir` 内，`../` 越界直接拒绝。
  Read / Glob / Grep 只读，不设范围。
- RM / MV 的路径必须是字面绝对路径（相对路径和 `~/xxx` 直接拒绝，不会先展开再接受）。
  解析软链接后仍须落在 `workdir` 内，禁止操作工作目录本身。
  RM 默认不递归，删除目录必须 `recursive=true`。MV 在目标已存在时拒绝覆盖。
- Edit / Write 修改已存在文件前，本会话必须 Read 过它，且磁盘未被外部改过。
- UNC 路径不做任何文件系统操作。设备文件（如 `/dev/zero`）在黑名单里。
- `run_command` 的 cwd 受 `workdir` 约束。命令没有完整白名单，但会拒绝直接调用
  `rm`、`mv`（包括 `/bin/rm`、`/usr/bin/mv`），请改用 RM / MV。这是安全带，
  拦不住 `python` 删文件、`find -delete`、`unlink`、`git rm` 或把 `rm` 写进脚本；
  用 Write / Edit 写含 `rm` 的清理脚本仍然允许。
- `run_command` 继承 `PATH`、`HOME` 等运行环境，但不传递 `MINI_AGENT_API_KEY`、`ARK_API_KEY` 以及其他常见密钥变量。
  Task 子代理同样先清理这些变量，再由父进程显式传回实际生效的 API key 和模型配置。
- 前台超时默认 60 秒，上限 300 秒。交互模式下到点不杀，升级为后台任务并打开完成通知；
  `-p` / 非交互模式仍杀掉整棵进程树。后台任务不受这 300 秒硬杀限制。
  终止时 POSIX 对进程组发 `SIGKILL`，Windows 用 Job Object（Job 创建失败时才用 `taskkill`）。
  退出交互式 CLI 或 Ctrl+C 离开 REPL 时，会清理仍在跑的后台进程。
- stdin 关闭。vim / top 这类交互命令会自己退出。
- 工具异常变成 `ERROR: ...` 返回给模型，不让主循环崩溃。

## 已知限制

- 后台 Task 没有做：子代理仍然会挡住父轮。下载/编译用后台 `run_command` 即可。
- `-p` 模式关闭完成通知（退出后没有人接手），结束时会清掉还在跑的后台任务。
- 没有 watch_patterns、heartbeat、跨重启恢复，也没有关掉窗口还能继续跑的网关。
- 审批没有实现。`check_permissions` 只是扩展点。
- Task 只有 single 模式：没有 parallel / chain，没有 `agents/*.md` 自定义定义。
  父终端看得到步骤级进度。Ctrl+C 会杀掉子进程。
- Read 能读 jpg、png、gif、webp、bmp。PDF 和 notebook 仍不读。列目录用 Ls，它只列一层。
  默认的 DeepSeek-V4.1-Flash（含经济版）会看图：这一轮工具结果后面多一条
  user 消息，内容是 `image_url` 的 data URL。纯文本模型（如 `deepseek-v4-pro`）
  只留文字，并换成一句不支持图片的说明。`MINI_AGENT_VISION=0` 强制不附，`1` 强制附。
- Glob / Grep 的排除规则来自 rg（隐藏文件、`.gitignore`），没有按路径的权限规则。
- API 重试覆盖网络抖动、限流、5xx 和流中途断开。上下文溢出自动压缩后重试一次。
