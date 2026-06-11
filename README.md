<div align="center">
  <br>
  <h1>🤖 LangChain AI 聊天助手</h1>
  <p><strong>带对话记忆 · 工具调用 · 多轮对话</strong></p>
  <br>
</div>

## ✨ 功能

| 功能 | 说明 |
|------|------|
| 🧠 对话记忆 | 基于 LangChain `ConversationBufferMemory`，记住上下文 |
| 🔧 工具调用 | LangChain `@tool` 装饰器 + `create_tool_calling_agent` |
| 🌤️ 查天气 | 查询任意城市的实时天气 |
| 🧮 数学计算 | 表达式求值，支持 + - × ÷ ** sqrt 等 |
| 🕐 时区时间 | 获取任意时区的当前日期和时间 |
| 🌐 IP 查询 | 查询 IP 地址的地理位置和 ISP 信息 |

## 🛠️ 技术栈

```
AI 框架       ─  LangChain 0.3+ (Agent + Tool Calling)
AI 模型       ─  DeepSeek API
前端/后端     ─  Streamlit
记忆机制      ─  ConversationBufferMemory
```

## 🚀 本地运行

```bash
# 1. 克隆
git clone https://github.com/LIUCHUHUAN7051/ai-chat-assistant.git
cd ai-chat-assistant

# 2. 装依赖
pip install -r requirements.txt

# 3. 配置 API Key
cp .env.example .env
# 在 .env 中填入 DEEPSEEK_API_KEY=你的key

# 4. 启动
streamlit run app.py
```

## 🧠 LangChain 特性展示

本项目重点展示以下 LangChain 核心能力：

| 组件 | 用途 |
|------|------|
| `ChatOpenAI` | 通过 OpenAI 兼容接口接入 DeepSeek |
| `create_tool_calling_agent` | 创建支持自动工具调用的 Agent |
| `AgentExecutor` | Agent 执行器，含容错和迭代限制 |
| `ConversationBufferMemory` | 对话历史记忆 |
| `ChatPromptTemplate` | 结构化提示词模板 |
| `@tool` | 装饰器模式定义工具 |
| `MessagesPlaceholder` | 动态消息占位符 |

## 📂 项目结构

```
ai-chat-assistant/
├── app.py              # 主程序（Agent + UI）
├── requirements.txt    # 依赖清单
├── .env.example        # API Key 模板
├── .gitignore          # Git 忽略规则
└── README.md           # 项目说明
```

## 👤 关于

- 作者：刘理鑫
- 求职方向：AI 应用开发工程师
- 邮箱：CHUIZI705179074@outlook.com
