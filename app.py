"""LangChain AI 聊天助手 — 基于 LangChain 1.x 的智能对话应用"""

import json
import os
import urllib.request
import urllib.parse
import datetime
import zoneinfo

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

# ==================== 工具定义 ====================


@tool
def get_weather(city: str) -> str:
    """查询任意城市的实时天气"""
    try:
        url = f"https://wttr.in/{urllib.parse.quote(city)}?format=%C+%t+%h+%w&lang=zh"
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode().strip()
            return f"🌤️ {city} 天气：{raw}"
    except Exception as e:
        return f"查询天气失败：{e}"


@tool
def calculate(expression: str) -> str:
    """计算数学表达式，支持 + - × ÷ ** sqrt 等运算"""
    allowed = {"abs", "int", "float", "str", "len", "range", "list",
               "sum", "min", "max", "round", "pow", "sqrt"}
    try:
        expr = expression.replace("×", "*").replace("÷", "/")\
                         .replace("（", "(").replace("）", ")")
        code = compile(expr, "<string>", "eval")
        for name in code.co_names:
            if name not in allowed and not name.startswith("_"):
                return f"不支持的运算：{name}"
        import math
        builtins = {n: getattr(math, n, None) for n in allowed}
        result = eval(expr, {"__builtins__": {}}, builtins)
        return f"🧮 计算结果：{result}"
    except Exception as e:
        return f"计算失败：{e}"


@tool
def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """获取指定时区的当前日期和时间"""
    try:
        tz = zoneinfo.ZoneInfo(timezone)
        now = datetime.datetime.now(tz)
        weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        wd = weekdays[now.weekday()]
        return (
            f"🕐 {timezone} 当前时间："
            f"{now.strftime('%Y年%m月%d日')} {wd} {now.strftime('%H:%M:%S')}"
        )
    except Exception as e:
        return f"获取时间失败：{e}"


@tool
def get_ip_info(ip: str = "") -> str:
    """查询 IP 地址的地理位置和 ISP 信息"""
    try:
        url = (f"http://ip-api.com/json/{ip}?lang=zh-CN" if ip
               else "http://ip-api.com/json/?lang=zh-CN")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            if data.get("status") == "success":
                return (
                    f"🌐 IP 信息：{data.get('query', ip)}\n"
                    f"📍 {data.get('country', '')} {data.get('regionName', '')} "
                    f"{data.get('city', '')}\n"
                    f"🏢 {data.get('isp', '未知')}"
                )
            return f"查询失败：{data.get('message', '未知错误')}"
    except Exception as e:
        return f"查询 IP 失败：{e}"


tools = [get_weather, calculate, get_current_time, get_ip_info]


# ==================== 初始化 ====================

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY") or st.secrets.get("DEEPSEEK_API_KEY", "")

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key,
    base_url="https://api.deepseek.com/v1",
    temperature=0.7,
)

checkpointer = MemorySaver()

# ==================== Streamlit UI ====================

st.set_page_config(page_title="LangChain AI 助手", page_icon="🤖", layout="centered")

# ---------- 样式 ----------
st.markdown("""
<style>
    @keyframes fadeIn { from { opacity:0; transform:translateY(-10px); } to { opacity:1; transform:translateY(0); } }
    @keyframes slideUp { from { opacity:0; transform:translateY(15px); } to { opacity:1; transform:translateY(0); } }
    @keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)} }
    @keyframes gradientShift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }

    .stApp { background: #f5f7fb; }
    .block-container { padding-top: 1.5rem !important; animation: fadeIn 0.5s ease-out; }
    .header {
        background: linear-gradient(135deg, #0891b2, #2563eb, #7c3aed);
        background-size: 200% 200%; animation: gradientShift 6s ease infinite;
        color: white; padding: 1.5rem 2rem; border-radius: 16px;
        margin-bottom: 1.2rem; text-align: center;
    }
    .header h1 { margin:0; font-size:1.8rem; font-weight:700; }
    .header p { margin:0.3rem 0 0; opacity:0.85; font-size:0.9rem; }
    .header .badge {
        display:inline-block; background:rgba(255,255,255,0.2); border-radius:6px;
        padding:2px 10px; font-size:0.7rem; margin-top:8px;
    }
    .tool-tag {
        display:inline-block; background:#e8edf5; color:#1e3a5f;
        font-size:0.75rem; padding:3px 12px; border-radius:12px;
        margin:2px 3px; border:1px solid #d1d9e8; transition:all 0.2s;
    }
    .tool-tag:hover { background:#d1d9e8; transform:translateY(-1px); }
    div[data-testid="stChatMessage"] { animation: slideUp 0.3s ease-out; border-radius:12px; }
    div[data-testid="stChatInput"] input {
        border-radius:24px !important; border:1px solid #e5e7eb !important;
        padding:0.6rem 1.2rem !important; transition:all 0.25s !important;
    }
    div[data-testid="stChatInput"] input:focus {
        border-color:#2563eb !important; box-shadow:0 0 0 3px rgba(37,99,235,0.1) !important;
    }
    .empty-state { text-align:center; padding:4rem 1rem; color:#9ca3af; animation:fadeIn 0.8s; }
    .empty-state .icon { font-size:4rem; animation:float 3s ease-in-out infinite; }
    section[data-testid="stSidebar"] > div:first-child { background:#fff; border-right:1px solid #e5e7eb; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <h1>🤖 LangChain AI 助手</h1>
    <p>带对话记忆 · 工具调用 · 多轮对话</p>
    <span class="badge">⚡ LangChain 1.x + LangGraph</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; margin-bottom:0.8rem;">
    <span class="tool-tag">🌤️ 查天气</span>
    <span class="tool-tag">🧮 计算器</span>
    <span class="tool-tag">🕐 时区时间</span>
    <span class="tool-tag">🌐 IP 查询</span>
</div>
""", unsafe_allow_html=True)

# ---------- 状态 ----------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "chat-1"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "你是一个智能 AI 助手。你可以使用工具来获取实时信息"
            "（天气、时间、IP 等）或进行计算。若用户闲聊，直接回答即可。"
            "回答简洁自然，使用中文。"
        ),
        checkpointer=checkpointer,
    )

# ---------- 聊天区域 ----------
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">🤖</div>
        <p>你好！我是你的 AI 助手</p>
        <p style="font-size:0.85rem; margin-top:8px;">
            试试问我天气、算数、查时间或 IP 地址
        </p>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt_input := st.chat_input("试试说「北京天气」「计算 1024×768」「现在几点」..."):
    st.chat_message("user").markdown(prompt_input)
    st.session_state.messages.append({"role": "user", "content": prompt_input})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("⏳ 思考中...")

        try:
            result = st.session_state.agent.invoke(
                {"messages": [HumanMessage(content=prompt_input)]},
                config={"configurable": {"thread_id": st.session_state.thread_id}},
            )
            ai_msg = result["messages"][-1]
            reply = ai_msg.content if isinstance(ai_msg, AIMessage) else str(ai_msg)
            placeholder.markdown(reply)
            st.session_state.messages.append(
                {"role": "assistant", "content": reply}
            )
        except Exception as e:
            error_msg = f"😅 出错了：{e}"
            placeholder.markdown(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )

# ---------- 侧边栏 ----------
with st.sidebar:
    st.markdown("### 🧠 LangChain 1.x 特性展示")
    st.markdown("""
    <div style="background:#f8fafc; border-radius:8px; padding:0.8rem; font-size:0.8rem; line-height:1.6;">
        ✅ <strong>create_agent</strong> — LangGraph 版 Agent<br>
        ✅ <strong>MemorySaver</strong> — 对话记忆检查点<br>
        ✅ <strong>@tool 装饰器</strong> — 自定义工具<br>
        ✅ <strong>LangGraph 图执行</strong> — 自动工具循环<br>
        ✅ <strong>thread_id</strong> — 多会话隔离<br>
        ✅ <strong>DeepSeek + OpenAI 兼容接口</strong>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("##### 💡 试试这样说")
    st.caption("北京天气怎么样？")
    st.caption("1024×768 等于多少？")
    st.caption("东京现在几点了？")
    st.caption("查一下 8.8.8.8 的位置")
    st.caption("sqrt(144) + 3² 等于多少？")

    if st.session_state.messages:
        st.divider()
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.thread_id = "chat-2"
            st.session_state.messages = []
            st.session_state.agent = create_agent(
                model=llm,
                tools=tools,
                system_prompt=(
                    "你是一个智能 AI 助手。你可以使用工具来获取实时信息"
                    "（天气、时间、IP 等）或进行计算。若用户闲聊，直接回答即可。"
                    "回答简洁自然，使用中文。"
                ),
                checkpointer=MemorySaver(),
            )
            st.rerun()
