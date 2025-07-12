# 🌌 **LLMgine**

LLMgine is a _pattern-driven_ framework for building **production-grade, tool-augmented LLM applications** in Python.  
It offers a clean separation between _**engines**_ (conversation logic), _**models/providers**_ (LLM back-ends), _**tools**_ (function calling), a streaming **message-bus** for commands & events, and opt-in **observability**.  
Think _FastAPI_ for web servers or _Celery_ for tasks—LLMgine plays the same role for complex, chat-oriented AI.

---

## 🚀 **Enhanced Features for University Adoption**

This repository extends the base llmgine framework with comprehensive enhancements designed for university club deployment and enterprise-grade applications. The implementation transforms the basic LLM framework into a sophisticated AI platform capable of supporting diverse organizational needs while maintaining ease of use for non-technical users.

### **Advanced Architecture Implementation**

The enhanced version implements three core design patterns that significantly improve system reliability and maintainability. The Singleton pattern ensures consistent configuration management across all system components, eliminating configuration drift and providing centralized performance metrics tracking. The Observer pattern enables real-time monitoring of all tool executions, providing automated performance optimization and security threat detection. The Mediator pattern facilitates sophisticated inter-component communication while maintaining loose coupling between system modules.

### **Creative AI Tool Suite**

4 sophisticated AI tools have been integrated to provide immediate practical value for university clubs and research organizations. The Advanced Code Analysis Engine performs comprehensive static analysis with complexity metrics calculation, providing actionable optimization recommendations and anti-pattern detection. The Intelligent Research Platform synthesizes information from multiple sources with built-in fact-checking and bias detection capabilities. The System Performance Profiling tool offers comprehensive resource analysis with bottleneck identification and optimization pathways. The Database Intelligence System provides schema analysis with security vulnerability detection and performance optimization suggestions.

### **Production-Ready Infrastructure**

The enhanced implementation includes enterprise-grade infrastructure components essential for reliable deployment in university environments. Error resilience mechanisms include exponential backoff retry logic and comprehensive exception handling with specific error types. Performance optimization systems feature intelligent caching with automatic invalidation, request deduplication, and dynamic resource scaling recommendations. Security frameworks provide real-time threat assessment, data access auditing, and GDPR-compliant data handling suitable for university compliance requirements.

### **University-Specific Adaptations**

The architecture has been specifically adapted to support multi-club deployment scenarios common in university environments. Modular component systems allow selective feature deployment based on individual club needs. Resource sharing mechanisms enable inter-club collaboration while maintaining appropriate access controls. The scalable architecture supports concurrent usage across multiple organizations without performance degradation.

---

## ✨ Feature Highlights
| Area | What you get | Key files |
|------|--------------|-----------|
| **Engines** | Plug-n-play `Engine` subclasses (`SinglePassEngine`, `ToolChatEngine`, …) with session isolation, tool-loop orchestration, and CLI front-ends | `engines/*.py`, `src/llmgine/llm/engine/` |
| **Enhanced Engines** | Advanced Chat Engine with design patterns, creative AI tools, and production-ready features | `programs/engine/advanced_chat_engine.py` |
| **Design Patterns** | Singleton, Observer, and Mediator pattern implementations for enterprise architecture | `programs/engine/advanced_design_patterns.py` |
| **Creative AI Tools** | Code analysis, research platform, performance profiling, and database intelligence | `programs/engine/creative_ai_tools.py` |
| **Message Bus** | Async **command bus** (1 handler) + **event bus** (N listeners) + **sessions** for scoped handlers | `src/llmgine/bus/` |
| **Tooling** | Declarative function-to-tool registration, multi-provider JSON-schema parsing (OpenAI, Claude, DeepSeek), async execution pipeline | `src/llmgine/llm/tools/` |
| **Providers / Models** | Wrapper classes for OpenAI, OpenRouter, Gemini 2.5 Flash etc. _without locking you in_ | `src/llmgine/llm/providers/`, `src/llmgine/llm/models/` |
| **Context Management** | Simple and in-memory chat history managers, event-emitting for retrieval/update | `src/llmgine/llm/context/` |
| **UI** | Rich-powered interactive CLI (`EngineCLI`) with live spinners, confirmation prompts, tool result panes | `src/llmgine/ui/cli/` |
| **Observability** | Console + JSONL file handlers, per-event metadata, easy custom sinks | `src/llmgine/observability/` |
| **Bootstrap** | One-liner `ApplicationBootstrap` that wires logging, bus startup, and observability | `src/llmgine/bootstrap.py` |

---

## 🏗️ High-Level Architecture

```mermaid
flowchart TD
    %% Nodes
    AppBootstrap["ApplicationBootstrap"]
    Bus["MessageBus<br/>(async loop)"]
    Obs["Observability<br/>Handlers"]
    Eng["Engine(s)"]
    TM["ToolManager"]
    Tools["Your&nbsp;Tools"]
    Session["BusSession"]
    CLI["CLI / UI"]

    %% Edges
    AppBootstrap -->|starts| Bus

    Bus -->|events| Obs
    Bus -->|commands| Eng
    Bus -->|events| Session

    Eng -- status --> Bus
    Eng -- tool_calls --> TM

    TM -- executes --> Tools
    Tools -- ToolResult --> CLI

    Session --> CLI
```

*Every component communicates _only_ through the bus, so engines, tools, and UIs remain fully decoupled.*

---

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/your-org/llmgine.git
cd llmgine
python -m venv .venv && source .venv/bin/activate
pip install -e ".[openai]"   # extras: openai, openrouter, dev, …
export OPENAI_API_KEY="sk-…" # or OPENROUTER_API_KEY / GEMINI_API_KEY
```

### 2. Run the demo CLI

```bash
python -m llmgine.engines.single_pass_engine  # pirate translator
# or
python -m llmgine.engines.tool_chat_engine    # automatic tool loop
# or
python programs/engine/advanced_chat_engine.py  # enhanced engine with creative tools
```

You'll get an interactive prompt with live status updates and tool execution logs.

### 3. Try the Enhanced Features

```bash
# Run comprehensive demonstration
python programs/engine/comprehensive_demo.py

# Test advanced design patterns
python programs/engine/advanced_design_patterns.py

# Explore database integration
python programs/engine/explore_database.py
```

---

## 🧠 **Enhanced Engine Capabilities**

### **Advanced Chat Engine**
The `AdvancedChatEngine` extends the base engine with sophisticated features designed for production deployment:

```python
from advanced_chat_engine import AdvancedChatEngine
from llmgine.llm import SessionID

# Initialize enhanced engine
engine = AdvancedChatEngine(SessionID("university-club-session"))

# Access creative AI tools
await engine.handle_command(AdvancedChatEngineCommand(
    prompt="Analyze this Python code for optimization opportunities"
))
```

### **Creative AI Tools**
Four specialized tools provide immediate practical value:

- **Code Analysis**: Complexity metrics, optimization suggestions, anti-pattern detection
- **Research Platform**: Multi-source synthesis, fact-checking, bias detection  
- **Performance Profiling**: Resource analysis, bottleneck identification, optimization paths
- **Database Intelligence**: Schema analysis, security assessment, performance optimization

### **Design Pattern Integration**
Enterprise-grade architecture patterns ensure reliability and maintainability:

- **Singleton Pattern**: Centralized configuration management
- **Observer Pattern**: Real-time monitoring and performance tracking
- **Mediator Pattern**: Decoupled component communication

---

## 🧑‍💻 Building Your Own Engine

```python
from llmgine.llm.engine.engine import Engine
from llmgine.messages.commands import Command, CommandResult
from llmgine.bus.bus import MessageBus

class MyCommand(Command):
    prompt: str = ""

class MyEngine(Engine):
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.bus = MessageBus()

    async def handle_command(self, cmd: MyCommand) -> CommandResult:
        await self.bus.publish(Status("thinking", session_id=self.session_id))
        # call LLM or custom logic here …
        answer = f"Echo: {cmd.prompt}"
        await self.bus.publish(Status("finished", session_id=self.session_id))
        return CommandResult(success=True, result=answer)

# Wire into CLI
from llmgine.ui.cli.cli import EngineCLI
chat = EngineCLI(session_id="demo")
chat.register_engine(MyEngine("demo"))
chat.register_engine_command(MyCommand, MyEngine("demo").handle_command)
await chat.main()
```

---

## 🔧 Registering Tools in 3 Lines

```python
from llmgine.llm.tools.tool import Parameter
from llmgine.engines.tool_chat_engine import ToolChatEngine

def get_weather(city: str):
    """Return current temperature for a city.
    Args:
        city: Name of the city
    """
    return f"{city}: 17 °C"

engine = ToolChatEngine(session_id="demo")
await engine.register_tool(get_weather)               # ← introspection magic ✨
```

The engine now follows the **OpenAI function-calling loop**:

```
User → Engine → LLM (asks to call get_weather) → ToolManager → get_weather()
          ↑                                        ↓
          └───────────    context update   ────────┘ (loops until no tool calls)
```

---

## 📰 Message Bus in Depth

```python
from llmgine.bus.bus import MessageBus
from llmgine.bus.session import BusSession

bus = MessageBus()
await bus.start()

class Ping(Command): pass
class Pong(Event): msg: str = "pong!"

async def ping_handler(cmd: Ping):
    await bus.publish(Pong(session_id=cmd.session_id))
    return CommandResult(success=True)

with bus.create_session() as sess:
    sess.register_command_handler(Ping, ping_handler)
    sess.register_event_handler(Pong, lambda e: print(e.msg))
    await sess.execute_with_session(Ping())      # prints "pong!"
```

*Handlers are **auto-unregistered** when the `BusSession` exits—no leaks.*

---

## 📊 Observability

Add structured logs with zero boilerplate:

```python
from llmgine.bootstrap import ApplicationBootstrap, ApplicationConfig
config = ApplicationConfig(enable_console_handler=True,
                           enable_file_handler=True,
                           log_level="debug")
await ApplicationBootstrap(config).bootstrap()
```

*All events/commands flow through `ConsoleEventHandler` and `FileEventHandler`
to a timestamped `logs/events_*.jsonl` file.*

---

## 📁 Repository Layout (abridged)

```
llmgine/
│
├─ engines/            # Turn-key example engines (single-pass, tool chat, …)
├─ programs/engine/    # Enhanced engines with advanced features
│   ├─ advanced_chat_engine.py         # Main enhanced engine
│   ├─ advanced_design_patterns.py     # Design pattern implementations
│   ├─ creative_ai_tools.py            # Creative AI tool suite
│   ├─ comprehensive_demo.py           # Full feature demonstration
│   └─ explore_database.py             # Database integration tools
└─ src/llmgine/
   ├─ bus/             # Message bus core + sessions
   ├─ llm/
   │   ├─ context/     # Chat history & context events
   │   ├─ engine/      # Engine base + dummy
   │   ├─ models/      # Provider-agnostic model wrappers
   │   ├─ providers/   # OpenAI, OpenRouter, Gemini, Dummy, …
   │   └─ tools/       # ToolManager, parser, register, types
   ├─ observability/   # Console & file handlers, log events
   └─ ui/cli/          # Rich-based CLI components
```

---

## 🎯 **University Club Applications**

### **Business & Economics Clubs**
- Budget analysis and optimization recommendations
- Market research with automated fact-checking
- Financial performance analytics and reporting

### **Engineering & Technology Societies**  
- Code review automation for hackathon projects
- System performance optimization for club websites
- Database schema analysis for membership systems

### **Research & Academic Groups**
- Literature review assistance with source verification
- Data analysis and statistical pattern identification
- Grant opportunity research and proposal optimization

### **Social & Cultural Organizations**
- Event planning optimization based on attendance patterns
- Social media strategy analysis and improvement suggestions
- Resource allocation optimization for activities and equipment

---

## 🏁 Roadmap

- [ ] **Streaming responses** with incremental event dispatch  
- [ ] **WebSocket / FastAPI** front-end (drop-in replacement for CLI)  
- [ ] **Persistent vector memory** layer behind `ContextManager`  
- [ ] **Plugin system** for third-party Observability handlers  
- [ ] **More providers**: Anthropic, Vertex AI, etc.
- [ ] **University integration APIs** for student portal connectivity
- [ ] **Multi-club collaboration features** for resource sharing
- [ ] **Mobile application interface** for enhanced accessibility

---

## 🤝 Contributing

1. Fork & create a feature branch  
2. Ensure `pre-commit` passes (`ruff`, `black`, `isort`, `pytest`)  
3. Open a PR with context + screenshots/GIFs if UI-related  

---

## 📄 License

LLMgine is distributed under the **MIT License**—see [`LICENSE`](LICENSE) for details.

---

## 🎓 **DS Cubed Integration**

This enhanced implementation supports DS Cubed's vision of expanding LLM technology adoption across University of Melbourne clubs. The architecture provides immediate practical value while maintaining the flexibility to adapt to diverse organizational needs. For deployment guidance and university-specific configuration, see the documentation in `programs/engine/README.md`.

> _"Build architecturally sound LLM apps, not spaghetti code.  
> Welcome to the engine room."_
