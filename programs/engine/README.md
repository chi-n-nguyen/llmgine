# Advanced Chat Engine

This engine demonstrates the **Week 2 LLM Application Architecture** concepts with modular design patterns from basic scripts to scalable production systems.

## Project 2.1 Implementation

**Repository**: [https://github.com/chi-n-nguyen/llmgine](https://github.com/chi-n-nguyen/llmgine)  
**Location**: `programs/engine/advanced_chat_engine.py` (as specified in Project 2.1)

## Architecture Overview

The Advanced Chat Engine implements a **modular architecture** with clear separation of concerns:

### Core Components

1. **AdvancedLLMManager**: Handles OpenAI API interactions
2. **AdvancedToolManager**: Manages tool registration with confirmation system
3. **AdvancedChatContext**: Manages conversation history and context
4. **AdvancedChatEngine**: Main orchestrator that coordinates all components

### Key Features

- **Automatic Schema Generation**: Uses llmgine's built-in tool registration
- **User Confirmation System**: Sensitive tools require confirmation before execution
- **Enhanced Error Handling**: Specific exception types for better debugging
- **Event-Driven Architecture**: Full integration with llmgine's message bus
- **Modular Design**: Each component can be used independently

## Usage

### 1. Interactive CLI

```bash
# Activate virtual environment
source /path/to/.venv/bin/activate

# Run the engine
cd programs/engine
python advanced_chat_engine.py
```

**Important CLI Note**: When typing in the engine terminal, use `Alt + Enter` or `Esc + Enter` to submit your input. Regular `Enter` just creates a new line.

### 2. Programmatic Usage

```python
from advanced_chat_engine import AdvancedChatEngine
from llmgine.llm import SessionID

# Create engine
engine = AdvancedChatEngine(session_id=SessionID("my-session"))

# Register custom tools
await engine.register_tool(my_function, require_confirmation=True)

# Process commands
command = AdvancedChatEngineCommand(prompt="Calculate the area of a 5x3 rectangle")
result = await engine.handle_command(command)
print(result.result)
```

## Example Tools

The engine includes three demonstration tools:

1. **`calculate_area(length, width)`**: Simple math calculation (no confirmation)
2. **`get_system_info()`**: System information retrieval (requires confirmation)
3. **`search_web(query)`**: Simulated web search (no confirmation)

## Testing

Run the comprehensive test suite:

```bash
python test_advanced_chat_engine.py
```

Tests cover:
- Component instantiation
- Tool registration and confirmation settings
- Command structure validation
- Context management functionality

## Database Exploration

The engine includes database exploration capabilities:

```bash
python explore_database.py
```

This script connects to the PostgreSQL database and explores the **Medallion Architecture**:
- **Bronze Layer**: Raw data ingestion
- **Silver Layer**: Cleaned and transformed data
- **Gold Layer**: Business analytics and aggregated data

## Week 2 Concepts Demonstrated

### 1. From Basic Scripts to Scalable Architecture
- **Separated Components**: LLM Manager, Tool Manager, Chat Context
- **Improved Reusability**: Each component works independently
- **Better Extensibility**: Easy to add new tools or modify behavior
- **Enhanced Error Handling**: Specific exception types and proper logging

### 2. Design Patterns Applied
- **Dependency Injection**: Components are injected into the main engine
- **Strategy Pattern**: Different managers can be swapped
- **Observer Pattern**: Event-driven architecture via message bus
- **Factory Pattern**: Tool registration and schema generation

### 3. Production-Ready Features
- **Automatic Schema Generation**: No manual JSON schema creation
- **User Confirmation System**: Safety for sensitive operations
- **Comprehensive Testing**: Full test coverage
- **Code Quality**: Type hints, docstrings, error handling

## Code Quality Improvements

Recent optimizations include:
- **Removed redundant schema generation** (let base manager handle it)
- **Improved error handling** with specific exception types
- **Efficient context management** (update vs recreate)
- **Cleaned unused imports** and code
- **Enhanced tool execution** with proper error propagation

## Files

- `advanced_chat_engine.py`: Main engine implementation
- `test_advanced_chat_engine.py`: Comprehensive test suite
- `explore_database.py`: Database exploration script
- `README.md`: This documentation

## Next Steps

This implementation is ready for **Project 2.2: Brain Architecture** where we'll create PostgreSQL tables and implement the database layer for the AI memory system.

---

**Built with**: Python 3.11+, OpenAI API, llmgine framework  
**License**: MIT 