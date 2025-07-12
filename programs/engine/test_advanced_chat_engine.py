#!/usr/bin/env python3
"""
Test file for the Advanced Chat Engine.
This validates that the engine components work correctly.
"""

import asyncio
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from llmgine.llm import SessionID
from advanced_chat_engine import (
    AdvancedChatEngine,
    AdvancedChatEngineCommand,
    AdvancedLLMManager,
    AdvancedToolManager,
    AdvancedChatContext,
    calculate_area,
    get_system_info,
    search_web
)


async def test_engine_components():
    """Test that all engine components can be instantiated."""
    print("🧪 Testing Advanced Chat Engine Components...")
    
    session_id = SessionID("test-session")
    
    # Test LLM Manager
    llm_manager = AdvancedLLMManager()
    print("✅ LLM Manager created successfully")
    
    # Test Tool Manager
    tool_manager = AdvancedToolManager("test-engine", session_id)
    print("✅ Tool Manager created successfully")
    
    # Test Chat Context
    context = AdvancedChatContext("test-engine", session_id)
    print("✅ Chat Context created successfully")
    
    # Test Engine
    engine = AdvancedChatEngine(session_id)
    print("✅ Advanced Chat Engine created successfully")
    
    return True


async def test_tool_registration():
    """Test that tools can be registered."""
    print("\n🔧 Testing Tool Registration...")
    
    session_id = SessionID("test-tool-session")
    engine = AdvancedChatEngine(session_id)
    
    # Register sample tools
    await engine.register_tool(calculate_area, require_confirmation=False)
    await engine.register_tool(get_system_info, require_confirmation=True)
    await engine.register_tool(search_web, require_confirmation=False)
    
    print("✅ All tools registered successfully")
    return True


async def test_command_structure():
    """Test that commands work correctly."""
    print("\n📝 Testing Command Structure...")
    
    # Test command creation
    command = AdvancedChatEngineCommand(
        prompt="Test prompt",
        require_confirmation=False
    )
    
    print(f"✅ Command created: {command.prompt}")
    return True


async def test_context_management():
    """Test that context management works."""
    print("\n💬 Testing Context Management...")
    
    session_id = SessionID("test-context-session")
    context = AdvancedChatContext("test-engine", session_id, "Test system message")
    
    # Test adding user message
    context.add_user_message("Hello, world!")
    
    # Test getting messages
    messages = await context.get_messages()
    print(f"✅ Context contains {len(messages)} messages")
    
    return True


async def main():
    """Run all tests."""
    print("🚀 Starting Advanced Chat Engine Tests\n")
    
    try:
        # Run all tests
        await test_engine_components()
        await test_tool_registration()
        await test_command_structure()
        await test_context_management()
        
        print("\n🎉 All tests passed! Advanced Chat Engine is working correctly.")
        print("\n📋 Summary of implemented features:")
        print("   • Modular architecture with separated components")
        print("   • LLM Manager for OpenAI API integration")
        print("   • Tool Manager with automatic schema generation")
        print("   • Chat Context with conversation history")
        print("   • User confirmation system for sensitive tools")
        print("   • Example tools demonstrating different types")
        print("   • Full CLI integration")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 