import uuid
import json
import asyncio
import inspect
from typing import Any, Dict, List, Optional, get_type_hints
from dataclasses import dataclass

from llmgine.bus.bus import MessageBus
from llmgine.llm.context.memory import SimpleChatHistory
from llmgine.llm.models.openai_models import Gpt41Mini
from llmgine.llm.providers.providers import Providers
from llmgine.llm.tools.tool_manager import ToolManager
from llmgine.llm.tools import ToolCall
from llmgine.llm.models.openai_models import OpenAIResponse
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from llmgine.messages.commands import Command, CommandResult
from llmgine.messages.events import Event
from llmgine.ui.cli.cli import EngineCLI
from llmgine.ui.cli.components import EngineResultComponent
from llmgine.llm import SessionID, AsyncOrSyncToolFunction


@dataclass
class AdvancedChatEngineCommand(Command):
    """Command for the Advanced Chat Engine."""
    prompt: str = ""
    require_confirmation: bool = False


@dataclass
class AdvancedChatEngineStatusEvent(Event):
    """Event emitted when the status of the engine changes."""
    status: str = ""


@dataclass
class AdvancedChatEngineToolResultEvent(Event):
    """Event emitted when a tool is executed."""
    tool_name: str = ""
    result: Any = None
    confirmed: bool = False


@dataclass
class AdvancedChatEngineConfirmationEvent(Event):
    """Event emitted when user confirmation is requested."""
    tool_name: str = ""
    arguments: Dict[str, Any] = None
    description: str = ""


class AdvancedLLMManager:
    """Manages LLM interactions following Week 2 specification."""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model = Gpt41Mini(Providers.OPENAI)
        self.model_name = model_name
    
    async def generate_response(self, messages: List[Dict], tools: Optional[List[Dict]] = None) -> OpenAIResponse:
        """Generate a response from the LLM with optional tools."""
        return await self.model.generate(messages=messages, tools=tools)


class AdvancedToolManager:
    """Enhanced tool manager with automatic schema generation."""
    
    def __init__(self, engine_id: str, session_id: SessionID):
        self.base_manager = ToolManager(
            engine_id=engine_id, 
            session_id=session_id, 
            llm_model_name="openai"
        )
        self.tools = {}
        self.confirmable_tools = set()
        self.tool_schemas = []
    
    async def register_tool(self, func: AsyncOrSyncToolFunction, require_confirmation: bool = False):
        """Register a function as a tool with automatic schema generation."""
        # Register with base manager
        await self.base_manager.register_tool(func)
        
        # Store for our enhanced functionality
        name = func.__name__
        self.tools[name] = func
        
        if require_confirmation:
            self.confirmable_tools.add(name)
        
        # Generate enhanced schema
        schema = self._generate_enhanced_schema(func)
        self.tool_schemas.append(schema)
        
        print(f"Advanced tool registered: {name} (confirmation: {require_confirmation})")
    
    def _generate_enhanced_schema(self, func) -> Dict:
        """Generate enhanced schema with type hints."""
        name = func.__name__
        docstring = func.__doc__ or "No description available"
        sig = inspect.signature(func)
        
        try:
            type_hints = get_type_hints(func)
        except Exception:
            type_hints = {}
        
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param.default is inspect.Parameter.empty:
                required.append(param_name)
            
            param_type = type_hints.get(param_name, Any)
            json_type = self._python_type_to_json_type(param_type)
            
            properties[param_name] = {
                "type": json_type,
                "description": f"The {param_name} parameter"
            }
        
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": docstring.split("\n")[0] if docstring else "No description",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
    
    def _python_type_to_json_type(self, python_type) -> str:
        """Convert Python type to JSON schema type."""
        if python_type == str:
            return "string"
        elif python_type == int:
            return "integer"
        elif python_type == float:
            return "number"
        elif python_type == bool:
            return "boolean"
        elif python_type == list:
            return "array"
        elif python_type == dict:
            return "object"
        else:
            return "string"  # Default fallback
    
    async def get_tools(self) -> List[Dict]:
        """Get all registered tools."""
        return await self.base_manager.get_tools()
    
    async def execute_tool_call(self, tool_call: ToolCall) -> Any:
        """Execute a tool call."""
        return await self.base_manager.execute_tool_call(tool_call)
    
    def needs_confirmation(self, tool_name: str) -> bool:
        """Check if a tool needs confirmation."""
        return tool_name in self.confirmable_tools


class AdvancedChatContext:
    """Enhanced chat context manager following Week 2 specification."""
    
    def __init__(self, engine_id: str, session_id: SessionID, system_message: str = "You are a helpful assistant."):
        self.base_context = SimpleChatHistory(engine_id=engine_id, session_id=session_id)
        self.system_message = system_message
        self.base_context.set_system_prompt(system_message)
    
    def add_user_message(self, content: str):
        """Add a user message to the context."""
        self.base_context.store_string(content, "user")
    
    async def add_assistant_message(self, message: ChatCompletionMessage):
        """Add an assistant message to the context."""
        await self.base_context.store_assistant_message(message)
    
    def add_tool_result(self, tool_call_id: str, name: str, content: str):
        """Add a tool result to the context."""
        self.base_context.store_tool_call_result(tool_call_id, name, content)
    
    async def get_messages(self) -> List[Dict]:
        """Get all messages in the context."""
        return await self.base_context.retrieve()
    
    def clear(self):
        """Clear the context."""
        self.base_context.clear()


class AdvancedChatEngine:
    """Advanced Chat Engine implementing Week 2 specification concepts."""
    
    def __init__(self, session_id: SessionID):
        """Initialize the Advanced Chat Engine with modular components."""
        self.message_bus = MessageBus()
        self.engine_id = str(uuid.uuid4())
        self.session_id = session_id
        
        # Initialize modular components following Week 2 spec
        self.llm_manager = AdvancedLLMManager()
        self.tool_manager = AdvancedToolManager(self.engine_id, self.session_id)
        self.context = AdvancedChatContext(
            self.engine_id, 
            self.session_id, 
            "You are an advanced AI assistant with access to various tools. You can help with tasks and answer questions."
        )
        
        print(f"Advanced Chat Engine initialized with session: {self.session_id}")
    
    async def handle_command(self, command: AdvancedChatEngineCommand) -> CommandResult:
        """Handle a command with advanced processing."""
        try:
            await self.message_bus.publish(
                AdvancedChatEngineStatusEvent(
                    status="processing request", 
                    session_id=self.session_id
                )
            )
            
            # Add user message to context
            self.context.add_user_message(command.prompt)
            
            # Main processing loop
            while True:
                # Get current context and available tools
                current_context = await self.context.get_messages()
                tools = await self.tool_manager.get_tools()
                
                # Call LLM
                await self.message_bus.publish(
                    AdvancedChatEngineStatusEvent(
                        status="calling LLM", 
                        session_id=self.session_id
                    )
                )
                
                response = await self.llm_manager.generate_response(
                    messages=current_context, 
                    tools=tools
                )
                
                # Extract response message
                response_message = response.raw.choices[0].message
                await self.context.add_assistant_message(response_message)
                
                # Check for tool calls
                if not response_message.tool_calls:
                    # No tool calls, return the response
                    await self.message_bus.publish(
                        AdvancedChatEngineStatusEvent(
                            status="completed", 
                            session_id=self.session_id
                        )
                    )
                    return CommandResult(
                        success=True, 
                        result=response_message.content or "", 
                        session_id=self.session_id
                    )
                
                # Process tool calls
                for tool_call in response_message.tool_calls:
                    tool_call_obj = ToolCall(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                    )
                    
                    try:
                        # Check if confirmation is needed
                        if self.tool_manager.needs_confirmation(tool_call_obj.name):
                            arguments = json.loads(tool_call_obj.arguments)
                            await self.message_bus.publish(
                                AdvancedChatEngineConfirmationEvent(
                                    tool_name=tool_call_obj.name,
                                    arguments=arguments,
                                    description=f"Execute {tool_call_obj.name} with arguments: {arguments}",
                                    session_id=self.session_id
                                )
                            )
                            # For now, assume confirmation (in real implementation, this would be interactive)
                            confirmed = True
                        else:
                            confirmed = False
                        
                        # Execute the tool
                        await self.message_bus.publish(
                            AdvancedChatEngineStatusEvent(
                                status=f"executing tool: {tool_call_obj.name}", 
                                session_id=self.session_id
                            )
                        )
                        
                        result = await self.tool_manager.execute_tool_call(tool_call_obj)
                        
                        # Convert result to string if needed
                        if isinstance(result, dict):
                            result_str = json.dumps(result)
                        else:
                            result_str = str(result)
                        
                        # Store tool result
                        self.context.add_tool_result(
                            tool_call_id=tool_call_obj.id,
                            name=tool_call_obj.name,
                            content=result_str,
                        )
                        
                        # Publish tool result event
                        await self.message_bus.publish(
                            AdvancedChatEngineToolResultEvent(
                                tool_name=tool_call_obj.name,
                                result=result_str,
                                confirmed=confirmed,
                                session_id=self.session_id,
                            )
                        )
                        
                    except Exception as e:
                        error_msg = f"Error executing tool {tool_call_obj.name}: {str(e)}"
                        print(error_msg)
                        
                        # Store error result
                        self.context.add_tool_result(
                            tool_call_id=tool_call_obj.id,
                            name=tool_call_obj.name,
                            content=error_msg,
                        )
                
                # Continue loop to get final response with tool results
        
        except Exception as e:
            print(f"ERROR in AdvancedChatEngine: {e}")
            import traceback
            traceback.print_exc()
            
            return CommandResult(
                success=False, 
                error=str(e), 
                session_id=self.session_id
            )
    
    async def register_tool(self, function: AsyncOrSyncToolFunction, require_confirmation: bool = False):
        """Register a function as a tool."""
        await self.tool_manager.register_tool(function, require_confirmation)
    
    async def clear_context(self):
        """Clear the conversation context."""
        self.context.clear()
    
    def set_system_prompt(self, prompt: str):
        """Set the system prompt."""
        self.context = AdvancedChatContext(
            self.engine_id, 
            self.session_id, 
            prompt
        )


# Example tools demonstrating Week 2 concepts
def calculate_area(length: float, width: float) -> float:
    """Calculate the area of a rectangle.
    
    Args:
        length: The length of the rectangle
        width: The width of the rectangle
        
    Returns:
        The area of the rectangle
    """
    return length * width


def get_system_info() -> Dict[str, str]:
    """Get basic system information (requires confirmation).
    
    Returns:
        Dictionary containing system information
    """
    import platform
    import os
    
    return {
        "platform": platform.system(),
        "version": platform.version(),
        "architecture": platform.architecture()[0],
        "current_directory": os.getcwd(),
        "user": os.getenv("USER", "unknown")
    }


async def search_web(query: str) -> Dict[str, str]:
    """Search the web for information (simulated).
    
    Args:
        query: The search query string
        
    Returns:
        Dictionary containing search results
    """
    # Simulated web search
    return {
        "query": query,
        "results": f"Simulated search results for: {query}",
        "source": "Advanced Chat Engine Web Search"
    }


async def main():
    """Main function to run the Advanced Chat Engine."""
    import os
    print(f"Current working directory: {os.getcwd()}")
    
    from llmgine.ui.cli.components import ToolComponent
    from llmgine.bootstrap import ApplicationBootstrap, ApplicationConfig
    
    # Bootstrap the application
    config = ApplicationConfig(enable_console_handler=False)
    bootstrap = ApplicationBootstrap(config)
    await bootstrap.bootstrap()
    
    # Create engine and register tools
    engine = AdvancedChatEngine(session_id=SessionID("advanced-test"))
    
    # Register tools with different confirmation requirements
    await engine.register_tool(calculate_area, require_confirmation=False)
    await engine.register_tool(get_system_info, require_confirmation=True)
    await engine.register_tool(search_web, require_confirmation=False)
    
    # Set up CLI
    cli = EngineCLI(SessionID("advanced-test"))
    cli.register_engine(engine)
    cli.register_engine_command(AdvancedChatEngineCommand, engine.handle_command)
    cli.register_engine_result_component(EngineResultComponent)
    cli.register_loading_event(AdvancedChatEngineStatusEvent)
    cli.register_component_event(AdvancedChatEngineToolResultEvent, ToolComponent)
    
    # Start the CLI
    await cli.main()


if __name__ == "__main__":
    asyncio.run(main()) 