#!/usr/bin/env python3
"""
Advanced Design Patterns Implementation for Week 2 Specification

This module demonstrates mastery of the design patterns mentioned in the spec:
- Singleton Pattern: Engine configuration and database connections
- Observer Pattern: Event-driven tool execution notifications
- Mediator Pattern: Centralized communication between engine components

References:
- https://refactoring.guru/design-patterns/singleton
- https://refactoring.guru/design-patterns/observer
- https://refactoring.guru/design-patterns/mediator
"""

import asyncio
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone


# ==================== SINGLETON PATTERN ====================
class EngineConfigSingleton:
    """
    Singleton pattern for engine configuration management.
    Ensures single source of truth for engine settings across the application.
    """
    _instance: Optional['EngineConfigSingleton'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'EngineConfigSingleton':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not getattr(self, '_initialized', False):
            self.config: Dict[str, Any] = {
                'max_tool_execution_time': 30.0,
                'enable_tool_caching': True,
                'max_context_length': 4000,
                'retry_attempts': 3,
                'rate_limit_per_minute': 60,
                'enable_security_sandbox': True,
                'debug_mode': False
            }
            self.performance_metrics: Dict[str, Any] = {}
            self._initialized = True
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config[key] = value
    
    def update_metric(self, metric_name: str, value: Any) -> None:
        """Update performance metric."""
        self.performance_metrics[metric_name] = {
            'value': value,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


# ==================== OBSERVER PATTERN ====================
class ToolExecutionObserver(ABC):
    """Abstract observer for tool execution events."""
    
    @abstractmethod
    async def on_tool_started(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Called when tool execution starts."""
        pass
    
    @abstractmethod
    async def on_tool_completed(self, tool_name: str, result: Any, execution_time: float) -> None:
        """Called when tool execution completes successfully."""
        pass
    
    @abstractmethod
    async def on_tool_failed(self, tool_name: str, error: Exception, execution_time: float) -> None:
        """Called when tool execution fails."""
        pass


class PerformanceMonitor(ToolExecutionObserver):
    """Observer that monitors tool performance metrics."""
    
    def __init__(self):
        self.execution_stats: Dict[str, List[float]] = {}
        self.error_counts: Dict[str, int] = {}
        self.config = EngineConfigSingleton()
    
    async def on_tool_started(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        # Log tool start for debugging
        if self.config.get('debug_mode'):
            print(f"🔧 Starting tool: {tool_name} with args: {arguments}")
    
    async def on_tool_completed(self, tool_name: str, result: Any, execution_time: float) -> None:
        # Track execution time statistics
        if tool_name not in self.execution_stats:
            self.execution_stats[tool_name] = []
        self.execution_stats[tool_name].append(execution_time)
        
        # Update global metrics
        self.config.update_metric(f'tool_{tool_name}_avg_time', 
                                self._calculate_average(self.execution_stats[tool_name]))
        
        if self.config.get('debug_mode'):
            print(f"✅ Completed tool: {tool_name} in {execution_time:.3f}s")
    
    async def on_tool_failed(self, tool_name: str, error: Exception, execution_time: float) -> None:
        # Track error statistics
        self.error_counts[tool_name] = self.error_counts.get(tool_name, 0) + 1
        self.config.update_metric(f'tool_{tool_name}_error_count', self.error_counts[tool_name])
        
        if self.config.get('debug_mode'):
            print(f"❌ Failed tool: {tool_name} after {execution_time:.3f}s - {error}")
    
    def _calculate_average(self, times: List[float]) -> float:
        """Calculate average execution time."""
        return sum(times) / len(times) if times else 0.0
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            'tool_statistics': {},
            'global_metrics': self.config.performance_metrics,
            'generated_at': datetime.now(timezone.utc).isoformat()
        }
        
        for tool_name, times in self.execution_stats.items():
            report['tool_statistics'][tool_name] = {
                'total_executions': len(times),
                'average_time': self._calculate_average(times),
                'min_time': min(times) if times else 0,
                'max_time': max(times) if times else 0,
                'error_count': self.error_counts.get(tool_name, 0),
                'success_rate': (len(times) - self.error_counts.get(tool_name, 0)) / len(times) * 100 if times else 0
            }
        
        return report


class SecurityAuditor(ToolExecutionObserver):
    """Observer that audits tool execution for security concerns."""
    
    def __init__(self):
        self.security_events: List[Dict[str, Any]] = []
        self.suspicious_patterns = {
            'file_access': ['open', 'read', 'write', 'delete'],
            'network_access': ['http', 'https', 'ftp', 'ssh'],
            'system_commands': ['exec', 'system', 'subprocess', 'shell']
        }
    
    async def on_tool_started(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        # Check for potentially dangerous operations
        security_flags = self._analyze_security_risk(tool_name, arguments)
        if security_flags:
            event = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'tool_name': tool_name,
                'arguments': arguments,
                'security_flags': security_flags,
                'event_type': 'potential_security_risk'
            }
            self.security_events.append(event)
            print(f"⚠️ Security alert: {tool_name} flagged for: {', '.join(security_flags)}")
    
    async def on_tool_completed(self, tool_name: str, result: Any, execution_time: float) -> None:
        # Log successful completion for audit trail
        pass
    
    async def on_tool_failed(self, tool_name: str, error: Exception, execution_time: float) -> None:
        # Log failures for security analysis
        if "permission" in str(error).lower() or "access" in str(error).lower():
            event = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'tool_name': tool_name,
                'error': str(error),
                'event_type': 'access_violation'
            }
            self.security_events.append(event)
    
    def _analyze_security_risk(self, tool_name: str, arguments: Dict[str, Any]) -> List[str]:
        """Analyze tool execution for security risks."""
        flags = []
        
        # Check tool name for suspicious patterns
        for category, patterns in self.suspicious_patterns.items():
            if any(pattern in tool_name.lower() for pattern in patterns):
                flags.append(category)
        
        # Check arguments for file paths or URLs
        for key, value in arguments.items():
            if isinstance(value, str):
                if any(char in value for char in ['/', '\\', '..']):
                    flags.append('file_path_manipulation')
                if value.startswith(('http://', 'https://', 'ftp://')):
                    flags.append('external_url_access')
        
        return flags
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate security audit report."""
        return {
            'total_security_events': len(self.security_events),
            'events': self.security_events,
            'generated_at': datetime.now(timezone.utc).isoformat()
        }


class ToolExecutionSubject:
    """Subject in Observer pattern - manages tool execution observers."""
    
    def __init__(self):
        self._observers: Set[ToolExecutionObserver] = set()
    
    def attach(self, observer: ToolExecutionObserver) -> None:
        """Attach an observer to the subject."""
        self._observers.add(observer)
    
    def detach(self, observer: ToolExecutionObserver) -> None:
        """Detach an observer from the subject."""
        self._observers.discard(observer)
    
    async def notify_tool_started(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Notify all observers that tool execution started."""
        for observer in self._observers:
            await observer.on_tool_started(tool_name, arguments)
    
    async def notify_tool_completed(self, tool_name: str, result: Any, execution_time: float) -> None:
        """Notify all observers that tool execution completed."""
        for observer in self._observers:
            await observer.on_tool_completed(tool_name, result, execution_time)
    
    async def notify_tool_failed(self, tool_name: str, error: Exception, execution_time: float) -> None:
        """Notify all observers that tool execution failed."""
        for observer in self._observers:
            await observer.on_tool_failed(tool_name, error, execution_time)


# ==================== MEDIATOR PATTERN ====================
@dataclass
class EngineComponent:
    """Base class for engine components that communicate through mediator."""
    name: str
    mediator: Optional['EngineMediator'] = None
    
    def set_mediator(self, mediator: 'EngineMediator') -> None:
        """Set the mediator for this component."""
        self.mediator = mediator


@dataclass
class ComponentMessage:
    """Message passed between components through mediator."""
    sender: str
    receiver: str
    message_type: str
    payload: Any
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class EngineMediator:
    """
    Mediator pattern implementation for engine component communication.
    Centralizes communication between LLM Manager, Tool Manager, and Context Manager.
    """
    
    def __init__(self):
        self.components: Dict[str, EngineComponent] = {}
        self.message_history: List[ComponentMessage] = []
        self.config = EngineConfigSingleton()
    
    def register_component(self, component: EngineComponent) -> None:
        """Register a component with the mediator."""
        self.components[component.name] = component
        component.set_mediator(self)
        if self.config.get('debug_mode'):
            print(f"📱 Registered component: {component.name}")
    
    async def send_message(self, sender: str, receiver: str, message_type: str, payload: Any) -> Any:
        """Send a message from one component to another."""
        message = ComponentMessage(sender, receiver, message_type, payload)
        self.message_history.append(message)
        
        if self.config.get('debug_mode'):
            print(f"📨 {sender} → {receiver}: {message_type}")
        
        # Route message to appropriate handler
        if receiver in self.components:
            return await self._handle_message(message)
        else:
            raise ValueError(f"Component '{receiver}' not found")
    
    async def _handle_message(self, message: ComponentMessage) -> Any:
        """Handle message routing and processing."""
        receiver_component = self.components[message.receiver]
        
        # Route based on message type
        if message.message_type == "tool_execution_request":
            return await self._handle_tool_execution(message)
        elif message.message_type == "context_update":
            return await self._handle_context_update(message)
        elif message.message_type == "llm_generation_request":
            return await self._handle_llm_generation(message)
        elif message.message_type == "performance_query":
            return await self._handle_performance_query(message)
        else:
            # Default handling - just acknowledge
            return {"status": "acknowledged", "message": f"Message received by {message.receiver}"}
    
    async def _handle_tool_execution(self, message: ComponentMessage) -> Any:
        """Handle tool execution coordination."""
        payload = message.payload
        
        # Could coordinate between Tool Manager and Security Auditor
        result = {
            "status": "coordinated",
            "tool_name": payload.get("tool_name"),
            "execution_approved": True,  # Could implement approval logic
            "security_flags": []  # Could integrate with security auditor
        }
        
        return result
    
    async def _handle_context_update(self, message: ComponentMessage) -> Any:
        """Handle context management coordination."""
        # Could coordinate between Context Manager and Performance Monitor
        return {"status": "context_updated", "timestamp": message.timestamp}
    
    async def _handle_llm_generation(self, message: ComponentMessage) -> Any:
        """Handle LLM generation coordination."""
        # Could coordinate between LLM Manager and various observers
        return {"status": "generation_coordinated", "tokens_estimated": 150}
    
    async def _handle_performance_query(self, message: ComponentMessage) -> Any:
        """Handle performance monitoring queries."""
        return {
            "status": "performance_data",
            "metrics": self.config.performance_metrics,
            "message_count": len(self.message_history)
        }
    
    def get_communication_report(self) -> Dict[str, Any]:
        """Generate report of component communications."""
        return {
            'total_messages': len(self.message_history),
            'registered_components': list(self.components.keys()),
            'message_types': list(set(msg.message_type for msg in self.message_history)),
            'recent_messages': self.message_history[-10:] if self.message_history else [],
            'generated_at': datetime.now(timezone.utc).isoformat()
        }


# ==================== INTEGRATION EXAMPLE ====================
class AdvancedPatternDemo:
    """Demonstration of all three design patterns working together."""
    
    def __init__(self):
        # Singleton pattern
        self.config = EngineConfigSingleton()
        
        # Observer pattern
        self.tool_subject = ToolExecutionSubject()
        self.performance_monitor = PerformanceMonitor()
        self.security_auditor = SecurityAuditor()
        
        # Mediator pattern
        self.mediator = EngineMediator()
        
        # Setup observers
        self.tool_subject.attach(self.performance_monitor)
        self.tool_subject.attach(self.security_auditor)
        
        # Setup components
        self._setup_components()
    
    def _setup_components(self):
        """Setup mediator components."""
        llm_component = EngineComponent("LLMManager")
        tool_component = EngineComponent("ToolManager")
        context_component = EngineComponent("ContextManager")
        
        self.mediator.register_component(llm_component)
        self.mediator.register_component(tool_component)
        self.mediator.register_component(context_component)
    
    async def simulate_tool_execution(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate a tool execution using all patterns."""
        start_time = time.time()
        
        try:
            # Observer pattern: Notify start
            await self.tool_subject.notify_tool_started(tool_name, arguments)
            
            # Mediator pattern: Coordinate execution
            coordination_result = await self.mediator.send_message(
                "ToolManager", "LLMManager", "tool_execution_request",
                {"tool_name": tool_name, "arguments": arguments}
            )
            
            # Simulate actual execution
            await asyncio.sleep(0.1)  # Simulate work
            
            execution_time = time.time() - start_time
            result = f"Tool {tool_name} executed successfully"
            
            # Observer pattern: Notify completion
            await self.tool_subject.notify_tool_completed(tool_name, result, execution_time)
            
            return {
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "coordination": coordination_result
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            await self.tool_subject.notify_tool_failed(tool_name, e, execution_time)
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time
            }
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive report using all patterns."""
        return {
            "design_patterns_demo": {
                "singleton_config": {
                    "pattern": "Singleton",
                    "description": "Centralized configuration management",
                    "current_config": self.config.config,
                    "metrics": self.config.performance_metrics
                },
                "observer_monitoring": {
                    "pattern": "Observer",
                    "description": "Event-driven tool execution monitoring",
                    "performance_report": self.performance_monitor.get_performance_report(),
                    "security_report": self.security_auditor.get_security_report()
                },
                "mediator_communication": {
                    "pattern": "Mediator",
                    "description": "Centralized component communication",
                    "communication_report": self.mediator.get_communication_report()
                }
            },
            "patterns_integration": {
                "demonstrates": [
                    "Loose coupling between components",
                    "Centralized configuration management",
                    "Event-driven architecture",
                    "Security monitoring",
                    "Performance tracking",
                    "Coordinated component communication"
                ],
                "benefits": [
                    "Maintainable code structure",
                    "Scalable architecture",
                    "Comprehensive monitoring",
                    "Security auditing",
                    "Performance optimization"
                ]
            }
        }


# ==================== DEMONSTRATION FUNCTION ====================
async def demonstrate_advanced_patterns():
    """Demonstrate all advanced design patterns working together."""
    print("🎯 Advanced Design Patterns Demonstration\n")
    print("=" * 60)
    
    demo = AdvancedPatternDemo()
    
    # Enable debug mode for detailed output
    demo.config.set('debug_mode', True)
    
    # Simulate various tool executions
    tools_to_test = [
        ("calculate_area", {"length": 5.0, "width": 3.0}),
        ("get_system_info", {}),
        ("search_web", {"query": "python design patterns"}),
        ("file_processor", {"file_path": "/etc/passwd"}),  # Will trigger security flags
        ("network_request", {"url": "https://api.example.com"})  # Will trigger security flags
    ]
    
    print("\n🔧 Simulating Tool Executions with Pattern Integration:\n")
    
    for tool_name, arguments in tools_to_test:
        result = await demo.simulate_tool_execution(tool_name, arguments)
        print(f"Tool: {tool_name} - {'✅ Success' if result['success'] else '❌ Failed'}")
        if not result['success']:
            print(f"  Error: {result['error']}")
        print(f"  Execution time: {result['execution_time']:.3f}s\n")
    
    # Generate comprehensive report
    print("\n📊 Comprehensive Pattern Integration Report:\n")
    report = demo.generate_comprehensive_report()
    
    print("🔄 Singleton Pattern - Configuration Management:")
    singleton_info = report['design_patterns_demo']['singleton_config']
    print(f"  Current config keys: {list(singleton_info['current_config'].keys())}")
    print(f"  Tracked metrics: {len(singleton_info['metrics'])}")
    
    print("\n👀 Observer Pattern - Event Monitoring:")
    observer_info = report['design_patterns_demo']['observer_monitoring']
    perf_report = observer_info['performance_report']
    security_report = observer_info['security_report']
    print(f"  Performance stats for {len(perf_report['tool_statistics'])} tools")
    print(f"  Security events detected: {security_report['total_security_events']}")
    
    print("\n📱 Mediator Pattern - Component Communication:")
    mediator_info = report['design_patterns_demo']['mediator_communication']
    comm_report = mediator_info['communication_report']
    print(f"  Total messages: {comm_report['total_messages']}")
    print(f"  Registered components: {comm_report['registered_components']}")
    print(f"  Message types: {comm_report['message_types']}")
    
    print("\n✨ Pattern Integration Benefits:")
    for benefit in report['patterns_integration']['benefits']:
        print(f"  • {benefit}")
    
    return report


if __name__ == "__main__":
    # Run the demonstration
    report = asyncio.run(demonstrate_advanced_patterns())
    
    # Save report to file for analysis
    import json
    with open('advanced_patterns_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n💾 Detailed report saved to 'advanced_patterns_report.json'")
    print("🎉 Advanced Design Patterns demonstration completed!") 