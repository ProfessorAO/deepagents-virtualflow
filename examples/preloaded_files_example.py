"""
Example demonstrating how to preload files into a DeepAgent's virtual filesystem.

This shows two approaches:
1. Using the pre_loaded_files parameter when creating the agent
2. Passing files directly when invoking the agent
"""

from deepagents import create_deep_agent

def example_with_preloaded_files():
    """Example using the pre_loaded_files parameter."""
    
    # Define some files to preload
    initial_files = {
        "README.md": "# My Project\n\nThis is a sample project with preloaded files.",
        "src/main.py": "def main():\n    print('Hello from preloaded file!')\n\nif __name__ == '__main__':\n    main()",
        "config/settings.json": '{\n    "debug": true,\n    "port": 8080\n}'
    }
    
    # Create agent with preloaded files
    agent = create_deep_agent(
        tools=[],  # No additional tools needed for this example
        instructions="You are a helpful assistant with access to preloaded files.",
        pre_loaded_files=initial_files
    )
    
    # Invoke the agent - files are automatically available
    result = agent.invoke({
        "messages": [{"role": "user", "content": "List all available files and show me the content of README.md"}]
    })
    
    print("Files in state after agent run:")
    print(result.get("files", {}))
    
    return result

def example_with_direct_file_passing():
    """Example passing files directly when invoking the agent."""
    
    # Create agent without preloaded files
    agent = create_deep_agent(
        tools=[],
        instructions="You are a helpful assistant."
    )
    
    # Pass files directly when invoking
    result = agent.invoke({
        "messages": [{"role": "user", "content": "What files do you have access to?"}],
        "files": {
            "data.csv": "name,age\nAlice,30\nBob,25\nCharlie,35",
            "notes.txt": "Important meeting notes:\n- Discuss project timeline\n- Review budget"
        }
    })
    
    print("Files in state after agent run:")
    print(result.get("files", {}))
    
    return result

if __name__ == "__main__":
    print("=== Example 1: Using pre_loaded_files parameter ===")
    example_with_preloaded_files()
    
    print("\n=== Example 2: Passing files directly ===")
    example_with_direct_file_passing()
