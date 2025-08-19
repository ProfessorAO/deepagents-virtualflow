#!/usr/bin/env python3
"""
Simple test script to verify preloaded files functionality.
"""

import sys
import os

# Add the src directory to the path so we can import deepagents
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from deepagents import create_deep_agent

def test_preloaded_files():
    """Test that preloaded files are available to the agent."""
    
    # Define test files
    test_files = {
        "test.txt": "This is a test file with some content.",
        "config.json": '{"setting": "value", "enabled": true}',
        "src/helper.py": "def helper_function():\n    return 'Hello from helper!'"
    }
    
    # Create agent with preloaded files
    agent = create_deep_agent(
        tools=[],
        instructions="You are a test assistant. List all files you have access to.",
        pre_loaded_files=test_files
    )
    
    # Test that files are available
    result = agent.invoke({
        "messages": [{"role": "user", "content": "What files do you have access to? Please list them."}]
    })
    
    # Check that our files are in the result
    final_files = result.get("files", {})
    
    print("Test files that were preloaded:")
    for path, content in test_files.items():
        print(f"  {path}: {content[:50]}{'...' if len(content) > 50 else ''}")
    
    print("\nFiles in final state:")
    for path, content in final_files.items():
        print(f"  {path}: {content[:50]}{'...' if len(content) > 50 else ''}")
    
    # Verify all test files are present
    for path in test_files:
        if path not in final_files:
            print(f"❌ ERROR: {path} not found in final state")
            return False
        elif final_files[path] != test_files[path]:
            print(f"❌ ERROR: Content mismatch for {path}")
            return False
    
    print("✅ All preloaded files are present and correct!")
    return True

if __name__ == "__main__":
    print("Testing preloaded files functionality...")
    success = test_preloaded_files()
    sys.exit(0 if success else 1)
