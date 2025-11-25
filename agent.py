from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
import asyncio
import sys

load_dotenv()

async def create_agent():
    """Create agent with MCP tools."""
    # Connect to the Loomis MCP server
    client = MultiServerMCPClient({
        "loomis": {
            "command": sys.executable,
            "args": [r"c:\Users\USER\OneDrive\Desktop\Project\Loomis_Drawing_Assistant\mcp.py"],
            "transport": "stdio"
        }
    })
    
    # Get tools from MCP server
    tools = await client.get_tools()
    
    # Create agent with Ollama
    llm = ChatOllama(model="qwen2.5:1.5b", temperature=0)
    
    system_message = "You're a Loomis method drawing assistant. Help users apply geometric construction to draw heads accurately."
    
    agent = create_react_agent(llm, tools, prompt=system_message)
    
    return agent, client

async def run_agent(user_input: str) -> str:
    """Run the agent with MCP tools."""
    agent, client = await create_agent()
    
    try:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config={"recursion_limit": 50}
        )
        return result["messages"][-1].content
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        await client.cleanup()

def main():
    """Interactive CLI."""
    print("🎨 Loomis Drawing Assistant")
    print("=" * 50)
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            break
        
        response = asyncio.run(run_agent(user_input))
        print(f"\nAssistant: {response}")

if __name__ == "__main__":
    main()