import asyncio
from mcp import StdioServerParameters
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition, ToolNode
from typing import Annotated, List
from typing_extensions import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Import the MultiServerMCPClient
from langchain_mcp_adapters.client import MultiServerMCPClient

# --- Multi-server configuration dictionary ---
# This dictionary defines all the servers the client will connect to
server_configs = {
    "weather": {
        "command": "python",
        "args": ["weather_server.py"],  # Your original weather server
        "transport": "stdio",
    },
    "tasks": {
        "command": "python",
        "args": ["task_server.py"],  # The new task management server
        "transport": "stdio",
    }
}


# LangGraph state definition (remains the same)
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


async def list_all_prompts(client: MultiServerMCPClient, server_configs: dict):
    print("\nAvailable Prompts from all servers:")
    print("-----------------------------------")

    any_prompts_found = False

    # Iterate through the names of the servers defined in your server_configs
    for server_name in server_configs.keys():
        try:
            # Opening a session for a specific server to interact with its prompts
            async with client.session(server_name) as session:
                prompt_response = await session.list_prompts()

                if prompt_response and prompt_response.prompts:
                    any_prompts_found = True
                    # Print a header for the server to group the prompts
                    print(f"\n--- Server: '{server_name}' ---")
                    for p in prompt_response.prompts:
                        print(f"  Prompt: {p.name}")
                        if p.arguments:
                            arg_list = [f"{arg.name}" for arg in p.arguments]
                            print(f"    Arguments: {', '.join(arg_list)}")
                        else:
                            print("    Arguments: None")
        except Exception as e:
            print(f"\nCould not fetch prompts from server '{server_name}': {e}")

    print("\nUse: /prompt <server_name> <prompt_name> \"arg1\" \"arg2\" ...")
    print("-----------------------------------")
    if not any_prompts_found:
        print("\nNo prompts were found on any connected servers.")


import shlex


async def handle_prompt_invocation(client: MultiServerMCPClient, command: str) -> str | None:
    try:
        # Use shlex to correctly parse arguments, even with spaces
        parts = shlex.split(command.strip())

        # Command should be: /prompt <server_name> <prompt_name> [args...]
        if len(parts) < 3:
            print("\nUsage: /prompt <server_name> <prompt_name> \"arg1\" \"arg2\" ...")
            return None

        server_name = parts[1]
        prompt_name = parts[2]
        user_args = parts[3:]

        # --- Validate the prompt and arguments against the specific server ---
        prompt_def = None
        async with client.session(server_name) as session:
            all_prompts_resp = await session.list_prompts()
            if all_prompts_resp and all_prompts_resp.prompts:
                # Find the matching prompt definition
                prompt_def = next((p for p in all_prompts_resp.prompts if p.name == prompt_name), None)

        if not prompt_def:
            print(f"\nError: Prompt '{prompt_name}' not found on server '{server_name}'.")
            return None

        # Check if the number of user-provided arguments matches what the prompt expects
        if len(user_args) != len(prompt_def.arguments):
            expected_args = [arg.name for arg in prompt_def.arguments]
            print(f"\nError: Invalid number of arguments for prompt '{prompt_name}'.")
            print(f"Expected {len(expected_args)} arguments: {', '.join(expected_args)}")
            return

        # --- Fetch and execute the prompt ---
        arg_dict = {arg.name: val for arg, val in zip(prompt_def.arguments, user_args)}

        # Use the client's get_prompt method, specifying server, prompt, and args
        prompt_messages = await client.get_prompt(
            server_name=server_name,
            prompt_name=prompt_name,
            arguments=arg_dict
        )

        # The prompt text is the content of the first message returned
        prompt_text = prompt_messages[0].content

        print("\n--- Prompt loaded successfully. Preparing to execute... ---")

        # Return the fetched text instead of calling the agent
        return prompt_text

    except Exception as e:
        print(f"\nAn error occurred during prompt invocation: {e}")


async def list_all_resources(client: MultiServerMCPClient, server_configs: dict):
    print("\nAvailable Resources from all servers:")
    print("-------------------------------------")

    any_resources_found = False

    # Iterate through the names of the servers defined in your server_configs
    for server_name in server_configs.keys():
        try:
            # Opening a session for a specific server to list its resources
            async with client.session(server_name) as session:
                # The method to list resources is session.list_resources()
                resource_response = await session.list_resources()

                if resource_response and resource_response.resources:
                    any_resources_found = True
                    print(f"\n--- Server: '{server_name}' ---")
                    for r in resource_response.resources:
                        # The most important identifier for a resource is its URI
                        print(f"  Resource URI: {r.uri}")
                        if r.description:
                            print(f"    Description: {r.description}")
        except Exception as e:
            print(f"\nCould not fetch resources from server '{server_name}': {e}")

    print("\nUse: /resource <server_name> <resource_uri>")
    print("-----------------------------------")

    if not any_resources_found:
        print("\nNo resources were found on any connected servers.")


import shlex


async def handle_resource_invocation(client: MultiServerMCPClient, command: str) -> str | None:
    try:

        parts = command.strip().split()
        if len(parts) != 3:
            print("\nUsage: /resource <server_name> <resource_uri>")
            return None

        server_name = parts[1]
        resource_uri = parts[2]

        print(f"\n--- Fetching resource '{resource_uri}' from server '{server_name}'... ---")

        # It requires the server_name and the URI of the resource

        blobs = await client.get_resources(server_name=server_name, uris=[resource_uri])

        if not blobs:
            print("Error: Resource not found or content is empty.")
            return None

        # Converting LangChain Blobs to string content

        resource_content = blobs[0].as_string()

        print("--- Resource content loaded successfully. ---")
        return resource_content

    except Exception as e:
        print(f"\nAn error occurred while fetching the resource: {e}")
        return None


async def main():
    # The client will manage the server subprocesses internally
    client = MultiServerMCPClient(server_configs)

    # Get a single, unified list of tools from all connected servers
    all_tools = await client.get_tools()

    # Create the LangGraph agent with the aggregated list of tools
    agent = create_graph(all_tools)

    print("MCP Agent is ready (connected to Weather and Task servers).")
    print("Type a question, or use one of the following commands:")
    print("  /resources                             - to list available resources")
    print("  /resource <server_name> <resource_uri>      - to load a resource for the agent")

    message_to_agent = ""

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            break
        if user_input.lower() == "/resources":
            await list_all_resources(client, server_configs)
            continue
        elif user_input.startswith("/resource"):
            resource_content = await handle_resource_invocation(client, user_input)

            if resource_content:
                action_prompt = input(
                    "Resource loaded. What should I do with this content? (Press Enter to just save to context)\n> ").strip()

                # If user provides an action, combine it with the resource content
                if action_prompt:
                    message_to_agent = f"""
                    CONTEXT from a loaded resource:
                    ---
                    {resource_content}
                    ---
                    TASK: {action_prompt}
                    """

                # If user provides no action, create a default message to save the context
                else:
                    print("No action specified. Adding resource content to conversation memory...")
                    message_to_agent = f"""
                    Please remember the following context for our conversation. Just acknowledge that you have received it.
                    ---
                    CONTEXT:
                    {resource_content}
                    ---
                    """
            else:
                # If resource loading failed, loop back for next input
                continue

        else:
            # For a normal chat message, the message is just the user's input
            message_to_agent = user_input

        # Final agent invocation
        if message_to_agent:
            try:
                response = await agent.ainvoke(
                    {"messages": [("user", message_to_agent)]},
                    config={"configurable": {"thread_id": "multi-server-session"}}
                )
                print("AI:", response["messages"][-1].content)
            except Exception as e:
                print("Error:", e)

# --- 'create_graph' now accepts the list of tools directly ---
def create_graph(tools: list):
    # LLM configuration (remains the same)
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key="AIzaSyCbSWI26wv5-93DjystyU_H-Ca-MOxzjIw")
    llm_with_tools = llm.bind_tools(tools)

    # --- Updated system prompt to reflect new capabilities ---
    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. You have access to tools for checking the weather and managing a to-do list. Use the tools when necessary based on the user's request."),
        MessagesPlaceholder("messages")
    ])

    chat_llm = prompt_template | llm_with_tools

    # Define chat node (remains the same)
    def chat_node(state: State) -> State:
        response = chat_llm.invoke({"messages": state["messages"]})
        return {"messages": [response]}

    # Build LangGraph with tool routing (remains the same)
    graph = StateGraph(State)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tool_node", ToolNode(tools=tools))
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition, {
        "tools": "tool_node",
        "__end__": END
    })
    graph.add_edge("tool_node", "chat_node")

    return graph.compile(checkpointer=MemorySaver())


async def main():
    # Instantiate the client which will manage the server subprocesses internally
    client = MultiServerMCPClient(server_configs)

    # Get a single, unified list of tools from all connected servers
    all_tools = await client.get_tools()

    # Create the LangGraph agent with the aggregated list of tools
    agent = create_graph(all_tools)

    print("MCP Agent is ready (connected to Weather and Task servers).")
    # Update the instructions for the user
    print("Type a question, or use one of the following commands:")
    print("  /prompts                                       - to list available prompts")
    print("  /prompt <server_name> <prompt_name> \"args\"   - to run a specific prompt")

    while True:
        # This variable will hold the final message to be sent to the agent
        message_to_agent = ""

        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            break

        # --- Command Handling Logic ---
        if user_input.lower() == "/prompts":
            await list_all_prompts(client, server_configs)
            continue  # Command is done, loop back for next input

        elif user_input.startswith("/prompt"):

            # The function returns the prompt text or None.
            prompt_text = await handle_prompt_invocation(client, user_input)
            if prompt_text:
                message_to_agent = prompt_text
            else:
                # If prompt fetching failed, loop back for next input
                continue

        else:
            # For a normal chat message, the message is just the user's input
            message_to_agent = user_input

        # --- Final agent invocation ---

        if message_to_agent:
            try:
                response = await agent.ainvoke(
                    {"messages": [("user", message_to_agent)]},
                    config={"configurable": {"thread_id": "multi-server-session"}}
                )
                print("AI:", response["messages"][-1].content)
            except Exception as e:
                print("Error:", e)


if __name__ == "__main__":
    asyncio.run(main())