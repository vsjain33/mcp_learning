import os
from mcp.server.fastmcp import FastMCP
from typing import List

# Define the file where tasks will be stored
TASKS_FILE = "tasks.txt"

# Initialize the FastMCP server with a descriptive name
mcp = FastMCP("TaskManagementAssistant")


from pathlib import Path

@mcp.resource("file://meeting_notes")
def meeting_notes_resource() -> list[str]:
    """
    Reads a meeting notes file and returns its contents as a list of lines.
    The notes contain discussion points and action items for an Ed Tech company.
    """
    try:
        notes_file = Path("meeting_notes.txt")
        if not notes_file.exists():
            return ["Error: The meeting_notes.txt file was not found on the server."]

        # Read the file, remove leading/trailing whitespace, and split into lines
        return notes_file.read_text(encoding="utf-8").strip().splitlines()

    except Exception as e:
        return [f"An unexpected error occurred while reading the meeting notes: {str(e)}"]

@mcp.prompt()
def plan_trip_prompt(destination: str, duration_in_days: int) -> str:
    """
    Creates a sample travel itinerary for a given destination and duration, then saves it as a series of tasks in the user's to-do list.
    This is the best prompt to use when a user asks to plan a trip, create an itinerary, or asks for travel suggestions.

    Args:
        destination: The city or country for the trip (e.g., "Paris", "Japan").
        duration_in_days: The number of days for the trip (e.g., 3).
    """
    return f"""
    You are an expert travel consultant. Your goal is to help the user by generating a sample travel itinerary and saving it to their task list for later reference.

    The user wants a plan for a {duration_in_days}-day trip to {destination}.

    Follow these steps carefully:
    1.  First, use your general knowledge to brainstorm a simple, day-by-day itinerary. Suggest one or two key attractions or activities for each day of the trip.
    2.  After you have formulated the plan, you MUST perform a critical action: for each individual activity or attraction in your suggested itinerary, save it to the user's task list. For example, if you suggest visiting the Louvre, you must call the tool for that specific item.
    3.  Once all the itinerary items have been added as tasks, present a friendly confirmation message to the user. Inform them that you have created a sample plan and saved it to their to-do list.
    """

@mcp.tool()
def add_task(task_description: str) -> str:
    """
    Adds a new task to the persistent task list file.

    This tool will create the task file if it doesn't exist. It appends
    the new task to the end of the file, ensuring each task is on a new line.

    Args:
        task_description: A string describing the task to be added.
                          For example: "Buy groceries" or "Finish the report".

    Returns:
        A string confirming that the task was successfully added.
    """
    try:
        # 'a' mode will append to the file, and create it if it doesn't exist
        with open(TASKS_FILE, "a") as f:
            f.write(f"{task_description}\n")
        return f"Task '{task_description}' was added successfully."
    except Exception as e:
        return f"An error occurred while adding the task: {e}"

@mcp.tool()
def list_tasks() -> List[str]:
    """
    Lists all the tasks from the persistent task list file.

    This tool reads all tasks from the file. If the file does not exist or
    is empty, it returns an empty list.

    Returns:
        A list of strings, where each string is a task. Returns an empty
        list if no tasks are found.
    """
    if not os.path.exists(TASKS_FILE):
        # Return an empty list if the file doesn't exist, as there are no tasks
        # The LLM can interpret this as "no tasks to show"
        return []

    try:
        with open(TASKS_FILE, "r") as f:
            # Read all lines, and strip leading/trailing whitespace (like newlines)
            tasks = [line.strip() for line in f.readlines()]
            # Filter out any empty lines that might have been created
            return [task for task in tasks if task]
    except Exception as e:
        # In case of an error, we can return a list with an error message,
        # but for simplicity and better type consistency, we'll return an empty list.
        # The LLM can be prompted to handle this gracefully
        print(f"Error reading tasks file: {e}")
        return []

if __name__ == "__main__":
    print("Starting MCP Task Server...")
    # The server will run and listen for requests from the client over stdio
    mcp.run(transport="stdio")