"""
Tools JARVIS can use.

Each tool has:
  1. A JSON schema description (so Claude knows it exists and how to call it)
  2. A Python function that actually performs the action (Windows-specific)

To add a new tool: write the function, add its schema to TOOL_SCHEMAS,
and add a branch in execute_tool().
"""

import os
import subprocess
import webbrowser
import requests

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def open_application(app_name: str) -> str:
    """Open a Windows application by name (e.g. 'notepad', 'chrome', 'calc')."""
    try:
        os.startfile(app_name)  # Windows-only
        return f"Opened {app_name}, sir."
    except FileNotFoundError:
        # Fall back: try common apps mapped to their actual executable name
        common_apps = {
            "notepad": "notepad.exe",
            "calculator": "calc.exe",
            "paint": "mspaint.exe",
            "explorer": "explorer.exe",
            "file explorer": "explorer.exe",
            "task manager": "taskmgr.exe",
            "control panel": "control.exe",
        }
        exe = common_apps.get(app_name.lower())
        if exe:
            try:
                os.startfile(exe)
                return f"Opened {app_name}, sir."
            except Exception as e:
                return f"I couldn't open {app_name}: {e}"
        return f"I couldn't find an application called '{app_name}', sir."
    except Exception as e:
        return f"I couldn't open {app_name}: {e}"


def open_website(url: str) -> str:
    """Open a website in the default browser."""
    if not url.startswith("http"):
        url = "https://" + url
    webbrowser.open(url)
    return f"Opening {url}, sir."


def get_weather(city: str = None) -> str:
    """Get current weather for a city using OpenWeatherMap's free API."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "I don't have a weather API key configured yet, sir."

    city = city or os.getenv("OPENWEATHER_CITY", "London")
    try:
        resp = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": city, "appid": api_key, "units": "metric"},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        desc = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        feels = data["main"]["feels_like"]
        return (
            f"It's currently {temp:.0f}\u00b0C in {city}, feels like {feels:.0f}\u00b0C, "
            f"with {desc}, sir."
        )
    except Exception as e:
        return f"I couldn't retrieve the weather for {city}: {e}"


# ---------------------------------------------------------------------------
# Schemas Claude sees (Anthropic tool-use format)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "name": "open_application",
        "description": "Open a desktop application on the user's Windows machine, e.g. notepad, calculator, paint, chrome, spotify.",
        "input_schema": {
            "type": "object",
            "properties": {
                "app_name": {
                    "type": "string",
                    "description": "Name of the application to open, e.g. 'notepad' or 'chrome'.",
                }
            },
            "required": ["app_name"],
        },
    },
    {
        "name": "open_website",
        "description": "Open a website in the default web browser.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL or domain to open, e.g. 'youtube.com'.",
                }
            },
            "required": ["url"],
        },
    },
    {
        "name": "get_weather",
        "description": "Get the current weather for a given city. If no city is given, uses the user's default city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'Bengaluru'. Optional.",
                }
            },
            "required": [],
        },
    },
]


def execute_tool(name: str, tool_input: dict) -> str:
    """Dispatch a tool call by name to its implementation."""
    if name == "open_application":
        return open_application(tool_input["app_name"])
    if name == "open_website":
        return open_website(tool_input["url"])
    if name == "get_weather":
        return get_weather(tool_input.get("city"))
    return f"Unknown tool: {name}"
