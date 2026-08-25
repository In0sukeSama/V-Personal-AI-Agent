"""
Tools V can use.

Each tool is a plain function decorated with @register_tool(...), which
registers its name, description, and parameter schema with the central
ToolRegistry (see tool_registry.py). There is no separate schema list and no
if/elif dispatcher to maintain - the registry is the single source of truth,
and brain.py executes tools via registry.execute(name, args).

To add a new tool: write a function, decorate it with @register_tool(...).
That's it - it's automatically available to the model and executable.
"""

import os
import subprocess
import webbrowser
import glob
import json
import shutil
import requests

from tool_registry import register_tool, RISK_SAFE, RISK_LOW, RISK_CONSEQUENTIAL, RISK_DESTRUCTIVE

MEMORY_FILE = os.path.join(os.path.dirname(__file__), "memory.json")


def _load_memory() -> dict:
    if not os.path.exists(MEMORY_FILE):
        return {}
    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_memory(data: dict):
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)


@register_tool(
    name="remember",
    description=(
        "Save a piece of information to long-term memory for future conversations, "
        "e.g. a username, preference, or fact about the user. Use this whenever the "
        "user tells you something worth remembering, or explicitly asks you to "
        "remember it."
    ),
    parameters={
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "A short identifier for this fact, e.g. 'github_username' or 'favorite_city'.",
            },
            "value": {
                "type": "string",
                "description": "The actual value to remember, e.g. 'jagat123'.",
            },
        },
        "required": ["key", "value"],
    },
    category="memory",
    risk=RISK_LOW,
)
def remember(key: str, value: str) -> str:
    """Save a fact to long-term memory, e.g. remember('github_username', 'jagat123')."""
    data = _load_memory()
    data[key] = value
    _save_memory(data)
    return f"Locked it in - {key} is now {value}."


@register_tool(
    name="recall",
    description=(
        "Retrieve a previously remembered fact by its key, e.g. 'github_username'. "
        "Use this before asking the user for information they may have already told you. "
        "Matching is fuzzy - close variations of the key (e.g. 'github' vs 'github_username') "
        "will still find a match."
    ),
    parameters={
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "The identifier of the fact to retrieve, e.g. 'github_username'.",
            }
        },
        "required": ["key"],
    },
    category="memory",
    risk=RISK_SAFE,
)
def recall(key: str) -> str:
    """Retrieve a previously remembered fact by key. Matching is fuzzy: falls
    back to substring matching against stored keys if there's no exact hit,
    since the model may not phrase the same key identically across turns."""
    data = _load_memory()

    # Exact match first
    if key in data:
        return f"{key}: {data[key]}"

    # Fuzzy fallback: normalize (lowercase, strip spaces/underscores) and
    # check for substring overlap in either direction against stored keys.
    key_norm = key.lower().replace(" ", "").replace("_", "")
    for stored_key, value in data.items():
        stored_norm = stored_key.lower().replace(" ", "").replace("_", "")
        if key_norm in stored_norm or stored_norm in key_norm:
            return f"{stored_key}: {value}"

    return f"No memory logged for '{key}' yet."


@register_tool(
    name="list_memories",
    description=(
        "List everything currently saved in long-term memory. Use this when unsure "
        "what's been remembered, or when recall for a specific key comes up empty but "
        "the info might be saved under a different key name."
    ),
    parameters={"type": "object", "properties": {}, "required": []},
    category="memory",
    risk=RISK_SAFE,
)
def list_memories() -> str:
    """Return every key-value pair currently in long-term memory."""
    data = _load_memory()
    if not data:
        return "Nothing saved in memory yet."
    lines = [f"{k}: {v}" for k, v in data.items()]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File operations
# ---------------------------------------------------------------------------

_SEARCH_DIRS = [
    os.path.expanduser("~/Desktop"),
    os.path.expanduser("~/Documents"),
    os.path.expanduser("~/Downloads"),
    os.path.expanduser("~/Pictures"),
    os.path.expanduser("~/OneDrive/Desktop"),
    os.path.expanduser("~/OneDrive/Documents"),
]

_QUARANTINE_DIR = os.path.join(os.path.dirname(__file__), "deleted_files")


@register_tool(
    name="find_file",
    description="Search common user folders (Desktop, Documents, Downloads, Pictures) for a file matching a name.",
    parameters={
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Full or partial filename to search for, e.g. 'resume.pdf' or 'resume'.",
            }
        },
        "required": ["filename"],
    },
    category="files",
    risk=RISK_SAFE,
)
def find_file(filename: str) -> str:
    """Search common user folders (Desktop, Documents, Downloads, Pictures) for a file by name."""
    filename_lower = filename.lower()
    matches = []
    for base_dir in _SEARCH_DIRS:
        if not os.path.isdir(base_dir):
            continue
        for root, _, files in os.walk(base_dir):
            for f in files:
                if filename_lower in f.lower():
                    matches.append(os.path.join(root, f))
            if len(matches) >= 10:
                break
        if len(matches) >= 10:
            break

    if not matches:
        return f"No file matching '{filename}' turned up in the usual spots, choom."
    if len(matches) == 1:
        return f"Found it: {matches[0]}"
    listing = "\n".join(matches[:10])
    return f"Got {len(matches)} matches:\n{listing}"


@register_tool(
    name="move_file",
    description="Move a file from one path to another. Use find_file first if you don't have the exact source path.",
    parameters={
        "type": "object",
        "properties": {
            "source": {"type": "string", "description": "Full path of the file to move."},
            "destination": {"type": "string", "description": "Full destination path, including filename."},
        },
        "required": ["source", "destination"],
    },
    category="files",
    risk=RISK_LOW,
)
def move_file(source: str, destination: str) -> str:
    """Move a file from source path to destination path/folder."""
    if not os.path.exists(source):
        return f"Can't find '{source}' - check the path's right."
    try:
        os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
        shutil.move(source, destination)
        return f"Moved. {os.path.basename(source)} is now at {destination}."
    except Exception as e:
        return f"Hit static moving that file: {e}"


@register_tool(
    name="delete_file",
    description=(
        "Delete a file. For safety this moves the file to a local quarantine folder "
        "rather than permanently erasing it. Use find_file first if you don't have the exact path."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Full path of the file to delete."}
        },
        "required": ["path"],
    },
    category="files",
    risk=RISK_DESTRUCTIVE,
    reversible=True,  # quarantine folder makes it recoverable, but it's still a destructive user-facing action
)
def delete_file(path: str) -> str:
    """'Delete' a file by moving it to a local quarantine folder, not a permanent erase."""
    if not os.path.exists(path):
        return f"Can't find '{path}' - nothing to delete there."
    try:
        os.makedirs(_QUARANTINE_DIR, exist_ok=True)
        dest = os.path.join(_QUARANTINE_DIR, os.path.basename(path))
        shutil.move(path, dest)
        return (
            f"{os.path.basename(path)} pulled from sight, sir - not gone for good though, "
            f"it's parked in the deleted_files folder in case you need it back."
        )
    except Exception as e:
        return f"Hit static deleting that file: {e}"


@register_tool(
    name="open_file",
    description=(
        "Open a local file (document, image, PDF, video, etc.) with its default desktop "
        "application. Use this - not open_website - for any actual file on disk. Use "
        "find_file first if you don't have the exact path."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Full path of the file to open."}
        },
        "required": ["path"],
    },
    category="files",
    risk=RISK_LOW,
)
def open_file(path: str) -> str:
    """Open a local file (document, image, pdf, etc.) with its default desktop
    application - NOT the browser. Use find_file first if the exact path isn't known."""
    if not os.path.exists(path):
        return f"Can't find '{path}' on disk - try find_file first to locate it."
    try:
        os.startfile(path)
        return f"{os.path.basename(path)} is open."
    except Exception as e:
        return f"Hit static opening that file: {e}"


# ---------------------------------------------------------------------------
# Application control
# ---------------------------------------------------------------------------

_START_MENU_DIRS = [
    os.path.expandvars(r"%APPDATA%\Microsoft\Windows\Start Menu\Programs"),
    os.path.expandvars(r"%ProgramData%\Microsoft\Windows\Start Menu\Programs"),
]


def _find_shortcut(app_name: str) -> str:
    """Search Start Menu shortcut folders (recursively) for a .lnk matching app_name."""
    app_name_lower = app_name.lower()
    best_match = None
    for base_dir in _START_MENU_DIRS:
        if not os.path.isdir(base_dir):
            continue
        for path in glob.glob(os.path.join(base_dir, "**", "*.lnk"), recursive=True):
            filename = os.path.splitext(os.path.basename(path))[0].lower()
            if filename == app_name_lower:
                return path
            if app_name_lower in filename and best_match is None:
                best_match = path
    return best_match


def _find_appsfolder_app(app_name: str) -> str:
    """Search Windows' unified Apps Folder (shell:AppsFolder), which lists
    BOTH traditional desktop apps and Microsoft Store/UWP apps in one place
    with their AppUserModelIDs."""
    try:
        ps_command = (
            "Get-StartApps | "
            "Select-Object Name, AppID | "
            "ConvertTo-Csv -NoTypeInformation"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_command],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None

        app_name_lower = app_name.lower()
        best_match = None
        for line in result.stdout.splitlines()[1:]:
            parts = line.rsplit(",", 1)
            if len(parts) != 2:
                continue
            name, app_id = parts[0].strip('"'), parts[1].strip('"')
            if name.lower() == app_name_lower:
                return app_id
            if app_name_lower in name.lower() and best_match is None:
                best_match = app_id
        return best_match
    except Exception:
        return None


@register_tool(
    name="open_application",
    description=(
        "Open a desktop application on the user's Windows machine, e.g. notepad, "
        "calculator, paint, chrome, spotify, steam, or any installed game/app."
    ),
    parameters={
        "type": "object",
        "properties": {
            "app_name": {"type": "string", "description": "Name of the application to open, e.g. 'notepad' or 'chrome'."}
        },
        "required": ["app_name"],
    },
    category="apps",
    timeout=15,  # may shell out to PowerShell (Get-StartApps) which can be slow
    risk=RISK_LOW,
)
def open_application(app_name: str) -> str:
    """Open a Windows application by name. Tries the actual desktop app first
    (via Windows' unified Apps Folder, which covers both regular and
    Microsoft Store apps); falls back to a web version in the browser if
    genuinely not installed."""
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
            return f"{app_name} is up and running."
        except Exception as e:
            return f"Hit some static trying to open {app_name}: {e}"

    app_id = _find_appsfolder_app(app_name)
    if app_id:
        try:
            os.startfile(f"shell:appsFolder\\{app_id}")
            return f"{app_name} is up and running."
        except Exception as e:
            return f"Found {app_name}, but hit static trying to launch it: {e}"

    shortcut = _find_shortcut(app_name)
    if shortcut:
        try:
            os.startfile(shortcut)
            return f"{app_name} is up and running."
        except Exception as e:
            return f"Found {app_name}, but hit static trying to launch it: {e}"

    web_fallbacks = {
        "spotify": "https://open.spotify.com",
        "whatsapp": "https://web.whatsapp.com",
        "discord": "https://discord.com/app",
        "youtube": "https://youtube.com",
        "netflix": "https://netflix.com",
        "gmail": "https://mail.google.com",
        "outlook": "https://outlook.com",
        "slack": "https://slack.com",
        "notion": "https://notion.so",
        "figma": "https://figma.com",
    }
    web_url = web_fallbacks.get(app_name.lower())
    if web_url:
        webbrowser.open(web_url)
        return f"{app_name}'s not installed on this rig, so I pulled it up in the browser instead."

    return (
        f"Can't find '{app_name}' on this rig, and I don't have a web fallback for it either. "
        f"If it's tucked away somewhere unusual, open it manually once so I can learn the "
        f"shortcut, or give me the exact path."
    )


@register_tool(
    name="open_website",
    description="Open a website in the default web browser.",
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL or domain to open, e.g. 'youtube.com'."}
        },
        "required": ["url"],
    },
    category="web",
    risk=RISK_LOW,
)
def open_website(url: str) -> str:
    """Open a website in the default browser."""
    if not url.startswith("http"):
        url = "https://" + url
    webbrowser.open(url)
    return f"Pulling up {url} now."


def _get_running_processes() -> list:
    """Return (process_name, window_title) for all running processes with a
    visible window."""
    try:
        ps_command = (
            "Get-Process | Where-Object {$_.MainWindowTitle -ne ''} | "
            "Select-Object ProcessName, MainWindowTitle | "
            "ConvertTo-Csv -NoTypeInformation"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_command],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
        processes = []
        for line in result.stdout.splitlines()[1:]:
            parts = line.rsplit(",", 1)
            if len(parts) != 2:
                continue
            proc_name, title = parts[0].strip('"'), parts[1].strip('"')
            processes.append((proc_name, title))
        return processes
    except Exception:
        return []


def _find_matching_process(app_name: str) -> tuple:
    """Fuzzy-match a spoken app name against currently running processes,
    checking both the raw process name and its window title."""
    app_name_clean = app_name.lower().replace(" ", "")
    processes = _get_running_processes()

    exact = None
    close_matches = []
    for proc_name, title in processes:
        proc_clean = proc_name.lower().replace(" ", "").replace(".exe", "")
        title_clean = title.lower().replace(" ", "")
        if app_name_clean == proc_clean or app_name_clean == title_clean:
            exact = (proc_name, title)
            break
        if app_name_clean in proc_clean or app_name_clean in title_clean:
            close_matches.append((proc_name, title))

    return exact, close_matches


@register_tool(
    name="close_application",
    description="Close/quit a running desktop application by name, e.g. notepad, chrome, spotify.",
    parameters={
        "type": "object",
        "properties": {
            "app_name": {"type": "string", "description": "Name of the application to close, e.g. 'notepad' or 'chrome'."}
        },
        "required": ["app_name"],
    },
    category="apps",
    timeout=15,  # shells out to PowerShell (Get-Process) which can be slow
    risk=RISK_LOW,
)
def close_application(app_name: str) -> str:
    """Close a running application by name. Matches against actually-running
    processes (both process name and window title) rather than a fixed list."""
    exact, close_matches = _find_matching_process(app_name)

    target = exact or (close_matches[0] if len(close_matches) == 1 else None)

    if not target and len(close_matches) > 1:
        options = ", ".join(f"'{title}'" for _, title in close_matches[:5])
        return (
            f"Got a few things running that could match '{app_name}': {options}. "
            f"Which one did you mean?"
        )

    if not target:
        return f"Nothing running right now looks like '{app_name}' - might already be closed."

    proc_name, title = target
    try:
        result = subprocess.run(
            ["taskkill", "/IM", f"{proc_name}.exe", "/F"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return f"{title}, closed. Clean shutdown."
        return f"Tried to close {title} but hit static - might need a manual kill from Task Manager."
    except Exception as e:
        return f"Hit static trying to close {title}: {e}"


@register_tool(
    name="search_web",
    description="Search the web for a query and open the results in the default browser.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query, e.g. 'best noise cancelling headphones 2026'."}
        },
        "required": ["query"],
    },
    category="web",
    risk=RISK_SAFE,
)
def search_web(query: str) -> str:
    """Open a Google search for the given query in the default browser."""
    search_url = "https://www.google.com/search?q=" + query.replace(" ", "+")
    webbrowser.open(search_url)
    return f"Running a search on '{query}' now."


@register_tool(
    name="get_weather",
    description="Get the current weather for a given city. If no city is given, uses the user's default city.",
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": ["string", "null"],
                "description": "City name, e.g. 'Bengaluru'. Optional - pass null to use the default city.",
            }
        },
        "required": ["city"],
    },
    category="info",
    timeout=15,  # network call - give it more room than local operations
    risk=RISK_SAFE,
)
def get_weather(city: str = None) -> str:
    """Get current weather for a city using OpenWeatherMap's free API."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "No weather feed hooked up yet - need an API key for that."

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
            f"{city}'s reading {temp:.0f} degrees, feels like {feels:.0f}, "
            f"with {desc} out there."
        )
    except Exception as e:
        return f"Couldn't pull the weather feed for {city}: {e}"
