import re

ACTION_KEYWORDS = [
    "close", "end", "terminate", "shut", "kill", "quit", "stop", "dismiss", "exit", "clear", "clean", "wipe", "reload", "restart", "reopen", "refresh",
    "open", "launch", "start", "boot", "run", "initiate", "fire up", "bring up",
    "close", "end", "terminate", "shut", "shut down", "kill", "quit", "exit", "stop", "halt", "dismiss", "cut off", "force stop", "log out",
    "reload", "restart", "refresh", "reopen", "reset",
    "clear", "clean", "wipe", "erase", "flush", "empty", "purge",
    "hide", "show", "minimize", "maximize", "enable", "disable", "uninstall"
]
ACTION_KEYWORDS += [
    "execute", "fire", "trigger", "activate", "invoke", "spin up", "pop up",
    "bring online", "run up", "call", "deploy", "boot up", "launch into",
    "wrap up", "freeze", "disconnect", "lock", "log off", "mute", "pause",
    "resume", "halt process", "shut off", "wind down", "turn off", "turn on"
]





# ➋ All known application names (single- and multi-word)

APP_KEYWORDS = [
    "chrome", "firefox", "brave", "opera", "safari", "tor browser",
    "vs code", "visual studio code", "pycharm", "sublime text", "notepad++", "notepad", "android studio","calendar",
    "spotify", "youtube", "vlc", "apple music", "mx player", "prime video", "netflix", "hotstar",
    "whatsapp", "telegram", "discord", "gmail", "outlook", "teams", "zoom", "skype", "signal", "camera",
    "ms word", "excel", "powerpoint", "onenote", "google docs", "google sheets", "todoist", "notion",
    "task manager", "file explorer", "settings", "control panel", "command prompt", "terminal", "system monitor",
    "gta 5", "minecraft", "valorant", "pubg", "free fire", "fortnite", "apex legends", "activity monitor", "calculator", 
    "github desktop", "docker", "postman", "figma", "firebase console", "aws console", "cloudflare dashboard"
]
APP_KEYWORDS += [
    "obs studio", "davinci resolve", "after effects", "photoshop", "illustrator",
    "canva", "krita", "notion", "trello", "asana", "todoist", "google keep",
    "edge", "vivaldi", "anki", "gimp", "audacity", "filmora", "steam", "epic games",
    "battle.net", "rockstar launcher", "ubisoft connect", "riot client",
    "cmd", "powershell", "system settings", "notification center",
    "taskbar", "volume mixer", "network settings", "bluetooth settings",
    "nvidia control panel", "amd software", "intel graphics command center"
]





OPEN_ALIASES  = {"open", "launch", "start", "boot", "run", "initiate",
                 "fire up", "bring up", "reopen", "reload", "restart", "refresh",
                 "ignite", "boot up", "pop open", "crank up", "spin up","turn on", "activate", "access", "fire", "pull up",
                 "jump into", "load", "kickstart", "unpause", "get into",
                 "spin on", "boot into", "dive into", "bring online", "enable now",
                 "call up", "wake", "rev up", "switch on", "plug in",
                 "engage", "unfold", "light up", "deploy", "trigger",
                 "roll out", "power on", "flip on", "wake up", "hook up",
                 "show", "display", "reveal", "uncover", "expose", "present",
                 "enable", "minimize", "maximize"}
CLOSE_ALIASES = {
                 "close", "end", "terminate", "shut", "shut down", "kill","turn off", "deactivate", "abort", "break off", "pull the plug",
                 "power down", "nuke", "put to sleep", "exit out", "quit now",
                 "wrap", "freeze", "terminate session", "shutdown now", "disconnect now",
                 "mute", "blank out", "knock off", "log off", "cut out",
                 "quit", "exit", "stop", "halt", "dismiss", "cut off","shutdown", "power off", "switch off", "disable now", "turn off",
                 "cut", "disconnect", "drop", "terminate now", "cease",
                 "halt now", "cancel", "wrap up", "wind down", "kill now",
                 "force stop", "log out", "disable", "uninstall"
                 }

# ➌ Build the normalization map from ACTION_KEYWORDS
ACTION_MAP = {}
for verb in ACTION_KEYWORDS:
    low = verb.lower()
    if low in OPEN_ALIASES:
        ACTION_MAP[low] = "open"
    elif low in CLOSE_ALIASES:
        ACTION_MAP[low] = "close"
    else:
        # any other verb defaults to “open” (or handle specially)
        ACTION_MAP[low] = "open"



def normalize_command(input_text: str) -> list[str]:
    # Fix 1: Clean punctuation
    input_text = re.sub(r'([,;:.])', r' ', input_text)

    # Fix 2: Conjunction skip list
    SKIP_WORDS = {"and", "then", "or", "also"}

    # Sort apps by length desc for greedy match
    apps = sorted(APP_KEYWORDS, key=len, reverse=True)

    # Build regex for action keywords
    all_keywords = list(ACTION_MAP.keys())
    actions_pattern = '|'.join(sorted(map(re.escape, all_keywords), key=len, reverse=True))
    command_regex = re.compile(rf'\b({actions_pattern})\b', flags=re.IGNORECASE)

    tokens = input_text.strip().split()
    result = []
    i = 0
    current_action = None

    while i < len(tokens):
        word = tokens[i].lower()

        if word in SKIP_WORDS:
            i += 1
            continue

        if word in ACTION_MAP:
            current_action = ACTION_MAP[word]
            i += 1
            continue

        matched = False
        for app in apps:
            app_tokens = app.split()
            if tokens[i:i+len(app_tokens)] == app_tokens:
                if current_action:
                    result.append(f"{current_action} {app}")
                i += len(app_tokens)
                matched = True
                break

        if not matched:
            if current_action:
                result.append(f"{current_action} {tokens[i]}")
            i += 1

    return result




if __name__ == "__main__":
    query = "terminate vs code notepad open brave discord"
    queries = [
    "terminate vs code then close visual studio code open chrome",
    "shut down android studio,and reopen pycharm; kill notepad++ then start sublime text",
    "launch terminal then launch activity monitor and kill terminal",
    "quit notepad exit notepad open edge",
    "open google docs sheets then close google sheets",
    "  eXIt   Excel    then    MAXIMIZE   Calendar    MINIMIZE calendar ",
    "start notepad++ stop notepad++ reboot notepad",
    "fire up battlefield then kill spotify",
    "launch vs code,and close vs code.or open brave;kill discord",
    "open gmail outlook teams",
    "run cmd explorer close cmd",
    "launch cloudflare dashboard and shut down dashboard",
    "start spotify then stop spotify restart firefox then terminate vs code and open android studio",
    "invoke zoom then dismiss zoom and invoke slack",
    "open chrome then close",
]

    # print(normalize_command(query))
    # # → ['terminate vs code', 'terminate notepad', 'open brave', 'open discord']


    for q in queries:
        print(f"User query: {q}")
        print(f"splits: {normalize_command(q)}\n")


