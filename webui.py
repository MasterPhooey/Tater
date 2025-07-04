# webui.py
import streamlit as st
import redis
import os
import time
import json
import re
import asyncio
import dotenv
import logging
import base64
import requests
import importlib
import threading
from datetime import datetime
from PIL import Image
from io import BytesIO
from plugin_registry import plugin_registry
from rss import RSSManager
from platform_registry import platform_registry
from helpers import (
    OllamaClientWrapper,
    load_image_from_url,
    run_async,
    set_main_loop,
    parse_function_json,
    send_waiting_message
)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)-20s %(message)s"
)

logging.getLogger("discord.http").setLevel(logging.WARNING)
logging.getLogger("irc3").setLevel(logging.WARNING)

dotenv.load_dotenv()

@st.cache_resource(show_spinner=False)
def _start_rss(_ollama_client):
    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        rss_manager = RSSManager(ollama_client=_ollama_client)
        loop.run_until_complete(rss_manager.poll_feeds())
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t

@st.cache_resource(show_spinner=False)
def _start_platform(key):
    stop_flag = st.session_state.setdefault("platform_stop_flags", {}).get(key)
    thread = st.session_state.setdefault("platform_threads", {}).get(key)

    # 🛑 Don't start again if already running
    if thread and thread.is_alive():
        return thread, stop_flag

    stop_flag = threading.Event()

    def runner():
        try:
            module = importlib.import_module(f"platforms.{key}")
            if hasattr(module, "run"):
                module.run(stop_event=stop_flag)
            else:
                print(f"⚠️ No run(stop_event) in platforms.{key}")
        except Exception as e:
            print(f"❌ Error in platform {key}: {e}")

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()

    # Save for future shutdown/restart
    st.session_state["platform_threads"][key] = thread
    st.session_state["platform_stop_flags"][key] = stop_flag

    return thread, stop_flag

# Redis configuration for the web UI (using a separate DB)
redis_host = os.getenv('REDIS_HOST', '127.0.0.1')
redis_port = int(os.getenv('REDIS_PORT', 6379))
redis_client = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)

# Setup Ollama client for web UI using our wrapper.
ollama_host = os.getenv('OLLAMA_HOST', '127.0.0.1')
ollama_port = int(os.getenv('OLLAMA_PORT', 11434))
ollama_client = OllamaClientWrapper(host=f'http://{ollama_host}:{ollama_port}')

# Set the main event loop used for run_async.
main_loop = asyncio.get_event_loop()
set_main_loop(main_loop)

st.set_page_config(
    page_title="Tater Chat",
    page_icon=":material/tooltip_2:"
)

def save_message(role, username, content):
    message_data = {
        "role": role,
        "username": username,
        "content": content  # can be string or dict
    }
    redis_client.rpush("webui:chat_history", json.dumps(message_data))
    redis_client.ltrim("webui:chat_history", -20, -1)

def load_chat_history():
    history = redis_client.lrange("webui:chat_history", 0, -1)
    return [json.loads(msg) for msg in history]

def clear_chat_history():
    # Clear persisted history
    redis_client.delete("webui:chat_history")
    # Clear in-memory session list
    st.session_state.pop("chat_messages", None)

assistant_avatar = load_image_from_url()

# ----------------- SETTINGS HELPER FUNCTIONS -----------------
def get_chat_settings():
    settings = redis_client.hgetall("chat_settings")
    return {
        "username": settings.get("username", "User"),
        "avatar": settings.get("avatar", None)
    }

def save_chat_settings(username, avatar=None):
    mapping = {"username": username}
    if avatar is not None:
        mapping["avatar"] = avatar
    redis_client.hset("chat_settings", mapping=mapping)

def load_avatar_image(avatar_b64):
    try:
        avatar_bytes = base64.b64decode(avatar_b64)
        return Image.open(BytesIO(avatar_bytes))
    except Exception as e:
        return None

def get_plugin_enabled(plugin_name):
    # Try to get the state from Redis; default to False (disabled) if not set.
    enabled = redis_client.hget("plugin_enabled", plugin_name)
    if enabled is None:
        return False
    return enabled.lower() == "true"

def set_plugin_enabled(plugin_name, enabled):
    redis_client.hset("plugin_enabled", plugin_name, "true" if enabled else "false")

def render_plugin_controls(plugin_name):
    current_state = get_plugin_enabled(plugin_name)
    toggle_state = st.toggle(plugin_name, value=current_state, key=f"plugin_toggle_{plugin_name}")
    set_plugin_enabled(plugin_name, toggle_state)

def render_platform_controls(platform, redis_client):
    category     = platform["label"]
    key          = platform["key"]
    required     = platform["required"]
    short_name   = category.replace(" Settings", "").strip()
    state_key    = f"{key}_running"
    cooldown_key = f"tater:cooldown:{key}"
    cooldown_secs = 10

    # read current on/off from Redis
    is_running = (redis_client.get(state_key) == "true")
    emoji      = "🟢" if is_running else "🔴"

    # force‐off gadget for cooldown toggle feedback
    force_off_key = f"{category}_toggle_force_off"
    if st.session_state.get(force_off_key):
        del st.session_state[force_off_key]
        is_enabled = False
    else:
        is_enabled = st.toggle(f"{emoji} Enable {short_name}",
                               value=is_running,
                               key=f"{category}_toggle")

    # --- TURNING ON ---
    if is_enabled and not is_running:
        # cooldown check
        last = redis_client.get(cooldown_key)
        now  = time.time()
        if last and now - float(last) < cooldown_secs:
            remaining = int(cooldown_secs - (now - float(last)))
            st.warning(f"⏳ Wait {remaining}s before restarting {short_name}.")
            st.session_state[force_off_key] = True
            st.rerun()

        # actually start it
        _start_platform(key)
        redis_client.set(state_key, "true")
        st.success(f"{short_name} started.")

    # --- TURNING OFF ---
    elif not is_enabled and is_running:
        # grab the cached thread & flag, shut it down
        _, stop_flag = _start_platform(key)
        stop_flag.set()

        # clear the cache so next enable will spawn anew
        _start_platform.clear()

        redis_client.set(state_key, "false")
        redis_client.set(cooldown_key, str(time.time()))
        st.success(f"{short_name} stopped.")

    # --- SETTINGS FORM ---
    redis_key = f"{key}_settings"
    current_settings = redis_client.hgetall(redis_key)
    new_settings = {}
    for setting_key, setting in required.items():
        new_settings[setting_key] = st.text_input(
            setting["label"],
            value=current_settings.get(setting_key, setting.get("default", "")),
            help=setting.get("description", ""),
            key=f"{category}_{setting_key}"
        )
    if st.button(f"Save {short_name} Settings", key=f"save_{category}_unique"):
        redis_client.hset(redis_key, mapping=new_settings)
        st.success(f"{short_name} settings saved.")

# ----------------- PLUGIN SETTINGS -----------------
def get_plugin_settings(category):
    key = f"plugin_settings:{category}"
    return redis_client.hgetall(key)

def save_plugin_settings(category, settings_dict):
    key = f"plugin_settings:{category}"
    redis_client.hset(key, mapping=settings_dict)

# Updated: Gather distinct plugin settings categories from the plugin registry,
# only including plugins that are enabled.
plugin_categories = {}

for plugin in plugin_registry.values():
    if not get_plugin_enabled(plugin.name):
        continue

    cat = getattr(plugin, "settings_category", None)
    settings = getattr(plugin, "required_settings", None)

    if not cat or not settings:
        continue  # Skip plugins with no settings UI

    if cat not in plugin_categories:
        plugin_categories[cat] = {}

    plugin_categories[cat].update(settings)

# Display an expander for each plugin settings category.
for category, settings in sorted(plugin_categories.items()):
    with st.sidebar.expander(category, expanded=False):
        st.subheader(f"{category} Settings")
        current_settings = get_plugin_settings(category)
        new_settings = {}

        for key, info in settings.items():
            input_type    = info.get("type", "text")
            default_value = current_settings.get(key, info.get("default", ""))

            # --- BUTTONS go first ---
            if input_type == "button":
                if st.button(info["label"], key=f"{category}_{key}_button"):
                    plugin_obj = next(
                        (p for p in plugin_registry.values()
                         if getattr(p, "settings_category", "") == category),
                        None
                    )
                    if plugin_obj and hasattr(plugin_obj, "handle_setting_button"):
                        result = plugin_obj.handle_setting_button(key)
                        if result:
                            st.success(result)
                # optionally show description under the button
                if info.get("description"):
                    st.caption(info["description"])
                continue  # skip saving a value for buttons

            # PASSWORD
            if input_type == "password":
                new_value = st.text_input(
                    info.get("label", key),
                    value=default_value,
                    help=info.get("description", ""),
                    type="password"
                )

            # FILE
            elif input_type == "file":
                uploaded_file = st.file_uploader(
                    info.get("label", key),
                    type=["json"],
                    key=f"{category}_{key}"
                )
                if uploaded_file is not None:
                    try:
                        file_content = uploaded_file.read().decode("utf-8")
                        json.loads(file_content)  # validate
                        new_value = file_content
                    except Exception as e:
                        st.error(f"Error in uploaded file for {key}: {e}")
                        new_value = default_value
                else:
                    new_value = default_value

            # TEXT (fallback)
            else:
                new_value = st.text_input(
                    info.get("label", key),
                    value=default_value,
                    help=info.get("description", "")
                )

            new_settings[key] = new_value

        if st.button(f"Save {category} Settings", key=f"save_{category}_unique"):
            save_plugin_settings(category, new_settings)
            st.success(f"{category} settings saved.")

# ----------------- SYSTEM PROMPT -----------------
def build_system_prompt():
    now = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

    base_prompt = (
        "You are Tater Totterson, an AI assistant with access to various tools and plugins.\n\n"
        "When a user requests one of these actions, reply ONLY with a JSON object in one of the following formats (and nothing else):\n\n"
    )

    tool_instructions = "\n\n".join(
        f"Tool: {plugin.name}\n"
        f"Description: {getattr(plugin, 'description', 'No description provided.')}\n"
        f"{plugin.usage}"
        for plugin in plugin_registry.values()
        if ("webui" in plugin.platforms or "both" in plugin.platforms) and get_plugin_enabled(plugin.name)
    )

    behavior_guard = (
        "Only use a tool if the user's most recent message clearly asks you to perform an action — like:\n"
        "'generate', 'summarize', 'download', 'search', etc.\n"
        "Do not call tools in response to casual remarks, praise, or jokes like 'thanks', 'nice job', or 'wow!'.\n"
        "If the user is asking a general question (e.g., 'are you good at music?'), reply normally — do not use a tool.\n"
        "Do not simulate or pretend to use a tool. Only include tool results when one was actually triggered.\n"
        "Avoid copying your own past responses or following patterns from earlier messages just because they included tools or emojis.\n"
        "Focus on clear user intent, not style imitation. If in doubt, respond simply.\n\n"
    )
    
    return (
        f"Current Date and Time is: {now}\n\n"
        f"{base_prompt}\n\n"
        f"{tool_instructions}\n\n"
        f"{behavior_guard}"
        "If no function is needed, reply normally."
    )

# ----------------- PROCESSING FUNCTIONS -----------------
async def process_message(user_name, message_content):
    final_system_prompt = build_system_prompt()

    # Load last 20 (including the one just saved)
    history = load_chat_history()[-20:]

    messages_list = [{"role": "system", "content": final_system_prompt}]
    for msg in history:
        # no more prefixing!
        messages_list.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    response = await ollama_client.chat(
        model=ollama_client.model,
        messages=messages_list,
        stream=False,
        keep_alive=-1,
        options={"num_ctx": ollama_client.context_length}
    )
    return response["message"]["content"].strip()

async def process_function_call(response_json, user_question=""):
    func = response_json.get("function")
    args = response_json.get("arguments", {})
    from plugin_registry import plugin_registry

    if func in plugin_registry:
        plugin = plugin_registry[func]

        # 🌟 Show waiting message if defined
        if hasattr(plugin, "waiting_prompt_template"):
            wait_msg = plugin.waiting_prompt_template.format(mention="User")
            run_async(send_waiting_message(
                ollama_client,
                prompt_text=wait_msg,
                save_callback=lambda t: save_message("assistant", "assistant", t),
                send_callback=lambda t: st.chat_message("assistant", avatar=assistant_avatar).write(t)
            ))

        # 🌟 NEW: Show spinner during plugin execution
        with st.spinner("Tater is thinking..."):
            result = await plugin.handle_webui(args, ollama_client, ollama_client.context_length)
        return result
    else:
        return "Received an unknown function call."

# ------------------ SIDEBAR EXPANDERS ------------------
st.sidebar.markdown("---")

with st.sidebar.expander("Plugin Settings", expanded=False):
    st.subheader("Enable/Disable Plugins")
    for plugin_name in sorted(plugin_registry.keys(), key=lambda n: n.lower()):
        render_plugin_controls(plugin_name)

for platform in sorted(platform_registry, key=lambda p: p["category"].lower()):
    label = platform["label"]
    with st.sidebar.expander(label, expanded=False):
        render_platform_controls(platform, redis_client)

with st.sidebar.expander("Chat Settings", expanded=False):
    st.subheader("User Settings")
    current_chat = get_chat_settings()
    username = st.text_input("USERNAME", value=current_chat["username"])
    uploaded_avatar = st.file_uploader("Upload your avatar", type=["png", "jpg", "jpeg"], key="avatar_uploader")
    if uploaded_avatar is not None:
        avatar_bytes = uploaded_avatar.read()
        avatar_b64 = base64.b64encode(avatar_bytes).decode("utf-8")
        save_chat_settings(username, avatar_b64)
    else:
        save_chat_settings(username)
    if st.button("Clear Chat History", key="clear_history"):
        clear_chat_history()
        st.success("Chat history cleared.")

# One-time file upload handler
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = f"sidebar_uploader_{int(time.time())}"
uploaded_files = st.sidebar.file_uploader(
    "Attach torrent or image", 
    accept_multiple_files=True, 
    key=st.session_state["uploader_key"]
)

# ------------------ RSS ------------------
_start_rss(ollama_client)

# ------------------ PLATFORM MANAGEMENT ------------------
for platform in platform_registry:
    key = platform["key"]  # e.g. irc_platform
    state_key = f"{key}_running"

    # Check Redis to determine if this platform should be running
    platform_should_run = redis_client.get(state_key) == "true"

    if platform_should_run:
        _start_platform(key)
        st.success(f"{platform['category']} auto-connected.")

# ------------------ Chat ------------------
st.title("Tater Chat Web UI")

chat_settings = get_chat_settings()
avatar_b64  = chat_settings.get("avatar")
user_avatar = load_avatar_image(avatar_b64) if avatar_b64 else None

# Initialize in‐memory chat list once
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = load_chat_history().copy()

# Render all messages from session state
for msg in st.session_state.chat_messages:
    role    = msg["role"]
    avatar  = user_avatar if role == "user" else assistant_avatar
    content = msg["content"]

    with st.chat_message(role, avatar=avatar):
        if isinstance(content, dict) and content.get("type") == "image":
            data = base64.b64decode(content["data"])
            st.image(Image.open(BytesIO(data)), caption=content.get("name", ""))
        elif isinstance(content, dict) and content.get("type") == "audio":
            data = base64.b64decode(content["data"])
            st.audio(data, format=content.get("mimetype", "audio/mpeg"))
        else:
            st.write(content)

# Handle attachments first (if any)
if uploaded_files:
    allowed_extensions = [".torrent", ".png", ".jpg", ".jpeg", ".gif"]
    for uf in uploaded_files:
        if not any(uf.name.lower().endswith(ext) for ext in allowed_extensions):
            err = run_async(
                ollama_client.chat(
                    model=ollama_client.model,
                    messages=[{"role":"system",
                               "content":"Only torrent and image files are allowed."}],
                    stream=False,
                    keep_alive=-1,
                    options={"num_ctx": ollama_client.context_length}
                )
            )
            st.chat_message("assistant", avatar=assistant_avatar).write(err)
            st.stop()

    torrent_file = next((f for f in uploaded_files if f.name.lower().endswith(".torrent")), None)
    image_file   = next((f for f in uploaded_files if f.name.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))), None)

    if torrent_file:
        # user message
        st.chat_message("user", avatar=user_avatar or "🦖").write(f"[Torrent: {torrent_file.name}]")
        save_message("user", chat_settings["username"], f"[Torrent: {torrent_file.name}]")
        st.session_state.chat_messages.append({
            "role": "user", "username": chat_settings["username"],
            "content": f"[Torrent: {torrent_file.name}]"
        })

        # waiting + plugin
        with st.spinner("Checking torrent..."):
            run_async(send_waiting_message(
                ollama_client=ollama_client,
                prompt_text="Please wait while I check this torrent.",
                save_callback=lambda t: save_message("assistant","assistant",t),
                send_callback=lambda t: st.chat_message("assistant",avatar=assistant_avatar).write(t)
            ))
            plugin = plugin_registry.get("premiumize_torrent")
            result = run_async(plugin.process_torrent_web(torrent_file.read(), torrent_file.name)) \
                     if plugin else "Error: no premiumize plugin."
        st.chat_message("assistant", avatar=assistant_avatar).write(result)
        save_message("assistant","assistant",result)
        st.session_state.chat_messages.append({
            "role": "assistant", "username": "assistant", "content": result
        })

    elif image_file:
        # user message
        st.chat_message("user", avatar=user_avatar or "🦖").write(f"[Image: {image_file.name}]")
        save_message("user", chat_settings["username"], f"[Image: {image_file.name}]")
        st.session_state.chat_messages.append({
            "role": "user", "username": chat_settings["username"],
            "content": f"[Image: {image_file.name}]"
        })

        # waiting + plugin
        with st.spinner("Analyzing image..."):
            run_async(send_waiting_message(
                ollama_client=ollama_client,
                prompt_text="Please wait while I analyze this image.",
                save_callback=lambda t: save_message("assistant","assistant",t),
                send_callback=lambda t: st.chat_message("assistant",avatar=assistant_avatar).write(t)
            ))
            plugin = plugin_registry.get("vision_describer")
            result = run_async(plugin.process_image_web(image_file.read(), image_file.name)) \
                     if plugin else "Error: no vision plugin."
        st.chat_message("assistant", avatar=assistant_avatar).write(result)
        save_message("assistant","assistant",result)
        st.session_state.chat_messages.append({
            "role": "assistant", "username": "assistant", "content": result
        })

    # reset uploader so sidebar clears
    st.session_state.pop("uploader_key", None)

# Handle plain‐text user input
elif user_input := st.chat_input("Chat with Tater…"):
    uname = chat_settings["username"]

    # Persist the user’s message to Redis
    save_message("user", uname, user_input)

    # Refresh in‐memory history from Redis
    st.session_state.chat_messages = load_chat_history().copy()

    # Display the user’s message
    st.chat_message("user", avatar=user_avatar or "🦖").write(user_input)

    # Get the assistant’s response
    with st.spinner("Tater is thinking..."):
        response_text = run_async(process_message(uname, user_input))

    # Check for a function call
    func_call = parse_function_json(response_text)
    if func_call:
        func_result = run_async(process_function_call(func_call, user_input))
        responses = func_result if isinstance(func_result, list) else [func_result]
    else:
        responses = [response_text]

    # Render & persist each assistant response
    for item in responses:
        with st.chat_message("assistant", avatar=assistant_avatar):
            if isinstance(item, dict) and item.get("type") == "image":
                data = base64.b64decode(item["data"])
                st.image(Image.open(BytesIO(data)), caption=item.get("name", ""))
            elif isinstance(item, dict) and item.get("type") == "audio":
                data = base64.b64decode(item["data"])
                st.audio(data, format=item.get("mimetype", "audio/mpeg"))
            else:
                st.write(item)

        save_message("assistant", "assistant", item)

    # refresh in‐memory history again so chat_messages is fully up to date
    st.session_state.chat_messages = load_chat_history().copy()
