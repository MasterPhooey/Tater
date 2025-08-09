# webui.py
import gradio as gr
import redis
import os
import time
import json
import base64
import importlib
import threading
import asyncio
from datetime import datetime
from io import BytesIO
from PIL import Image

from plugin_registry import plugin_registry
from platform_registry import platform_registry
from helpers import (
    LLMClientWrapper,
    run_async,
    set_main_loop,
    parse_function_json,
    get_tater_name,
)

# Redis configuration
redis_host = os.getenv("REDIS_HOST", "127.0.0.1")
redis_port = int(os.getenv("REDIS_PORT", 6379))
redis_client = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)

# LLM client setup
llm_host = os.getenv("LLM_HOST", "127.0.0.1")
llm_port = int(os.getenv("LLM_PORT", 11434))
llm_client = LLMClientWrapper(host=f"http://{llm_host}:{llm_port}")

# Configure event loop for run_async helper
main_loop = asyncio.get_event_loop()
set_main_loop(main_loop)

first_name, last_name = get_tater_name()

# -------------------- Chat history helpers --------------------
def save_message(role, username, content):
    key = "webui:chat_history"
    message_data = {"role": role, "username": username, "content": content}
    redis_client.rpush(key, json.dumps(message_data))
    try:
        max_store = int(redis_client.get("tater:max_store") or 20)
    except (ValueError, TypeError):
        max_store = 20
    if max_store > 0:
        redis_client.ltrim(key, -max_store, -1)

def load_chat_history():
    history = redis_client.lrange("webui:chat_history", 0, -1)
    return [json.loads(msg) for msg in history]

def clear_chat_history():
    redis_client.delete("webui:chat_history")

# -------------------- Avatar helpers --------------------
def load_default_tater_avatar():
    return Image.open("images/tater.png")

def get_tater_avatar():
    avatar_b64 = redis_client.get("tater:avatar")
    if avatar_b64:
        try:
            avatar_bytes = base64.b64decode(avatar_b64)
            return Image.open(BytesIO(avatar_bytes))
        except Exception:
            redis_client.delete("tater:avatar")
    return load_default_tater_avatar()

# -------------------- Chat settings helpers --------------------
def get_chat_settings():
    settings = redis_client.hgetall("chat_settings")
    return {
        "username": settings.get("username", "User"),
        "avatar": settings.get("avatar", None),
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
    except Exception:
        redis_client.hdel("chat_settings", "avatar")
        return None

# -------------------- Plugin enable/disable --------------------
def get_plugin_enabled(plugin_name):
    enabled = redis_client.hget("plugin_enabled", plugin_name)
    if enabled is None:
        return False
    return enabled.lower() == "true"

def set_plugin_enabled(plugin_name, enabled):
    redis_client.hset("plugin_enabled", plugin_name, "true" if enabled else "false")

# -------------------- Plugin settings --------------------
def get_plugin_settings(category):
    key = f"plugin_settings:{category}"
    return redis_client.hgetall(key)

def save_plugin_settings(category, settings_dict):
    key = f"plugin_settings:{category}"
    str_settings = {k: str(v) for k, v in settings_dict.items()}
    redis_client.hset(key, mapping=str_settings)

# -------------------- Platform controls --------------------
platform_threads = {}
platform_stop_flags = {}

def start_platform(key):
    thread = platform_threads.get(key)
    stop_flag = platform_stop_flags.get(key)
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
    platform_threads[key] = thread
    platform_stop_flags[key] = stop_flag
    return thread, stop_flag

def toggle_platform(key, short_name, enabled):
    state_key = f"{key}_running"
    cooldown_key = f"tater:cooldown:{key}"
    cooldown_secs = 10
    if enabled:
        last = redis_client.get(cooldown_key)
        now = time.time()
        if last and now - float(last) < cooldown_secs:
            remaining = int(cooldown_secs - (now - float(last)))
            return f"⏳ Wait {remaining}s before restarting {short_name}."
        start_platform(key)
        redis_client.set(state_key, "true")
        return f"{short_name} started."
    else:
        _, stop_flag = start_platform(key)
        if stop_flag:
            stop_flag.set()
        redis_client.set(state_key, "false")
        redis_client.set(cooldown_key, str(time.time()))
        return f"{short_name} stopped."

def save_platform_settings(key, settings_dict):
    redis_key = f"{key}_settings"
    redis_client.hset(redis_key, mapping=settings_dict)

# -------------------- LLM prompt/processing --------------------
def build_system_prompt():
    now = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
    base_prompt = (
        f"You are {first_name} {last_name}, an AI assistant with access to various tools and plugins.\n\n"
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
        "Only call a tool if the user's latest message clearly requests an action — such as 'generate', 'summarize', or 'download'.\n"
        "Never call a tool in response to casual or friendly messages like 'thanks', 'lol', or 'cool' — reply normally instead.\n"
    )
    return (
        f"Current Date and Time is: {now}\n\n"
        f"{base_prompt}\n\n"
        f"{tool_instructions}\n\n"
        f"{behavior_guard}"
        "If no function is needed, reply normally."
    )

async def process_message(user_name, message_content):
    final_system_prompt = build_system_prompt()
    max_llm = int(redis_client.get("tater:max_llm") or 8)
    history = load_chat_history()[-max_llm:]

    messages_list = [{"role": "system", "content": final_system_prompt}]
    for msg in history:
        content = msg["content"]
        if msg["role"] == "user":
            if isinstance(content, str):
                text = content
            elif isinstance(content, dict) and content.get("type") == "image":
                text = "[Image]"
            elif isinstance(content, dict) and content.get("type") == "audio":
                text = "[Audio]"
            else:
                text = "[Unknown]"
            messages_list.append({"role": "user", "content": text})
        elif msg["role"] == "assistant":
            if isinstance(content, dict) and content.get("marker") == "plugin_call":
                plugin_call_text = json.dumps({
                    "function": content.get("plugin"),
                    "arguments": content.get("arguments", {})
                }, indent=2)
                messages_list.append({"role": "assistant", "content": plugin_call_text})
            elif isinstance(content, dict) and content.get("marker") == "plugin_response":
                continue
            else:
                if isinstance(content, str):
                    messages_list.append({"role": "assistant", "content": content})
    response = await llm_client.chat(messages_list)
    return response["message"]["content"].strip()

async def process_function_call(response_json, user_question=""):
    func = response_json.get("function")
    args = response_json.get("arguments", {})
    if func in plugin_registry:
        save_message("assistant", "assistant", {
            "marker": "plugin_call",
            "plugin": func,
            "arguments": args
        })
        plugin = plugin_registry[func]
        if hasattr(plugin, "waiting_prompt_template"):
            wait_msg = plugin.waiting_prompt_template.format(mention="User")
            wait_response = await llm_client.chat(messages=[{"role": "system", "content": wait_msg}])
            wait_text = wait_response["message"]["content"].strip()
            save_message("assistant", "assistant", {
                "marker": "plugin_response",
                "content": wait_text
            })
        result = await plugin.handle_webui(args, llm_client)
        responses = result if isinstance(result, list) else [result]
        for r in responses:
            save_message("assistant", "assistant", {
                "marker": "plugin_response",
                "content": r
            })
        return responses
    return ["Unknown function"]

# -------------------- History conversion --------------------
def load_gradio_history():
    gr_hist = []
    for msg in load_chat_history():
        role = msg["role"]
        content = msg["content"]
        while isinstance(content, dict) and content.get("marker") == "plugin_response":
            content = content.get("content")
        if isinstance(content, dict) and content.get("marker") == "plugin_call":
            continue
        if not isinstance(content, str):
            content = json.dumps(content)
        if role == "user":
            gr_hist.append([content, None])
        else:
            if gr_hist and gr_hist[-1][1] is None:
                gr_hist[-1][1] = content
            else:
                gr_hist.append([None, content])
    return gr_hist

# -------------------- Chat handler --------------------
def chat_fn(user_message, history):
    uname = get_chat_settings()["username"]
    save_message("user", uname, user_message)
    response_text = run_async(process_message(uname, user_message))
    func_call = parse_function_json(response_text)
    if func_call:
        responses = run_async(process_function_call(func_call, user_message))
        if not isinstance(responses, list):
            responses = [responses]
    else:
        responses = [response_text]
    final = "\n".join(r if isinstance(r, str) else json.dumps(r) for r in responses)
    save_message("assistant", "assistant", final)
    history = history + [(user_message, final)]
    return history, ""

# -------------------- Settings handlers --------------------
def save_chat_settings_handler(username, avatar):
    if avatar is not None:
        buffered = BytesIO()
        avatar.save(buffered, format="PNG")
        b64 = base64.b64encode(buffered.getvalue()).decode()
    else:
        b64 = None
    save_chat_settings(username, b64)
    return "Chat settings saved."

def handle_setting_button(category, key):
    plugin_obj = next(
        (p for p in plugin_registry.values() if getattr(p, "settings_category", "") == category),
        None
    )
    if plugin_obj and hasattr(plugin_obj, "handle_setting_button"):
        result = plugin_obj.handle_setting_button(key)
        return result or ""
    return ""

# -------------------- Build UI --------------------
chat_settings = get_chat_settings()
user_avatar = load_avatar_image(chat_settings["avatar"]) if chat_settings.get("avatar") else None
assistant_avatar = get_tater_avatar()

plugin_categories = {}
for plugin in plugin_registry.values():
    if not get_plugin_enabled(plugin.name):
        continue
    cat = getattr(plugin, "settings_category", None)
    settings = getattr(plugin, "required_settings", None)
    if not cat or not settings:
        continue
    if cat not in plugin_categories:
        plugin_categories[cat] = {}
    plugin_categories[cat].update(settings)

avatars = [user_avatar, assistant_avatar]

with gr.Blocks(title=f"{first_name} Chat") as demo:
    with gr.Tab("Chat"):
        chatbot = gr.Chatbot(value=load_gradio_history(), avatar_images=avatars)
        msg = gr.Textbox(label=f"Chat with {first_name}")
        send = gr.Button("Send")
        send.click(chat_fn, [msg, chatbot], [chatbot, msg])
        clear = gr.Button("Clear History")
        def clear_history():
            clear_chat_history()
            return [], ""
        clear.click(clear_history, outputs=[chatbot, msg])

    with gr.Tab("Settings"):
        with gr.Accordion("Chat Settings", open=False):
            username = gr.Textbox(label="Display Name", value=chat_settings["username"])
            avatar = gr.Image(label="Avatar", value=user_avatar)
            chat_msg = gr.Markdown("")
            save_btn = gr.Button("Save")
            save_btn.click(save_chat_settings_handler, [username, avatar], chat_msg)

        with gr.Accordion("Plugins", open=False):
            for plugin in plugin_registry.values():
                cb = gr.Checkbox(label=plugin.name, value=get_plugin_enabled(plugin.name))
                cb.change(lambda val, name=plugin.name: set_plugin_enabled(name, val), cb, None)

        with gr.Accordion("Plugin Settings", open=False):
            for category, settings in sorted(plugin_categories.items()):
                with gr.Accordion(category, open=False):
                    current = get_plugin_settings(category)
                    inputs = {}
                    messages = gr.Markdown("")
                    for key, info in settings.items():
                        input_type = info.get("type", "text")
                        default_value = current.get(key, info.get("default", ""))
                        if input_type == "password":
                            comp = gr.Textbox(label=info.get("label", key), value=default_value, type="password")
                        elif input_type == "file":
                            comp = gr.File(label=info.get("label", key), file_types=[".json"])
                        elif input_type == "select":
                            options = info.get("options", [])
                            comp = gr.Dropdown(
                                label=info.get("label", key),
                                choices=options,
                                value=default_value if default_value in options else (options[0] if options else None),
                            )
                        elif input_type == "checkbox":
                            is_checked = (
                                default_value if isinstance(default_value, bool)
                                else str(default_value).lower() in ("true", "1", "yes")
                            )
                            comp = gr.Checkbox(label=info.get("label", key), value=is_checked)
                        elif input_type == "button":
                            btn = gr.Button(info["label"])
                            btn.click(lambda cat=category, k=key: handle_setting_button(cat, k), outputs=messages)
                            if info.get("description"):
                                gr.Markdown(info["description"])
                            continue
                        else:
                            comp = gr.Textbox(label=info.get("label", key), value=default_value)
                        inputs[key] = comp
                        if info.get("description"):
                            gr.Markdown(info["description"])
                    def save_cat_handler(*vals, cat=category, keys=list(inputs.keys())):
                        save_plugin_settings(cat, dict(zip(keys, vals)))
                        return f"{cat} settings saved."
                    save = gr.Button(f"Save {category}")
                    save.click(save_cat_handler, list(inputs.values()), messages)

        with gr.Accordion("Platforms", open=False):
            for platform in platform_registry:
                category = platform["label"]
                key = platform["key"]
                required = platform["required"]
                short_name = category.replace(" Settings", "").strip()
                is_running = (redis_client.get(f"{key}_running") == "true")
                with gr.Accordion(category, open=False):
                    platform_msg = gr.Markdown("")
                    toggle = gr.Checkbox(label=f"Enable {short_name}", value=is_running)
                    inputs = {}
                    current_settings = redis_client.hgetall(f"{key}_settings")
                    for s_key, setting in required.items():
                        val = current_settings.get(s_key, setting.get("default", ""))
                        comp = gr.Textbox(label=setting["label"], value=val)
                        if setting.get("description"):
                            gr.Markdown(setting["description"])
                        inputs[s_key] = comp
                    def save_platform_handler(*vals, k=key, keys=list(inputs.keys())):
                        save_platform_settings(k, dict(zip(keys, vals)))
                        return f"{short_name} settings saved."
                    save = gr.Button(f"Save {short_name}")
                    save.click(save_platform_handler, list(inputs.values()), platform_msg)
                    toggle.change(lambda val, k=key, sn=short_name: toggle_platform(k, sn, val), toggle, platform_msg)

if __name__ == "__main__":
    demo.launch()
