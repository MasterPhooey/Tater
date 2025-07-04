# plugins/unwatch_feed.py
import os
import logging
import redis
import asyncio
from dotenv import load_dotenv
from plugin_base import ToolPlugin
import streamlit as st
from PIL import Image
from io import BytesIO
import requests
import redis
from helpers import load_image_from_url, send_waiting_message

load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

assistant_avatar = load_image_from_url()  # Uses default URL from helpers.py

# Create a Redis client (adjust DB if needed)
redis_host = os.getenv('REDIS_HOST', '127.0.0.1')
redis_port = int(os.getenv('REDIS_PORT', 6379))
redis_client = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)

class UnwatchFeedPlugin(ToolPlugin):
    name = "unwatch_feed"
    usage = (
        "{\n"
        '  "function": "unwatch_feed",\n'
        '  "arguments": {"feed_url": "<RSS feed URL>"}\n'
        "}\n"
    )
    description = "Removes an RSS feed provided by the user from the watch list."
    waiting_prompt_template = (
        "Generate a brief message to {mention} telling them to wait a moment while I remove the feed from the watch list. Only generate the message. Do not respond to this message."
    )
    platforms = ["discord", "webui", "irc"]

    # --- Discord Handler ---
    async def handle_discord(self, message, args, ollama_client, context_length, max_response_length):
        feed_url = args.get("feed_url")
        if feed_url:
            removed = redis_client.hdel("rss:feeds", feed_url)
            if removed:
                final_message = f"Stopped watching feed: {feed_url}"
            else:
                final_message = f"Feed {feed_url} was not found in the watch list."
        else:
            final_message = "No feed URL provided for unwatching."

        await message.channel.send(final_message)
        return ""

    # --- Web UI Handler ---
    async def handle_webui(self, args, ollama_client, context_length):
        feed_url = args.get("feed_url")
        if not feed_url:
            final_message = "No feed URL provided for unwatching."
            return final_message

        removed = redis_client.hdel("rss:feeds", feed_url)
        if removed:
            final_message = f"Stopped watching feed: {feed_url}"
        else:
            final_message = f"Feed {feed_url} was not found in the watch list."

        return final_message

    # --- IRC Handler ---
    async def handle_irc(self, bot, channel, user, raw_message, args, ollama_client):
        mention = user
        feed_url = args.get("feed_url")
        if not feed_url:
            msg = f"{user}: No feed URL provided for unwatching."
            await bot.privmsg(channel, msg)
            return

        removed = redis_client.hdel("rss:feeds", feed_url)
        if removed:
            msg = f"{user}: Stopped watching feed: {feed_url}"
        else:
            msg = f"{user}: Feed {feed_url} was not found in the watch list."

        await bot.privmsg(channel, msg)

# Export the plugin instance.
plugin = UnwatchFeedPlugin()