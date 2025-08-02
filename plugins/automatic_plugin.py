# plugins/draw_picture.py
import os
import json
import requests
import base64
import asyncio
from io import BytesIO
from dotenv import load_dotenv
import ollama
from plugin_base import ToolPlugin
import streamlit as st
from PIL import Image
import discord
from helpers import redis_client

load_dotenv()

class AutomaticPlugin(ToolPlugin):
    name = "automatic_plugin"
    usage = (
        "{\n"
        '  "function": "automatic_plugin",\n'
        '  "arguments": {"prompt": "<Text prompt for the image>"}\n'
        "}\n"
    )
    description = "Draws a picture using a prompt provided by the user using AUTOMATIC1111 API."
    settings_category = "Automatic111"
    pretty_name = "Your Image"
    required_settings = {
        "AUTOMATIC_URL": {
            "label": "AUTOMATIC URL",
            "type": "string",
            "default": "http://localhost:7860",
            "description": "The URL for the Automatic1111 API."
        },
        "AUTOMATIC_STEPS": {
            "label": "AUTOMATIC Steps",
            "type": "number",
            "default": "4",
            "description": "The number of steps for image generation."
        },
        "AUTOMATIC_CFG_SCALE": {
            "label": "AUTOMATIC CFG Scale",
            "type": "number",
            "default": "1",
            "description": "The CFG scale parameter."
        },
        "AUTOMATIC_WIDTH": {
            "label": "AUTOMATIC Width",
            "type": "number",
            "default": "896",
            "description": "The width of the generated image."
        },
        "AUTOMATIC_HEIGHT": {
            "label": "AUTOMATIC Height",
            "type": "number",
            "default": "1152",
            "description": "The height of the generated image."
        },
        "AUTOMATIC_SAMPLER": {
            "label": "AUTOMATIC Sampler",
            "type": "string",
            "default": "DPM++ 2M",
            "description": "The sampler name for image generation."
        },
        "AUTOMATIC_SCHEDULER": {
            "label": "AUTOMATIC Scheduler",
            "type": "string",
            "default": "Simple",
            "description": "The scheduler to use for image generation."
        }
    }
    waiting_prompt_template = "Write a fun, casual message telling {mention} you’re drawing their masterpiece now! Only output that message."
    platforms = ["discord", "webui"]

    @staticmethod
    def _generate_image(prompt: str) -> bytes:
        settings = redis_client.hgetall("plugin_settings:Automatic")
        AUTOMATIC_URL = settings.get("AUTOMATIC_URL") or os.getenv("AUTOMATIC_URL")
        if not AUTOMATIC_URL:
            raise Exception("AUTOMATIC_URL is not set.")
        endpoint = f"{AUTOMATIC_URL.rstrip('/')}/sdapi/v1/txt2img"

        steps = int(settings.get("AUTOMATIC_STEPS", 4))
        cfg_scale = float(settings.get("AUTOMATIC_CFG_SCALE", 1))
        width = int(settings.get("AUTOMATIC_WIDTH", 896))
        height = int(settings.get("AUTOMATIC_HEIGHT", 1152))
        sampler_name = settings.get("AUTOMATIC_SAMPLER", "DPM++ 2M")
        scheduler = settings.get("AUTOMATIC_SCHEDULER", "Simple")

        payload = {
            "prompt": prompt,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "width": width,
            "height": height,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
        }

        response = requests.post(endpoint, json=payload)
        if response.status_code == 200:
            result = response.json()
            if "images" in result and result["images"]:
                return base64.b64decode(result["images"][0])
            raise Exception("No image returned from AUTOMATIC1111 API.")
        raise Exception(f"Image generation failed ({response.status_code}): {response.text}")

    async def _respond_to_image(self, prompt_text, image_bytes, ollama_client):
        safe_prompt = prompt_text[:300].strip()
        system_msg = f'The user has just been shown an image based on this prompt: "{safe_prompt}".'
        final_response = await ollama_client.chat(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": "Send a short, friendly response to accompany the image."}
            ]
        )
        return final_response["message"].get("content", "").strip() or "Here's your image!"

    # --- Discord Handler ---
    async def handle_discord(self, message, args, ollama_client):
        prompt_text = args.get("prompt")
        if not prompt_text:
            return "No prompt provided for Automatic111."

        try:
            async with message.channel.typing():
                image_bytes = await asyncio.to_thread(self._generate_image, prompt_text)
                image_file = discord.File(BytesIO(image_bytes), filename="generated_image.png")
                await message.channel.send(file=image_file)

                reply = await self._respond_to_image(prompt_text, image_bytes, ollama_client)

                return [
                    {
                        "type": "image",
                        "name": "generated_image.png",
                        "data": base64.b64encode(image_bytes).decode("utf-8"),
                        "mimetype": "image/png"
                    },
                    reply
                ]
        except Exception as e:
            error_msg = f"❌ Failed to generate image: {e}"
            await message.channel.send(error_msg)
            return error_msg

    # --- WebUI Handler ---
    async def handle_webui(self, args, ollama_client):
        prompt_text = args.get("prompt")
        if not prompt_text:
            return ["No prompt provided for Automatic111."]

        async def inner():
            try:
                image_bytes = await asyncio.to_thread(self._generate_image, prompt_text)
                image_data = {
                    "type": "image",
                    "name": "generated_image.png",
                    "data": base64.b64encode(image_bytes).decode("utf-8"),
                    "mimetype": "image/png"
                }
                message_text = await self._respond_to_image(prompt_text, image_bytes, ollama_client)
                return [image_data, message_text]
            except Exception as e:
                return [f"Failed to generate image: {e}"]

        try:
            return await inner()
        except RuntimeError:
            return asyncio.run(inner())

    # --- IRC Handler ---
    async def handle_irc(self, bot, channel, user, raw_message, args, ollama_client):
        await bot.privmsg(channel, f"{user}: ❌ This plugin only works in Discord or the WebUI.")

plugin = AutomaticPlugin()