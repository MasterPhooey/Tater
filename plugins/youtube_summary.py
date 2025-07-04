# plugins/youtube_summary.py
import os
import re
import time
import asyncio
import subprocess
import requests
import json
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from plugin_base import ToolPlugin
from helpers import load_image_from_url, format_irc, redis_client
import redis
import discord

load_dotenv()
assistant_avatar = load_image_from_url()

class YouTubeSummaryPlugin(ToolPlugin):
    name = "youtube_summary"
    usage = (
        '{\n'
        '  "function": "youtube_summary",\n'
        '  "arguments": {\n'
        '    "video_url": "<YouTube URL>"\n'
        '  }\n'
        '}'
    )
    description = "Summarizes a YouTube video using its transcript."
    settings_category = "YouTube Summary"
    required_settings = {
        "update_transcript_api": {
            "type": "button",
            "label": "Update YouTubeTranscriptApi",
            "description": "Manually check and install the latest version of the transcript API."
        }
    }
    waiting_prompt_template = (
        "Generate a brief message to {mention} elling them to wait a moment while I watch this boring video. Only generate the message. Do not respond to this message."
    )
    platforms = ["discord", "webui", "irc"]

    def handle_setting_button(self, key):
        if key == "update_transcript_api":
            try:
                subprocess.run(["pip", "install", "--upgrade", "youtube-transcript-api"], check=True)
                return "Successfully updated youtube-transcript-api."
            except subprocess.CalledProcessError as e:
                return f"Failed to update: {e}"
                
    @staticmethod
    def extract_video_id(youtube_url):
        parsed = urlparse(youtube_url)
        if parsed.hostname in ['www.youtube.com', 'youtube.com', 'm.youtube.com']:
            return parse_qs(parsed.query).get('v', [None])[0]
        elif parsed.hostname == 'youtu.be':
            return parsed.path.lstrip('/')
        return None

    @staticmethod
    def split_text_into_chunks(text, context_length):
        max_tokens = int(context_length * 0.8)
        words = text.split()
        chunks, chunk = [], []
        for word in words:
            chunk.append(word)
            if len(chunk) >= max_tokens:
                chunks.append(" ".join(chunk))
                chunk = []
        if chunk:
            chunks.append(" ".join(chunk))
        return chunks

    async def safe_send(self, channel, content):
        if len(content) <= 2000:
            await channel.send(content)
        else:
            for chunk in self.split_message(content, 1900):
                await channel.send(chunk)

    def get_transcript_api(self, video_id):
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return " ".join([seg["text"] for seg in transcript])
        except Exception as e:
            print(f"[YouTubeTranscriptApi error] {e}")
            return None

    async def summarize_chunks(self, chunks, ollama_client):
        partial_summaries = []
        for chunk in chunks:
            prompt = (
                "You are summarizing part of a longer YouTube video transcript. "
                "Write a brief summary of this portion using bullet points:\n\n" + chunk
            )
            resp = await ollama_client.chat(
                model=ollama_client.model,
                messages=[{"role": "system", "content": prompt}],
                stream=False,
                keep_alive=-1,
                options={"num_ctx": ollama_client.context_length}
            )
            partial_summaries.append(resp["message"]["content"].strip())

        final_prompt = (
            "This is a set of bullet point summaries from different sections of a YouTube video. "
            "Combine them into a single summary with a title and final bullet points:\n\n"
            + "\n\n".join(partial_summaries)
        )
        final = await ollama_client.chat(
            model=ollama_client.model,
            messages=[{"role": "system", "content": final_prompt}],
            stream=False,
            keep_alive=-1,
            options={"num_ctx": ollama_client.context_length}
        )
        return final["message"]["content"].strip()

    def format_article(self, article):
        return article.replace("### ", "# ")

    def split_message(self, text, chunk_size=1500):
        parts = []
        while len(text) > chunk_size:
            split = text.rfind("\n", 0, chunk_size)
            if split == -1:
                split = text.rfind(" ", 0, chunk_size)
            if split == -1:
                split = chunk_size
            parts.append(text[:split])
            text = text[split:].strip()
        parts.append(text)
        return parts

    # ---------------------------------------------------------
    # Discord handler
    # ---------------------------------------------------------
    async def async_fetch_summary(self, video_url, ollama_client):
        video_id = self.extract_video_id(video_url)
        transcript = self.get_transcript_api(video_id)
        if not transcript:
            return "Sorry, this video does not have a transcript available, or it may be restricted."
        chunks = self.split_text_into_chunks(transcript, ollama_client.context_length)
        return await self.summarize_chunks(chunks, ollama_client)

    async def handle_discord(self, message, args, ollama_client, context_length, max_response_length):
        video_url = args.get("video_url")
        if not video_url:
            return "No YouTube URL provided."

        # Resolve correct output channel
        channel = message.channel
        if not isinstance(channel, discord.DMChannel):
            try:
                response_channel_id = int(os.getenv("RESPONSE_CHANNEL_ID", self.response_channel_id))
                if message.guild:
                    fallback = message.guild.get_channel(response_channel_id)
                    if fallback:
                        channel = fallback
            except:
                pass

        summary = await self.async_fetch_summary(video_url, ollama_client)
        for part in self.split_message(self.format_article(summary), max_response_length):
            await self.safe_send(channel, part)

        return ""

    # ---------------------------------------------------------
    # WebUI handler
    # ---------------------------------------------------------
    async def handle_webui(self, args, ollama_client, context_length):
        video_url = args.get("video_url")
        if not video_url:
            return "No YouTube URL provided."

        summary = await self.async_fetch_summary(video_url, ollama_client)
        return "\n".join(self.split_message(self.format_article(summary)))

    # ---------------------------------------------------------
    # IRC handler
    # ---------------------------------------------------------
    async def handle_irc(self, bot, channel, user, raw_message, args, ollama_client):
        video_url = args.get("video_url")
        nick = user

        if not video_url:
            return f"{nick}: No YouTube URL provided."

        summary = await self.async_fetch_summary(video_url, ollama_client)
        if not summary:
            return f"{nick}: Could not generate summary."

        irc_output = format_irc(self.format_article(summary))
        for part in self.split_message(irc_output, 350):
            bot.privmsg(channel, part)
            await asyncio.sleep(0.1)

        return ""

plugin = YouTubeSummaryPlugin()