from loop import (
    APIProvider,
    sampling_loop_sync,
)
from tools import ToolResult
import base64
from datetime import datetime
from pathlib import Path
from typing import cast
from enum import StrEnum
from anthropic import APIResponse
from anthropic.types import TextBlock
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock
from anthropic.types.tool_use_block import ToolUseBlock


from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()


provider = "openai"
api_key = os.getenv("OPENAI_API_KEY")
only_n_most_recent_images = 2
messages = []
model = "omniparser + gpt-4o" #"omniparser + gpt-4o-orchestrated"
omniparser_server_url = "localhost:8000"
responses = {}
tools = {}

user_input = "help buy some milk on amazon"


def _api_response_callback(response: APIResponse[BetaMessage]):
    response_id = datetime.now().isoformat()
    responses[response_id] = response

def _tool_output_callback(tool_output: ToolResult, tool_id: str):
    tools[tool_id] = tool_output

def chatbot_output_callback(message, hide_images=False):
    def _render_message(message: str | BetaTextBlock | BetaToolUseBlock | ToolResult, hide_images=False):
        if isinstance(message, str):
            return message
        
        is_tool_result = not isinstance(message, str) and (
            isinstance(message, ToolResult)
            or message.__class__.__name__ == "ToolResult"
        )
        
        if is_tool_result:
            message = cast(ToolResult, message)
            if message.output:
                return message.output
            if message.error:
                return f"Error: {message.error}"
            if message.base64_image and not hide_images:
                return f'<img src="data:image/png;base64,{message.base64_image}">'
        
        elif isinstance(message, (BetaTextBlock, TextBlock)):
            return f"Next step Reasoning: {message.text}"
        
        elif isinstance(message, (BetaToolUseBlock, ToolUseBlock)):
            return None
        
        return message

    rendered_message = _render_message(message, hide_images)
    if rendered_message:
        messages.append({"role": "assistant", "content": rendered_message})

messages.append({"role": "user", "content": user_input})
# Process the message through sampling_loop_sync
for loop_msg in sampling_loop_sync(
                model=model,
                provider=provider,
                messages=[{"role": "user", "content": [TextBlock(type="text", text=msg["content"])]} for msg in messages],
                output_callback=chatbot_output_callback,
                tool_output_callback=_tool_output_callback,
                api_response_callback=_api_response_callback,
                api_key=api_key,
                only_n_most_recent_images=only_n_most_recent_images,
                max_tokens=16384,
                omniparser_url=omniparser_server_url,
                save_folder=str("./uploads")
            ):
    print(loop_msg)