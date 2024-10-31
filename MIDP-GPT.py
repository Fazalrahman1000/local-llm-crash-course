#     await msg.update()
#     message_history.append(response)
# @cl.on_chat_start

# tion Answering Pipeline

from typing import List
import chainlit as cl
from ctransformers import AutoModelForCausalLM


def get_prompt(instruction: str, history: List[str]) -> str:
    system = "AI assistant that give a helpful answer."
    prompt = f"### System:\n{system}\n\n### User:\n"
    if len(history) > 0:
        prompt += f"This is the Conversation history {''.join(history)}. Now answer the question: "
    prompt += f"{instruction}\n\n### Response:\n"
    print(f"prompt Created: {prompt}")
    return prompt


@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    msg = cl.Message(content="")
    await msg.send()

    prompt = get_prompt(message.content, message_history)
    response = ""
    for word in llm(prompt, stream=True):
        await msg.stream_token(word)
        response += word
    await msg.update()
    message_history.append(response)


@cl.on_chat_start
def on_chat_start():
    global llm
    llm = AutoModelForCausalLM.from_pretrained(
        "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
    )
