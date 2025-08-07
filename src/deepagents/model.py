from langchain_anthropic import ChatAnthropic
from langchain_fireworks import ChatFireworks


def get_default_model():
    return ChatFireworks(model="accounts/fireworks/models/kimi-k2-instruct", temperature=0.1, max_tokens=64000, max_retries=3)
