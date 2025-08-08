from langchain_anthropic import ChatAnthropic
from langchain_fireworks import ChatFireworks


def get_default_model():
    return ChatFireworks(model="accounts/fireworks/models/kimi-k2-instruct", temperature=0.3, max_tokens=64000, max_retries=3)

def get_default_structuring_model():
    return ChatFireworks(model="accounts/fireworks/models/llama4-maverick-instruct-basic", temperature=0, max_tokens=128000, max_retries=3)
