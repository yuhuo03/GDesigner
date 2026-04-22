from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt

from GDesigner.llm.config import get_openai_api_key, get_openai_base_url
from GDesigner.llm.format import Message
from GDesigner.llm.llm import LLM
from GDesigner.llm.llm_registry import LLMRegistry
from GDesigner.llm.openai_compatible import chat_completion, run_chat_completion_sync


OPENAI_API_KEYS = ['']
BASE_URL = ''


@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
async def achat(
    model: str,
    msg: List[Message],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    num_comps: Optional[int] = None,):
    return await chat_completion(
        base_url=get_openai_base_url(),
        api_key=get_openai_api_key(),
        model=model,
        messages=msg,
        max_tokens=max_tokens,
        temperature=temperature,
        num_comps=num_comps,
    )


@LLMRegistry.register('GPTChat')
class GPTChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return await achat(self.model_name,messages,max_tokens=max_tokens,temperature=temperature,num_comps=num_comps)
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        return run_chat_completion_sync(self.agen(messages, max_tokens, temperature, num_comps))
