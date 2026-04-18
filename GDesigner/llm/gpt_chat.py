import aiohttp
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Dict, Any
from dotenv import load_dotenv
import os

from GDesigner.llm.format import Message
from GDesigner.llm.price import cost_count
from GDesigner.llm.llm import LLM
from GDesigner.llm.llm_registry import LLMRegistry


OPENAI_API_KEYS = ['']
BASE_URL = ''

load_dotenv()
MINE_BASE_URL = os.getenv('BASE_URL')
MINE_API_KEYS = os.getenv('API_KEY')


@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
async def achat(
    model: str,
    msg: List[Dict],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    num_comps: Optional[int] = None,):
    base = MINE_BASE_URL.rstrip('/')
    if base.endswith('/v1'):
        request_url = base + "/chat/completions"
    else:
        request_url = base + "/v1/chat/completions"
    authorization_key = MINE_API_KEYS
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + authorization_key
    }
    data = {
        "model": model,
        "messages": msg,
    }
    if max_tokens is not None:
        data["max_tokens"] = max_tokens
    if temperature is not None:
        data["temperature"] = temperature
    if num_comps is not None:
        data["n"] = num_comps
    async with aiohttp.ClientSession() as session:
        async with session.post(request_url, headers=headers, json=data) as response:
            response_data = await response.json()
            if 'error' in response_data:
                raise Exception(f"API error {response.status}: {response_data['error']}")
            if num_comps and num_comps > 1:
                content = [choice['message']['content'] for choice in response_data['choices']]
                completion_text = "".join(content)
            else:
                content = response_data['choices'][0]['message']['content']
                completion_text = content
            prompt_text = "".join(item.get('content', '') for item in msg)
            cost_count(prompt_text, completion_text, model)
            return content


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
        pass
