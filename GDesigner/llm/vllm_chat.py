from typing import List, Optional, Union

from tenacity import retry, stop_after_attempt, wait_random_exponential

from GDesigner.llm.config import get_vllm_api_key, get_vllm_base_url, get_vllm_extra_body, get_vllm_model
from GDesigner.llm.format import Message
from GDesigner.llm.llm import LLM
from GDesigner.llm.llm_registry import LLMRegistry
from GDesigner.llm.openai_compatible import chat_completion, run_chat_completion_sync


@LLMRegistry.register("VLLMChat")
class VLLMChat(LLM):
    def __init__(self, model_name: str):
        self.model_name = model_name or get_vllm_model()

    @retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
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
        return await chat_completion(
            base_url=get_vllm_base_url(),
            api_key=get_vllm_api_key(),
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            num_comps=num_comps,
            extra_body=get_vllm_extra_body(),
            zero_cost=True,
        )

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        return run_chat_completion_sync(self.agen(messages, max_tokens, temperature, num_comps))
