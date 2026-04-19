from typing import Any, Dict, List

from GDesigner.graph.node import Node
from GDesigner.agents.agent_registry import AgentRegistry
from GDesigner.llm.llm_registry import LLMRegistry
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry


@AgentRegistry.register("BaselineAgent")
class BaselineAgent(Node):
    def __init__(
        self,
        id: str | None = None,
        role: str | None = None,
        domain: str = "",
        llm_name: str = "",
        prompt_style: str = "cot",
    ):
        super().__init__(id, "BaselineAgent", domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.prompt_style = prompt_style
        self.role = role or prompt_style

    async def _process_inputs(
        self,
        raw_inputs: Dict[str, str],
        spatial_info: Dict[str, Dict],
        temporal_info: Dict[str, Dict],
        **kwargs,
    ) -> List[Any]:
        system_prompt = self.prompt_set.get_baseline_constraint(self.prompt_style, self.role)
        user_prompt = self.prompt_set.get_baseline_answer_prompt(raw_inputs["task"], self.prompt_style)
        return system_prompt, user_prompt

    def _execute(
        self,
        input: Dict[str, str],
        spatial_info: Dict[str, Dict],
        temporal_info: Dict[str, Dict],
        **kwargs,
    ):
        system_prompt, user_prompt = self.prompt_set.get_baseline_constraint(
            self.prompt_style, self.role
        ), self.prompt_set.get_baseline_answer_prompt(input["task"], self.prompt_style)
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = self.llm.gen(message)
        return response

    async def _async_execute(
        self,
        input: Dict[str, str],
        spatial_info: Dict[str, Dict],
        temporal_info: Dict[str, Dict],
        **kwargs,
    ):
        system_prompt, user_prompt = await self._process_inputs(input, spatial_info, temporal_info)
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = await self.llm.agen(message, temperature=getattr(self, "llm_temperature", None))
        self.record_execution(system_prompt, user_prompt, response, spatial_info, temporal_info)
        self.print_agent_io(system_prompt, user_prompt, response)
        return response
