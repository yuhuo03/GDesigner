from typing import Optional
from class_registry import ClassRegistry

from GDesigner.llm.config import VLLM_BACKEND, get_llm_backend, resolve_runtime_model_name
from GDesigner.llm.llm import LLM


class LLMRegistry:
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)
    
    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, model_name: Optional[str] = None) -> LLM:
        if model_name == 'mock':
            model = cls.registry.get(model_name)
        else:
            model_name = resolve_runtime_model_name(model_name)

            if get_llm_backend() == VLLM_BACKEND:
                if 'VLLMChat' not in cls.registry.keys():
                    import GDesigner.llm.vllm_chat  # noqa: F401
                model = cls.registry.get('VLLMChat', model_name)
            else:
                if 'GPTChat' not in cls.registry.keys():
                    import GDesigner.llm.gpt_chat  # noqa: F401
                model = cls.registry.get('GPTChat', model_name)

        return model
