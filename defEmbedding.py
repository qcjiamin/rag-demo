import dashscope
from llama_index.core.embeddings import BaseEmbedding
from typing import List

dashscope.api_key = "sk-686ee9610efd4ad1b8c782f26847d060"


class QwenEmbedding(BaseEmbedding):
    def _get_query_embedding(self, query: str) -> List[float]:
        resp = dashscope.TextEmbedding.call(
            model="text-embedding-v2",
            input=query
        )
        return resp["output"]["embeddings"][0]["embedding"]

    def _get_text_embedding(self, text: str) -> List[float]:
        resp = dashscope.TextEmbedding.call(
            model="text-embedding-v2",
            input=text
        )
        return resp["output"]["embeddings"][0]["embedding"]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)
