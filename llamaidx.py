from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings
from defEmbedding import QwenEmbedding
# from llama_index.embeddings.openai_like import OpenAILikeEmbedding
llm = OpenAILike(
    model="qwen3-max",                     # 或 qwen-turbo / qwen-max
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-686ee9610efd4ad1b8c782f26847d060",
    is_chat_model=True,
)
# response = llm.complete("你好，你是谁？")
# print(str(response))

embed_model = QwenEmbedding()

Settings.llm = llm
Settings.embed_model = embed_model

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("清炒土豆丝怎么做？")

print(response)