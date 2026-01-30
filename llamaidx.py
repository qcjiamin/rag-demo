from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, PromptTemplate
from llama_index.llms.openai_like import OpenAILike
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

query_engine = index.as_query_engine(text_qa_template=PromptTemplate("""
    你是一个只能助手，请优先参考提供的上下文进行回答
    如果上下文不足，可以使用你自身知识补充，但要说明哪些来自上下文，哪些是常识。
    上下文：{context_str}
    问题：{query_str}
    回答：
"""))
response = query_engine.query("如何关闭车窗？")

print(response)