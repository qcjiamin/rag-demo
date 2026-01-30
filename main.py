import os

from llama_index.readers.dashscope.base import DashScopeParse
from llama_index.readers.dashscope.utils import ResultType

# 设置业务空间 ID 将决定文档解析结果在”创建知识库“步骤中上传到哪个业务空间
# os.environ['DASHSCOPE_WORKSPACE_ID'] = ""
os.environ['DASHSCOPE_API_KEY'] = "sk-686ee9610efd4ad1b8c782f26847d060"

# 第一种方式：使用文档解析器解析一个或多个文件
file = [
    'files/HS5用户手册1.pdf',
    'files/HS5用户手册2.pdf',
    'files/HS5用户手册3.pdf',
    # 需要解析的文件，支持pdf,doc,docx
]
# 解析文件
parse = DashScopeParse(result_type=ResultType.DASHSCOPE_DOCMIND)
documents = parse.load_data(file_path=file)

# 第二种方式：使用文档解析器解析一个文件夹内指定类型的文件
# from llama_index.core import SimpleDirectoryReader
# parse = DashScopeParse(result_type=ResultType.DASHSCOPE_DOCMIND)
# # 定义不同文档类型的解析器
# file_extractor = {".pdf": parse, '.doc': parse, '.docx': parse}
# # 读取文件夹，提取和解析文件信息
# documents = SimpleDirectoryReader(
#     "your_folder", file_extractor=file_extractor
# ).load_data(num_workers=1)

from llama_index.indices.managed.dashscope import DashScopeCloudIndex
# create a new index
index = DashScopeCloudIndex.from_documents(
    documents,
    "hs5_index",
    verbose=True,
)

# index = DashScopeCloudIndex("hs5_index")