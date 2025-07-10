[English](README.md) | 中文

本项目是上海交通大学2024电院新生杯-大模型虚拟教师专题赛的**冠军作品**，我们团队因此受邀参加了**2025亚马逊云科技中国峰会**进行分享。在亚马逊云资源的基础上，完成大模型记忆、语言输入输出、RAG 增强搜索、多模态输入和虚拟形象功能。现已开源，并支持更多功能！我们的宗旨是向学生提供一个由其自己主导的、十分个性化的人工智能学习系统。

我们强烈建议用 **PyCharm** 查看甚至运行本项目。

⚠️ 注意！本项目基于亚马逊云科技提供的三项服务（[Amazon Bedrock](https://aws.amazon.com/cn/bedrock/?refid=deb9a6fe-73e7-47f8-854a-423a366590c0), [Amazon Transcribe](https://aws.amazon.com/cn/transcribe/?nc2=type_a), [Amazon Polly](https://aws.amazon.com/cn/polly/?nc2=type_a)），运行前请将获得的亚马逊权限账号与密码添加到系统环境变量中。

# 项目特色

## RAG

### PDF 识别与整理

我们项目采用 [unstructured\[pdf\]](https://github.com/Unstructured-IO/unstructured) 库解析 PDF，将 PDF 内容分块处理，并将每个分块打上合适的 HTML 标签用于标识内容位置，便于大语言模型更好理解文本。

对于双栏文字，我们在提取内容后，获取分块的坐标，按照每一页的中线位置，让提取内容按照先左后右的顺序重排，避免了传统 OCR 识别双栏文字顺序混乱（双栏乱序）的问题。这一算法被我们称为**元素坐标分析算法**。

识别之后的文章被分别存储到单独的 txt 文件中，具体位于 ./RAG/processed_txt/ 中。

### 文本分块处理

我们使用 LangChain 对文本进行语义分块。通过其内置的方法按照语义相关性动态调整分块的大小。分块的界限被标记为`bubu`，经查询是所有文章中都没有出现过的单词序列。被分块标记过的文章以 txt 格式存储在 ./RAG/chunked_txt/ 中。

对于不同类型的文本，可以在代码中自行调整分块阈值。在本系统中并没有对文本类型做区分。

### HyDE 查询重写

对于用户提出的问题，我们采用 HyDE 查询重写策略对其进行处理。

### 检索

我们使用 sentence-transformer 库进行检索，接入了多种 embedding model，最后用该库的内置方法，按照余弦相似度进行排名，排名最高的几条相关信息将作为上下文提供给大模型。

Embedding model 如：

1. [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)
2. [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)
3. [BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5)
4. [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
5. [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base)
6. [ibm-granite/granite-embedding-278m-multilingual](https://huggingface.co/ibm-granite/granite-embedding-278m-multilingual)

用户可以根据教材语言和电脑配置自行选择合适的模型。

## 多模态

### 文件

通过按钮上传的 pdf 格式文件会经过与教材处理相同的方式进行文字识别，其所有文字将直接提供给大模型进行下一步生成。所以我们不建议上传大型文件。

通过按钮上传的图片文件（png, webp, jpeg）被解码并直接通过大模型请求体中的相应接口上传。

在设置中可以对已上传的文件进行管理。理论上，对于图片，我们仅取最新上传的五张图片提供给大模型。

### 语音

切换为语音输入后，用户可以在设置中选定语音的语言，并通过说话的方式输入。在设置中打开语音输出后，模型的回答将变得相对简单且口语化（但不失专业性），并在生成结束之后用语音读出。

我们接入了 live2D 虚拟形象，并用了其最新版本更新的内置函数进行语音和嘴型的同步输出。在文件夹 ./static/model/ 中存储了 2D 模型，可以自行更换。

## 其他功能

- 支持长期记忆，只需打开设置，在全局记忆管理中输入自己的偏好、长期状态即可。
- 支持浏览 RAG 生成结果，一键复制以及查看生成信息。

更多功能等待你的挖掘！🚀