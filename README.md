English | [中文](README-zh.md)

This project is the **champion work** of the 2024 SJTU School of Electronic Information and Electrical Engineering Freshman Cup - Large Language Model Virtual Teacher Challenge. Our team was honored to be invited to share it at the **2025 AWS China Summit**. Based on Amazon Web Services, we implemented features like large model memory, language input/output, RAG-enhanced search, multimodal input, and a virtual avatar. The project is now open-sourced and supports even more functions! Our goal is to provide students with a self-driven, highly personalized AI learning system.

We strongly recommend using **PyCharm** to view and even run this project.

⚠️ Attention! This project relies on three AWS services: [Amazon Bedrock](https://aws.amazon.com/cn/bedrock/?refid=deb9a6fe-73e7-47f8-854a-423a366590c0), [Amazon Transcribe](https://aws.amazon.com/cn/transcribe/?nc2=type_a), and [Amazon Polly](https://aws.amazon.com/cn/polly/?nc2=type_a). Before running the project, please add your AWS access credentials to your system environment variables.

# Project Highlights

## RAG

### PDF Parsing and Structuring

We use the [unstructuredpdfpdf](https://github.com/Unstructured-IO/unstructured) library to analyze PDFs. The contents are split into segments, and each chunk is tagged with appropriate HTML labels to help the language model better understand the structure and position of the content.

For dual-column documents, we extract coordinates for each chunk and reorder the content from left to right based on the page's midline, solving the common OCR issue of disordered text in dual-column layouts. We call this method the **Element Coordinate Analysis Algorithm**.

Processed texts are saved as individual `.txt` files under `./RAG/processed_txt/`.

### Text Chunking

We use LangChain to semantically chunk the texts. The chunk size is dynamically adjusted based on semantic similarity. Chunk boundaries are marked with `bubu`, a rare string that does not occur in any text. These chunked files are stored in `.txt` format in `./RAG/chunked_txt/`.

For different text types, chunking thresholds can be adjusted in the code. This system does not distinguish between different text types by default.

### HyDE Query Rewriting

User queries are processed using the HyDE (Hypothetical Document Embeddings) query rewriting strategy.

### Retrieval

We use the `sentence-transformers` library for information retrieval and support multiple embedding models. The top-ranking results based on cosine similarity are selected and used as the context for the large language model.

Supported embedding models include:
1. [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)
2. [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)
3. [BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5)
4. [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
5. [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base)
6. [ibm-granite/granite-embedding-278m-multilingual](https://huggingface.co/ibm-granite/granite-embedding-278m-multilingual)

Users can choose the model that best suits their textbook language and device performance.

## Multimodality

### Files

Uploaded PDF files are processed the same way as textbooks. Their entire text content is directly sent to the language model for generation. We do not recommend uploading large files.

Uploaded image files (png, webp, jpeg) are decoded and passed to the model through the appropriate multimodal input interface.

Uploaded files can be managed in the settings panel. For images, only the **five most recently uploaded** ones are sent to the model.

### Voice

After switching to voice input, users can select the spoken language in the settings and input by speaking. When voice output is enabled, the model's response becomes simpler and more conversational (while remaining professional), and is read aloud after generation.

We integrated a Live2D virtual avatar, using its latest version’s built-in functions to synchronize voice and lip movement. The 2D models are stored in `./static/model/` and can be replaced freely.

## Other Features

- Supports long-term memory. Just open the settings and enter your preferences and long-term states in the global memory manager.
- Supports viewing RAG generation results, one-click copying, and inspecting generation info.

More features await your exploration! 🚀