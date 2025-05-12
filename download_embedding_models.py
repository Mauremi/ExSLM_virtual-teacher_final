from sentence_transformers import SentenceTransformer

embedding_model_names = ['ibm-granite/granite-embedding-278m-multilingual']
for embedding_model_name in embedding_model_names:
    embedding_model = SentenceTransformer(embedding_model_name, device="cuda", trust_remote_code=True)
    print('successfully downloaded', embedding_model_name)