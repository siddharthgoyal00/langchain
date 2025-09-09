from langchain_huggingface import HuggingFaceEmbeddings
# embedding model on the system 
embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
text = "delhi is the capital of inda" 
vector = embedding.embed_query(text)
print (str(vector))
