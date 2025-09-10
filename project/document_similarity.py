from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# embedding model on the system 
embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Virat Kohli – Known as the Run Machine Kohli is one of the greatest modern-day batsmen. His aggressive style and consistency make him the backbone of Indian cricket.",

    "MS Dhoni – The Captain Cool led India to World Cup victories in 2007 (T20) and 2011 (ODI). Famous for his calmness under pressure, Dhoni is also one of the best finishers in cricket history.",

    "Sachin Tendulkar – Called the God of Cricket he holds the record for 100 international centuries. Tendulkar’s career inspired generations of cricketers across the world.",

    "AB de Villiers – Nicknamed Mr. 360 he could hit the ball to any part of the ground. His innovative batting style made him a nightmare for bowlers.",

    "Shane Warne – One of the greatest leg-spinners, Warne revived the art of spin bowling. His skill, charisma, and match-winning spells left a lasting legacy in world cricket."
]

query = 'tell me about virat kohli' 

# embedding created for the data and for the user query 
doc_embedding = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)
 
# both of them should be the list 
scores = cosine_similarity([query_embedding], doc_embedding)[0] 
index , score = sorted(list(enumerate(scores)), key= lambda x:x[1])[-1]

print("Query:", query)
print("Best Match Index:", index)
print("Similarity Score:", score)
print("Matched Document:", documents[index])
