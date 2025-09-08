from langchain_openai import OpenAI 
from dotenv import load_dotenv 

load_dotenv() 
# create object 
llm = OpenAI(model = 'gpt-3.5-turbo-instruct')
result  = llm.invoke("what is th capital of india")
print (result)