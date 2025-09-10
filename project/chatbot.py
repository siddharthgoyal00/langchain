from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# these are langchain message like human, system and the ai messages 
# these can be dynamic as well for this we use ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv 
load_dotenv() 

llmModel = HuggingFaceEndpoint (
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    task = "text-generation"
 )
model = ChatHuggingFace(llm = llmModel)
chat_history = [
   SystemMessage(content = "you are the helpful ai assistance ")
]
while True : 
   user_input =  input ('you: ')
   chat_history.append(HumanMessage(content = user_input))
   if user_input == 'exit' : 
      break 
   result = model.invoke(chat_history)
   chat_history.append(AIMessage(content = result.content))
   print ("ai: " , result.content) 

print(chat_history)
