from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline 

# getting the model on the system 
llm = HuggingFacePipeline.from_model_id(
    model_id='mistralai/Mistral-7B-Instruct-v0.2',
    task='text-generation',
    pipepline_kwargs=dict(
       temperature = 0.5 ,
       max_new_tokens=100
    )
)
model = ChatHuggingFace(llm=llm)
result = model.invoke("what is the capital of india")
print(result.content)