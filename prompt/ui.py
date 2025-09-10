from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from dotenv import load_dotenv
import streamlit as st 
from langchain_core.prompts import PromptTemplate,load_prompt
load_dotenv() 
st.header('reseach tool ')
llm = HuggingFaceEndpoint (
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    task = "text-generation"
 )
model = ChatHuggingFace(llm=llm)

# static prompt
"""
user_input = st.text_input('enter yout prompt') 
if st.button('summarize'):
    result = model.invoke(user_input)
    st.write(result.content)
"""

## dynamic prompt
paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template = load_prompt('template.json')

if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
        'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input
    })
    st.write(result.content)