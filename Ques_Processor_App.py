import json
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field


# Load environment variables from .env file
load_dotenv()
llm = ChatOpenAI(model="gpt-4o")

# Define the prompt template and model schema
class Person(BaseModel):
    Type: str = Field(description='type of question, is it QnA or HumanHandoff')
    Question: str = Field(description='question itself')
    Answer: str = Field(description='one liner answer')

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Extract information from the following phrase. Ensure that all fields are filled and not empty.\n Formatting Instructions: {format_instructions}"),
        ("user", "Question: {question}")
    ]
)
parser = JsonOutputParser(pydantic_object=Person)
chain = prompt_template | llm | parser

def process_question(question):
    response = chain.invoke(
        {
            "question": question,
            "format_instructions": parser.get_format_instructions()
        }
    )
    return response
    
# Streamlit app
st.title("Question Processor App")

uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file is not None:
    # Read the file
    content = uploaded_file.read().decode("utf-8")
    questions = content.splitlines()

    st.write("Processing questions...")

    results = []
    for question in questions:
        result = process_question(question)
        results.append(result)

    st.write("Results:")
    for result in results:
        st.json(result)
