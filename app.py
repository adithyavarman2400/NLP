import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


# Function to get the response
def getLLMResponse(form_input, email_sender, email_recipient, email_style):

    llm = ChatGroq(
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-70b-8192"
    )

    template = """
    Write an email in a {style} tone about the topic: "{email_topic}".
    Sender: {sender}
    Recipient: {recipient}

    Email Text:
    """

    prompt = PromptTemplate(
        input_variables=["style", "email_topic", "sender", "recipient"],
        template=template
    )

    response = llm.predict(prompt.format(
        email_topic=form_input,
        sender=email_sender,
        recipient=email_recipient,
        style=email_style
    ))

    return response


# Streamlit UI
st.set_page_config(page_title="Generate Emails", page_icon='📧', layout='centered')
st.header("Generate Emails 📧")

form_input = st.text_area('Enter the email topic', height=275)

col1, col2, col3 = st.columns([10, 10, 5])
with col1:
    email_sender = st.text_input('Sender Name')
with col2:
    email_recipient = st.text_input('Recipient Name')
with col3:
    email_style = st.selectbox('Writing Style', ('Formal', 'Appreciating', 'Not Satisfied', 'Neutral', 'Informal'), index=0)

submit = st.button("Generate")

if submit:
    st.write(getLLMResponse(form_input, email_sender, email_recipient, email_style))
