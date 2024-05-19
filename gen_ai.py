import streamlit as st
from openai import OpenAI
import time

def get_message_history(chat_dataframe, num_messages):
    message_history = ""
    last_100_messages = chat_dataframe.tail(num_messages)
    for index, row in last_100_messages.iterrows():
        message_history += f"{row['sender']}: {row['message']}\n"
    return message_history


def syllabus_request(user, sentiment, context):
    messages = [
        {
            "role": "system",
            "content": f"You are texting ",
        },
        {
            "role": "user",
            "content": f"""write the next text that {user} should write next based on the context of the conversation below. The reply should be in the same tone and context
            use a {sentiment} sentiment. Match the writting style of the conversation. Make it one short sentence. Only write the next message based on the context below:
            {context}""",
        },
    ]
    client = OpenAI()
    response = client.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages, temperature=0.3, max_tokens=2048
    )
    return response.choices[0].message.content

def generative_ai(chat_dataframe):
    st.header("Generative AI")
    st.write("Generate your next message using AI.")

    with st.expander("OpenAI API Configuration"):
        st.write("To use this feature, configure your OpenAI API settings below.")
        st.write(
            "Don't have an API key? Visit [OpenAI](https://beta.openai.com/signup/) to get one."
        )
        api_key = st.text_input("Enter your OpenAI API key")
        # Validation check for API key
        if st.button("Submit"):
            if not api_key:
                st.error("Please enter your OpenAI API key.")
            else:
                client = OpenAI(
                    api_key=api_key,  # this is also the default, it can be omitted
                    )
                st.success("API key set successfully!")

    st.write("Select the user to generate a message for:")
    reply_user = st.selectbox("User", chat_dataframe['sender'].unique(), key="ru")

    st.write("Select the sentiment of the message:")
    sentiment = st.selectbox("Sentiment", ["Positive", "Neutral", "Negative"])

    if st.button("Generate"):
        if not api_key:
            st.error("Please enter your OpenAI API key.")
        else:
            with st.spinner('Generating message...'):
                time.sleep(1)
                st.write(f"Based on the conversation, the next message for {reply_user} should be:")
                st.write(syllabus_request(reply_user, sentiment, get_message_history(chat_dataframe, 50)))