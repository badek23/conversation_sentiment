import streamlit as st
from data_preprocessing import analyze_chat_data
from data_preprocessing import split_messages_into_users
import data_exploration as de
import time

def column_content(user_messages, user_name):
    st.subheader(user_name)
    st.write(f"Number of messages: {de.number_of_messages(user_messages)}")
    st.write(f"Favorite emoji: {de.fav_emoji(user_messages)}")
    st.write(f"Average number of words per message: {de.avr_num_of_words(user_messages):.1f}")
    st.write(f"Length of longest message: {de.longest_message(user_messages)}")
    st.write(f"Hour with most messages: {de.hour_with_most_messages(user_messages)}")
    st.write(f"Day with most messages: {de.day_with_most_messages(user_messages)}")
    st.write(f"Number of unique words: {de.number_of_unique_words(user_messages)}")
    st.write(f"Top words: {', '.join(de.top_words(user_messages))}")

def data_analysis(chat_dataframe, user1_messages, user2_messages):

    st.header("Analysis Results")
    col1, col2 = st.columns(2)

    with col1:
        column_content(user1_messages, user1_messages['sender'].iloc[0])

    with col2:
        column_content(user2_messages, user2_messages['sender'].iloc[0])

    st.plotly_chart(de.messages_per_day(chat_dataframe))
    st.plotly_chart(de.messages_per_hour(chat_dataframe))
    st.plotly_chart(de.messages_per_day_of_week(chat_dataframe))

def sentiment_analysis(chat_dataframe):
    st.header("Sentiment Analysis")
    #button to analyze sentiment
    st.write("Analyze the sentiment of the chat messages.")

    st.write("Select the user to analyze:")
    sent_user = st.selectbox("User", chat_dataframe['sender'].unique(), key="su")

    st.write("Select the type of sentiment analysis:")
    analysis_type = st.selectbox("Analysis type", ["Sentiment", "Emotion"])

    if st.button("Analyze"):
        with st.spinner('Looking at the emotions...'):
            time.sleep(1)
            st.write("Coming soon!")

def generative_ai(chat_dataframe):
    st.header("Generative AI")
    st.write("Generate your next message using AI.")

    st.write("Select the user to generate a message for:")
    reply_user = st.selectbox("User", chat_dataframe['sender'].unique(), key="ru")

    st.write("Select the sentiment of the message:")
    sentiment = st.selectbox("Sentiment", ["Positive", "Neutral", "Negative"])

    if st.button("Generate"):
        with st.spinner('Generating message...'):
            time.sleep(1)
            st.write("Coming soon!")


# Main function to create the Streamlit app
def main():
    # Set title and description
    st.title("Chat Analysis App")
    st.write("Export your WhatsApp chat history and analyze it with this app! Choose to not attach media and upload the ZIP file.")

    # File uploader widget
    uploaded_file = st.file_uploader("Upload ZIP file", type=["zip"])

     # Provide link to WhatsApp documentation
    if uploaded_file is None:
        st.markdown("If you're not sure how to export your chat history from WhatsApp, you can refer to the [WhatsApp documentation](https://faq.whatsapp.com/android/chats/how-to-save-your-chat-history/?lang=en) for instructions.")

    if uploaded_file is not None:

        # Perform analysis on the chat data
        texts_dataframe = analyze_chat_data(uploaded_file)
        user1_messages, user2_messages = split_messages_into_users(texts_dataframe)

        tab1, tab2, tab3 = st.tabs(["Who writes the most?", "Who is angrier?", "What should I write next?"])

        with tab1:
            data_analysis(texts_dataframe, user1_messages, user2_messages)
        
        with tab2:
            sentiment_analysis(texts_dataframe)

        with tab3:
            generative_ai(texts_dataframe)

# Run the app
if __name__ == "__main__":
    main()