import streamlit as st
from data_preprocessing import analyze_chat_data
from data_preprocessing import split_messages_into_users
import data_exploration as de
import nltk
nltk.download('stopwords')

# Main function to create the Streamlit app
def main():
    # Set title and description
    st.title("Chat Analysis App")
    st.write("Upload your chat CSV file to get insights!")

    # File uploader widget
    uploaded_file = st.file_uploader("Upload CSV file", type=["txt"])

     # Provide link to WhatsApp documentation
    if uploaded_file is None:
        st.markdown("If you're not sure how to export your chat history from WhatsApp, you can refer to the [WhatsApp documentation](https://faq.whatsapp.com/android/chats/how-to-save-your-chat-history/?lang=en) for instructions.")

    if uploaded_file is not None:

        # Perform analysis on the chat data
        texts_dataframe = analyze_chat_data(uploaded_file)
        user1_messages, user2_messages = split_messages_into_users(texts_dataframe)

        # Divide into two columns
        st.header("Analysis Results")
        col1, col2 = st.columns(2)

        # Display the name of each user
        col1.subheader(f"{user1_messages['sender'].unique()[0]}")
        col2.subheader(f"{user2_messages['sender'].unique()[0]}")

        # Display the number of messages for each user
        col1.write(f"Number of messages: {de.number_of_messages(user1_messages)}")
        col2.write(f"Number of messages: {de.number_of_messages(user2_messages)}")

        # Display the favorite emoji for each user
        col1.write(f"Favorite emoji: {de.fav_emoji(user1_messages)}")
        col2.write(f"Favorite emoji: {de.fav_emoji(user2_messages)}")

        # Display the average number of words per message for each user
        col1.write(f"Average number of words per message: {de.avr_num_of_words(user1_messages):.1f}")
        col2.write(f"Average number of words per message: {de.avr_num_of_words(user2_messages):.1f}")

        # Display the length of the longest message for each user
        col1.write(f"Length of longest message: {de.longest_message(user1_messages)}")
        col2.write(f"Length of longest message: {de.longest_message(user2_messages)}")

        # Display the hour with the most messages for each user
        col1.write(f"Hour with most messages: {de.hour_with_most_messages(user1_messages)}")
        col2.write(f"Hour with most messages: {de.hour_with_most_messages(user2_messages)}")

        # Display the day with the most messages for each user
        col1.write(f"Day with most messages: {de.day_with_most_messages(user1_messages)}")
        col2.write(f"Day with most messages: {de.day_with_most_messages(user2_messages)}")

        # Display the number of unique words for each user
        col1.write(f"Number of unique words: {de.number_of_unique_words(user1_messages)}")
        col2.write(f"Number of unique words: {de.number_of_unique_words(user2_messages)}")

        # Display the top words for each user
        col1.write(f"Top words: {', '.join(de.top_words(user1_messages))}")
        col2.write(f"Top words: {', '.join(de.top_words(user2_messages))}")

        st.plotly_chart(de.messages_per_day(texts_dataframe))
        st.plotly_chart(de.messages_per_hour(texts_dataframe))
        st.plotly_chart(de.messages_per_day_of_week(texts_dataframe))

# Run the app
if __name__ == "__main__":
    main()