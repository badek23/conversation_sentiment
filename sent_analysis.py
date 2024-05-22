import pandas as pd
import streamlit as st
import time

def get_average_sentiment(chat_dataframe, user):
    user_messages = chat_dataframe[chat_dataframe['sender'] == user]
    return user_messages['compound'].mean()

def get_average_recent_sentiment(chat_dataframe, user, num_messages=10):
    user_messages = chat_dataframe[chat_dataframe['sender'] == user]
    recent_messages = user_messages.tail(num_messages)
    return recent_messages['compound'].mean()

def sentiment_difference(chat_dataframe, user1, user2):
    user1_sentiment = get_average_sentiment(chat_dataframe, user1)
    user2_sentiment = get_average_sentiment(chat_dataframe, user2)
    return user1_sentiment - user2_sentiment

def recent_sentiment_change(chat_dataframe, user):
    user_sentiment = get_average_sentiment(chat_dataframe, user)
    recent_sentiment = get_average_recent_sentiment(chat_dataframe, user)
    return recent_sentiment - user_sentiment

def usual_sentiment(chat_dataframe, user):
    sentiment = get_average_sentiment(chat_dataframe, user)
    if sentiment > 0.2:
        return f"positive"
    elif sentiment < -0.2:
        return f"negative"
    else:
        return f"neutral"

# check if one user has lately been more positive than usual
def is_user_more_positive(chat_dataframe, user):
    sentiment_change = recent_sentiment_change(chat_dataframe, user)
    sentiment = usual_sentiment(chat_dataframe, user)

    if sentiment_change > 0.2:
        return f"{user} has been more positive than usual lately."
    elif sentiment_change < -0.2:
        return f"{user} has been more negative than usual lately."
    else:
        return f"{user} has been as acting as usual, {sentiment}."

def who_is_more_postive(chat_dataframe, user1, user2):
    sentiment_diff = sentiment_difference(chat_dataframe, user1, user2)
    if sentiment_diff > 0.05:
        return f"{user1} is more into the conversation than {user2}."
    elif sentiment_diff < -0.05:
        return f"{user2} is more into the conversation than {user1}."
    else:
        return "Both of you are equally into the conversation."
    
def sentiment_analysis(chat_dataframe):
    st.header("Sentiment Analysis")
    #button to analyze sentiment
    st.write("Analyze the sentiment of the chat messages.")

    if st.button("Analyze"):
        with st.spinner('Looking at the emotions...'):
            time.sleep(1)
            st.write("Analysis complete!")
            st.write(is_user_more_positive(chat_dataframe, chat_dataframe['sender'].unique()[0]))
            st.write(is_user_more_positive(chat_dataframe, chat_dataframe['sender'].unique()[1]))
            st.write(who_is_more_postive(chat_dataframe, chat_dataframe['sender'].unique()[0], chat_dataframe['sender'].unique()[1]))