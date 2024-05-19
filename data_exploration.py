import pandas as pd
import plotly.express as px
import emoji
import streamlit as st

def number_of_messages(chat_dataframe):
    return len(chat_dataframe)

def get_emojis(text):
    emojis = emoji.distinct_emoji_list(text)
    return emojis

def fav_emoji(chat_dataframe):
     chat_dataframe = chat_dataframe.copy()
     chat_dataframe['emoji'] = chat_dataframe['message'].apply(get_emojis)
     user_emojis = sum(chat_dataframe['emoji'].tolist(), [])
     user_emojis = pd.Series(user_emojis).value_counts()
     if user_emojis.empty:
         return 'No emojis found'
     if user_emojis.idxmax() == '':
         return 'No emojis found'
     #if the most used emoji has been used less than 4 times, return 'Uses few emojis'
     if user_emojis.max() < 2:
         return 'Uses few emojis'
     return f'{user_emojis.idxmax()} x {user_emojis.max()}'

def avr_num_of_words(chat_dataframe):
    return chat_dataframe['message'].str.split().apply(len).mean()

def longest_message(chat_dataframe):
    if chat_dataframe['message'].str.len().max() > 450:
        return '400+'
    return chat_dataframe['message'].str.len().max()

def hour_with_most_messages(chat_dataframe):
    return chat_dataframe['hour'].mode()[0]

def day_with_most_messages(chat_dataframe):
    return chat_dataframe['day_of_week'].mode()[0]

def messages_per_day(chat_dataframe):
    fig = px.bar(chat_dataframe['date'].dt.date.value_counts(), title='Number of Messages per Day')
    fig.update_layout(xaxis_title='Date', yaxis_title='Number of messages')
    return fig

def messages_per_hour(chat_dataframe):
    fig = px.bar(chat_dataframe['hour'].value_counts(), title='Number of Messages per Hour')
    fig.update_layout(xaxis_title='Hour', yaxis_title='Number of messages')
    return fig

def messages_per_day_of_week(chat_dataframe):
    fig = px.bar(chat_dataframe['day_of_week'].value_counts(), title='Number of Messages per Day of the Week')
    fig.update_layout(xaxis_title='Day of the Week', yaxis_title='Number of messages')
    fig.update_xaxes(categoryorder='array', categoryarray=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    return fig

def number_of_unique_words(chat_dataframe):
    unique_words = set()
    for tokens in chat_dataframe['tokens']:
        unique_words.update(tokens)
    return len(unique_words)

def top_words(chat_dataframe):
    words = sum(chat_dataframe['tokens'].tolist(), [])
    words = pd.Series(words).value_counts()
    return words.index[:10].tolist()

def column_content(user_messages, user_name):
    st.subheader(user_name)
    st.write(f"Number of messages: {number_of_messages(user_messages)}")
    st.write(f"Favorite emoji: {fav_emoji(user_messages)}")
    st.write(f"Average number of words per message: {avr_num_of_words(user_messages):.1f}")
    st.write(f"Length of longest message: {longest_message(user_messages)}")
    st.write(f"Hour with most messages: {hour_with_most_messages(user_messages)}")
    st.write(f"Day with most messages: {day_with_most_messages(user_messages)}")
    st.write(f"Number of unique words: {number_of_unique_words(user_messages)}")
    st.write(f"Top words: {', '.join(top_words(user_messages))}")

def data_analysis(chat_dataframe, user1_messages, user2_messages):

    st.header("Analysis Results")
    col1, col2 = st.columns(2)

    with col1:
        column_content(user1_messages, user1_messages['sender'].iloc[0])

    with col2:
        column_content(user2_messages, user2_messages['sender'].iloc[0])

    st.plotly_chart(messages_per_day(chat_dataframe))
    st.plotly_chart(messages_per_hour(chat_dataframe))
    st.plotly_chart(messages_per_day_of_week(chat_dataframe))