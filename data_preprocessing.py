import pandas as pd
from io import StringIO
import emoji
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

number_of_messages = 4000
# Function to read the chat data from txt file
def read_chat_data(uploaded_file):
    stringio = StringIO(uploaded_file.getvalue().decode('utf-8'))
    data = stringio.read()
    data = data.split('\n')
    data = pd.DataFrame(data).iloc[-number_of_messages:]
    return data

# Delete messages that are not relevant to the chat
def clean_data(chat_dataframe):
    ltr_mark = '\u200e'
    location_str = f'{ltr_mark}Location'
    voice_call_str = f'{ltr_mark}Voice'
    missed_call_str = f'{ltr_mark}Missed'
    deleted_str = f'{ltr_mark}This message was deleted'

    filtered_data = chat_dataframe[
        ~chat_dataframe.iloc[:, 0].str.startswith(ltr_mark) &
        ~chat_dataframe.iloc[:, 0].str.contains(location_str) &
        ~chat_dataframe.iloc[:, 0].str.contains(voice_call_str) &
        ~chat_dataframe.iloc[:, 0].str.contains(missed_call_str) &
        ~chat_dataframe.iloc[:, 0].str.contains(deleted_str)
    ]

    return filtered_data.reset_index(drop=True)

# Aggregate messages that span multiple lines
def multiple_lines_to_single_line(chat_dataframe):
    aggregated_messages = []
    current_message = ""

    for i in range(len(chat_dataframe)):
        line = chat_dataframe.iloc[i].values[0]
        if line.startswith('['):  # Detects the start of a new message
            if current_message:
                aggregated_messages.append(current_message)
            current_message = line
        else:  # Continuation of the previous message
            current_message += ' ' + line

    # Don't forget to append the last message
    if current_message:
        aggregated_messages.append(current_message)

    # Convert the list back to a DataFrame
    return pd.DataFrame(aggregated_messages, columns=[chat_dataframe.columns[0]])

# Truncate messages longer than 500 characters
def truncate_long_messages(chat_dataframe):
    for i in range(0, len(chat_dataframe)):
        message = chat_dataframe.iloc[i].values[0]
        if len(message) > 500:
            chat_dataframe.iloc[i] = message[:500]
    return chat_dataframe

# Extract the date from the message
def feature_dates(chat_dataframe):
    chat_dataframe['date'] = chat_dataframe[0].str.extract(r'\[(\d+/\d+/\d+, \d+:\d+:\d+)\]')
    chat_dataframe['date'] = pd.to_datetime(chat_dataframe['date'], format="%m/%d/%y, %H:%M:%S")
    chat_dataframe['day'] = chat_dataframe['date'].dt.date.astype('str')
    chat_dataframe['hour'] = chat_dataframe['date'].dt.hour
    chat_dataframe['day_of_week'] = chat_dataframe['date'].dt.day_name()
    return chat_dataframe

# Get the text and user of each message
def split_messages_and_users(chat_dataframe):
    chat_dataframe['sender'] = chat_dataframe[0].str.extract(r'\[\d+/\d+/\d+, \d+:\d+:\d+\] ([a-zA-z]+):.*')
    chat_dataframe['message'] = chat_dataframe[0].str.extract(r'\[\d+/\d+/\d+, \d+:\d+:\d+\] [a-zA-z]+:(.*)')
    chat_dataframe['sender'] = chat_dataframe['sender'].str.strip()
    chat_dataframe['message'] = chat_dataframe['message'].str.strip()
    return chat_dataframe

def remove_symbols(text):
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def tokenize(text):
    return nltk.word_tokenize(text)

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    extra_stop_words = {'im', 'u', 'get', 'dont', 'thats', 'ill'}
    return [word for word in tokens if word not in stop_words and word not in extra_stop_words]

# Get tokens from text
def get_tokens(chat_dataframe):
    chat_dataframe['tokens'] = chat_dataframe['message'].apply(remove_symbols)
    chat_dataframe['tokens'] = chat_dataframe['tokens'].str.lower()
    chat_dataframe['tokens'] = chat_dataframe['tokens'].apply(tokenize)
    chat_dataframe['tokens'] = chat_dataframe['tokens'].apply(remove_stopwords)
    return chat_dataframe

# Function to perform analysis on the chat data
def analyze_chat_data(uploaded_file):
    # Read the uploaded CSV file
    chat_df = read_chat_data(uploaded_file)
    # Clean the data
    cleaned_data = clean_data(chat_df)
    # Aggregate messages that span multiple lines
    cleaned_data = multiple_lines_to_single_line(cleaned_data)
    # Truncate long messages
    cleaned_data = truncate_long_messages(cleaned_data)
    # Extract date features
    cleaned_data = feature_dates(cleaned_data)
    # Split messages and users
    cleaned_data = split_messages_and_users(cleaned_data)
    # Get tokens from text
    cleaned_data = get_tokens(cleaned_data)
    #drop the columns that are not needed
    cleaned_data = cleaned_data.drop(columns=[0])

    return cleaned_data

# Get messages from each unique user
def split_messages_into_users(chat_dataframe):
    users = chat_dataframe['sender'].unique()
    user1 = users[0]
    user2 = users[1]
    user1_data = chat_dataframe[chat_dataframe['sender'] == user1]
    user2_data = chat_dataframe[chat_dataframe['sender'] == user2]
    return user1_data, user2_data