import pandas as pd
from io import StringIO

number_of_messages = 500
# Function to read the chat data
def read_chat_data(uploaded_file):
    stringio = StringIO(uploaded_file.getvalue().decode('utf-8'))
    data = stringio.read()
    data = data.split('\n')
    data = pd.DataFrame(data).iloc[-number_of_messages:]
    return data

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

def multiple_lines_to_single_line(chat_dataframe):
    # Aggregate messages that span multiple lines
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

def truncate_long_messages(chat_dataframe):
    for i in range(0, len(chat_dataframe)):
        message = chat_dataframe.iloc[i].values[0]
        if len(message) > 500:
            chat_dataframe.iloc[i] = message[:500]
    return chat_dataframe

def feature_dates(chat_dataframe):
    # Extract the date from the message
    chat_dataframe['date'] = chat_dataframe[0].str.extract(r'\[(\d+/\d+/\d+, \d+:\d+:\d+)\]')
    chat_dataframe['date'] = pd.to_datetime(chat_dataframe['date'], format="%m/%d/%y, %H:%M:%S")
    chat_dataframe['day'] = chat_dataframe['date'].dt.date.astype('str')
    chat_dataframe['hour'] = chat_dataframe['date'].dt.hour
    chat_dataframe['day_of_week'] = chat_dataframe['date'].dt.day_name()
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
    return cleaned_data
