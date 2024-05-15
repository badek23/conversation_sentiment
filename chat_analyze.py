import pandas as pd
from io import StringIO

# Function to read the chat data
def read_chat_data(chat_data):
    stringio = StringIO(chat_data.getvalue().decode('utf-8'))
    data = stringio.read()
    data = data.split('\n')
    data = pd.DataFrame(data)
    return data

# Function to perform analysis on the chat data
def analyze_chat_data(chat_data):
    # Read the uploaded CSV file
    result = read_chat_data(chat_data)
    return result