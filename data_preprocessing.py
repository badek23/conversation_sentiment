from io import StringIO
import zipfile
import pandas as pd
import datetime
import emoji
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

number_of_messages = 8000
# Function to read the chat data from txt file
# TODO - Add support for txt file
# TODO - Add error handling for invalid files
def read_chat_data(uploaded_file):

    # Check if the uploaded file is a ZIP file
    if uploaded_file.name.endswith('.zip'):
        with zipfile.ZipFile(uploaded_file, 'r') as z:
            # Read the 'chat.txt' file
            with z.open('_chat.txt') as file:
                stringio = StringIO(file.read().decode('utf-8'))
                data = stringio.read()
                data = data.split('\n')
                data = pd.DataFrame(data).iloc[-number_of_messages:]
                #data = pd.DataFrame(data)
            return data
    elif uploaded_file.name.endswith('.txt'):
        data = uploaded_file.read().decode('utf-8')
        data = data.split('\n')
        data = pd.DataFrame(data).iloc[-number_of_messages:]
        return data
    

# Delete messages that are not relevant to the chat
def clean_data(chat_dataframe):
    ltr_mark = '\u200e'
    irrelevant_strings = [f'{ltr_mark}Location', f'{ltr_mark}Voice', f'{ltr_mark}Missed', f'{ltr_mark}This message was deleted', f'{ltr_mark}imagen', f'{ltr_mark}Llamada', f'{ltr_mark}sticker', f'{ltr_mark}audio']

    # Combine all conditions in a single expression
    conditions = (
        ~chat_dataframe.iloc[:, 0].str.startswith(ltr_mark) &
        ~chat_dataframe.iloc[:, 0].apply(lambda x: any(irrelevant_str in x for irrelevant_str in irrelevant_strings))
    )

    # Filter the DataFrame based on the combined conditions
    filtered_data = chat_dataframe[conditions]

    return filtered_data.reset_index(drop=True)

# Aggregate messages that span multiple lines
def multiple_lines_to_single_line(chat_dataframe):
    # Regular expression to detect the start of a new message
    message_start_re = re.compile(r'\[(\d+/\d+/\d+, \d+:\d+:\d+)\]')

    # Aggregate messages that span multiple lines
    aggregated_messages = []
    current_message = ""

    for i in range(len(chat_dataframe)):
        line = chat_dataframe.iloc[i].values[0]
        if message_start_re.match(line):  # Detects the start of a new message
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

def determine_date_format(chat_dataframe):
    # Sample multiple dates to ensure consistency
    date_samples = chat_dataframe['date'].sample(min(10, len(chat_dataframe)))  # Sample up to 10 dates

    mmddyy_count = 0
    ddmmyy_count = 0

    for sample_date in date_samples:
        try:
            # Try to parse the date in 'mm/dd/yy, HH:MM:SS' format
            datetime.datetime.strptime(sample_date, '%m/%d/%y, %H:%M:%S')
            mmddyy_count += 1
        except ValueError:
            pass

        try:
            # Try to parse the date in 'dd/mm/yy, HH:MM:SS' format
            datetime.datetime.strptime(sample_date, '%d/%m/%y, %H:%M:%S')
            ddmmyy_count += 1
        except ValueError:
            pass

    if mmddyy_count > ddmmyy_count:
        return '%m/%d/%y, %H:%M:%S'
    elif ddmmyy_count > mmddyy_count:
        return '%d/%m/%y, %H:%M:%S'
    else:
        raise ValueError('Date format is not consistent or recognizable as "mm/dd/yy" or "dd/mm/yy"')

# Extract the date from the message
def feature_dates(chat_dataframe):
    chat_dataframe['date'] = chat_dataframe[0].str.extract(r'\[(\d+/\d+/\d+, \d+:\d+:\d+)\]')
    date_format = determine_date_format(chat_dataframe)
    chat_dataframe['date'] = pd.to_datetime(chat_dataframe['date'], format=date_format)
    chat_dataframe['day'] = chat_dataframe['date'].dt.date.astype('str')
    chat_dataframe['hour'] = chat_dataframe['date'].dt.hour
    chat_dataframe['day_of_week'] = chat_dataframe['date'].dt.day_name()
    return chat_dataframe

# Get the text and user of each message
def split_messages_and_users(chat_dataframe):
    chat_dataframe['sender'] = chat_dataframe[0].str.extract(r'\[\d+/\d+/\d+, \d+:\d+:\d+\] ([a-zA-z\s]+):.*')
    chat_dataframe['message'] = chat_dataframe[0].str.extract(r'\[\d+/\d+/\d+, \d+:\d+:\d+\] [a-zA-z\s]+:(.*)')
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
    extra_stop_words = {'im', 'u', 'get', 'dont', 'thats', 'ill', 'si', 'q', 'voy', 'pa', 'pq', 'creo', 'mas', 'asi', 'solo', 'vas', 'ahi', 'ver'}
    esp_stop_words = set(stopwords.words('spanish'))
    return [word for word in tokens if word not in stop_words and word not in extra_stop_words and word not in esp_stop_words]

# Get tokens from text
def get_tokens(chat_dataframe):
    chat_dataframe['tokens'] = chat_dataframe['message'].apply(remove_symbols)
    chat_dataframe['tokens'] = chat_dataframe['tokens'].str.lower()
    chat_dataframe['tokens'] = chat_dataframe['tokens'].apply(tokenize)
    chat_dataframe['tokens'] = chat_dataframe['tokens'].apply(remove_stopwords)
    return chat_dataframe

# Function to check for custom criteria
def check_custom_criteria(message):
    bonus = 0.0
    laughing_emojis = ['😂','🤣','😆']
    sexy_emojis = ['😏','😍','😘','😚','😙','😗','😋','😛','😜','😝','😎','😈','👅','👄']
    
    # Check for words ending with repeating letters (more than 2) (e.g., "Heyyy", "gooo")
    if re.search(r'\b\w*(\w)\1{2,}\b', message):
        bonus += 0.5  # Adjust the bonus value as needed

    # Check for more than one exclamation marks in a row
    if re.search(r'!{2,}', message):
        bonus += 0.3  # Adjust the bonus value as needed

    # Check for specific emoji (e.g., laughing emoji)
    laughing_emoji_count = sum(1 for emoji in laughing_emojis if emoji in message)
    bonus += 0.2 * laughing_emoji_count  # Increase bonus by 0.1 for each laughing emoji

    sexy_emoji_count = sum(1 for emoji in sexy_emojis if emoji in message)
    bonus += 0.5 * laughing_emoji_count  # Increase bonus by 0.1 for each laughing emoji
    
    # Check for "heyy" with increasing bonus for more "y"s
    heyy_match = re.search(r'\bhey+y\b', message, re.IGNORECASE) or \
                re.search(r'\bhola+a\b', message, re.IGNORECASE) or \
                re.search(r'\bhi+i\b', message, re.IGNORECASE) or \
                re.search(r'\bhello+o\b', message, re.IGNORECASE)
    if heyy_match:
        # Number of "y"s minus one, as the base "hey" is not included in the bonus calculation
        num_ys = len(heyy_match.group()) - 3
        bonus += 0.2 * num_ys  # Increase bonus by 0.1 for each additional "y"

    #check for "jajaja" with increasing bonus for more "ja"s
    laugh_match = re.search(r'\b[jakhis]+[jakhis]\b', message, re.IGNORECASE)
    if laugh_match:
        # Number of "ja"s minus one, as the base "ja" is not included in the bonus calculation
        num_jas = len(laugh_match.group()) - 3
        bonus += 0.2 * num_jas
    
    #check for "JAJA" with increasing bonus for more "JA"s
    allcaps_laugh_match = re.search(r'\b[JAKHS]+[JAKHS]\b', message)
    if allcaps_laugh_match:
        # Number of "JA"s minus one, as the base "JA" is not included in the bonus calculation
        num_JAs = len(allcaps_laugh_match.group()) - 3
        bonus += 0.5 * num_JAs
    
    return bonus

def sentiment_analysis(chat_dataframe):
    # Initialize the sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    # Calculate the sentiment scores for each message
    def custom_sentiment_scores(message):
        scores = sid.polarity_scores(message)
        bonus = check_custom_criteria(message)
        # Adjust the compound score with the bonus
        scores['compound'] = min(2.0, max(0, scores['compound'] + bonus))  # Ensure the compound score is within [-1.0, 1.0], plus some extra points
        return scores

    # Calculate the sentiment scores for each message
    chat_dataframe['sent_scores'] = chat_dataframe['message'].apply(custom_sentiment_scores)
    chat_dataframe['compound'] = chat_dataframe['sent_scores'].apply(lambda x: x['compound'])
    chat_dataframe['neg'] = chat_dataframe['sent_scores'].apply(lambda x: x['neg'])
    chat_dataframe['neu'] = chat_dataframe['sent_scores'].apply(lambda x: x['neu'])
    chat_dataframe['pos'] = chat_dataframe['sent_scores'].apply(lambda x: x['pos'])
    chat_dataframe['sentiment'] = chat_dataframe['compound'].apply(lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral'))

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
    # Perform sentiment analysis
    cleaned_data = sentiment_analysis(cleaned_data)
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