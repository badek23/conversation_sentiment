import streamlit as st
import data_preprocessing as dp
import data_exploration as de
import sent_analysis as sa
import gen_ai as ga

st.set_page_config(page_title="Pythonic Love", page_icon="💬")

# Main function to create the Streamlit app
def main():
    # Set title and description
    st.title("Pythonic Love  💖🐍")
    st.write("""
    Welcome to **Pythonic Love** – the ultimate app where romance meets code! 🥰✨
    
    Are you curious if there's a spark flying in your WhatsApp conversations? Or maybe you just want to see who sends the most emojis? Whatever your reason, you've come to the right place!
    
    Using the power of Python and our secret love-detection algorithm, **Pythonic Love** analyzes your chat history to uncover hidden sentiments, patterns, and insights that could reveal if there’s more than just friendly banter happening. 

    So, grab your chat history, upload it, and let’s see if your conversations are coded in love! 💌💻
    """)
    st.caption("Note: This app is for educational purposes only and does not store any data.")

    # File uploader widget
    uploaded_file = st.file_uploader("Upload ZIP or TXT file", type=["zip", "txt"])

    if uploaded_file is not None:

        # Perform analysis on the chat data
        texts_dataframe = dp.analyze_chat_data(uploaded_file)
        user1_messages, user2_messages = dp.split_messages_into_users(texts_dataframe)

        tab1, tab2, tab3 = st.tabs(["Who writes the most?", "Is he/she into you?", "What should I write next?"])

        with tab1:
            de.data_analysis(texts_dataframe, user1_messages, user2_messages)
        
        with tab2:
            sa.sentiment_analysis(texts_dataframe)

        with tab3:
            ga.generative_ai(texts_dataframe)
    else:
        st.markdown("""_P.S. If you're not sure how to export your chat history from WhatsApp, don't worry! [Click here](https://faq.whatsapp.com/android/chats/how-to-save-your-chat-history/?lang=en) for a handy guide._""")

# Run the app
if __name__ == "__main__":
    main()