import streamlit as st
from chat_analyze import analyze_chat_data

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
        analysis_results = analyze_chat_data(uploaded_file)

        # Display analysis results
        st.write("Analysis Results:")
        st.write(analysis_results)

# Run the app
if __name__ == "__main__":
    main()