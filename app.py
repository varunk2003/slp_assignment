import streamlit as st
import summarizer
from summarizer.sbert import SBertSummarizer
from sentence_transformers import SentenceTransformer
import base64

# Load SBERT model for encoding
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to generate summary using SBERT
def generate_summary_sbert(body):
    summary = sbert_model.encode(body, convert_to_tensor=False)
    return summary[0]  # Return the first element of the summary

# Function to generate summary using SBertSummarizer
def generate_summary_sbertsummarizer(body):
    model = SBertSummarizer('paraphrase-MiniLM-L6-v2')
    summary = model(body, num_sentences=5)
    return summary

# Function to download summary as a text file
def download_summary(summary_text):
    b64 = base64.b64encode(summary_text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="summary.txt">Download Summary</a>'
    st.markdown(href, unsafe_allow_html=True)

# CSS styling for the home page
def home_css():
    css = """
    <style>
    body {
        background-color: #f0f2f6;
        font-family: Arial, sans-serif;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    .container {
        max-width: 800px;
        margin: auto;
        padding: 20px;
        text-align: center;
    }
    h1 {
        color: #333;
    }
    p {
        color: #666;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Home page
def home():
    home_css()  # Apply CSS styling
    st.title("Welcome to Text Summarization")
    st.write("""
    This is a simple Streamlit app that demonstrates text summarization using SBERT (Sentence-BERT) models.
    Choose the 'Predict' page from the sidebar to enter text and generate a summary.
    """)

# Predict page
def predict():
    st.title("Text Summarization")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ("Home", "Predict"))

    if page == "Home":
        home()
    elif page == "Predict":
        st.subheader("Generate Summary")
        option = st.radio("Choose option", ("Enter Text", "Upload Text File"))

        if option == "Enter Text":
            body = st.text_area("Enter the text to summarize:")
        elif option == "Upload Text File":
            uploaded_file = st.file_uploader("Choose a file", type=['txt'])
            if uploaded_file is not None:
                body = uploaded_file.read().decode("utf-8")
            else:
                body = ""

        if st.button("Generate Summary"):
            if body:
                summary_sbertsummarizer = generate_summary_sbertsummarizer(body)
                st.subheader("Summary (SBertSummarizer):")
                st.write(summary_sbertsummarizer)

                if st.button("Download Summary"):
                    download_summary(summary_sbertsummarizer.decode("utf-8"))
            else:
                st.warning("Please enter some text to summarize.")

# Main function
def main():
    predict()

if __name__ == "__main__":
    main()