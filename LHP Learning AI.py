import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import HumanMessage
from PyPDF2 import PdfReader
import os

# API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Giao diện
st.title("LHP Learning AI")
st.write("Please enter code here:")

# Hàm chuyển số nhúng
def embed_question(question: str, embeddings_model):
    """Chuyển đổi câu hỏi thành vector nhúng."""
    embedding = embeddings_model.embed_query(question)
    return embedding

# Tạo mô hình nhúng
embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

if "response" not in st.session_state:
    st.session_state["response"] = ""

# đọc nội dung từ file PDF
def read_pdf(file):
    """Đọc nội dung từ file PDF và trả về văn bản."""
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Nhập câu hỏi từ người dùng
user_question = st.text_input("Type your question here:")

# Cho phép tải lên file PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Xử lý khi người dùng tải lên file PDF
if uploaded_file is not None:
    # Đọc nội dung file PDF
    file_text = read_pdf(uploaded_file)
    st.write("File content extracted successfully. You can now ask questions.")
else:
    file_text = ""

if st.button("Submit"):
    if user_question.strip():
        # Chuyển câu hỏi thành mã nhúng
        embedded_question = embed_question(user_question, embeddings_model)

        # mô hình gpt
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0.7,
            max_tokens=1000,
            model_name="gpt-3.5-turbo",
        )

        # Kết hợp câu hỏi người dùng và nội dung file PDF
        if file_text:
            user_question = f"Based on the document content: {file_text} \n{user_question}"

        input_message = HumanMessage(content=user_question)

        # Gửi câu hỏi tới GPT và nhận phản hồi
        response = llm([input_message])
        st.session_state["response"] = response.content

# Hiển thị kết quả trả lời
if st.session_state["response"]:
    st.subheader("Answer:")
    st.write(st.session_state["response"])