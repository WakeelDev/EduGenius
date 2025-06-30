import streamlit as st
import openai
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Page config
st.set_page_config(page_title="EduGenius - AI Learning Assistant", page_icon="🎓")

# App title
st.title("🎓 EduGenius - Your AI Learning Assistant")

# Sidebar: Navigation
st.sidebar.title("Navigation")
options = ["Topic Explainer", "Study Plan Generator", "Quiz Generator", "Text Summarizer"]
choice = st.sidebar.radio("Choose a feature", options)

# Sidebar: API key input
st.sidebar.title("OpenAI API Key")
api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Sample documents for FAISS context
documents = [
    "Gradient descent is an optimization algorithm used to minimize the loss function in machine learning.",
    "Newton's first law states that an object remains at rest or in uniform motion unless acted upon by an external force.",
    "Photosynthesis is the process by which plants use sunlight to produce energy in the form of glucose.",
    "Machine learning is a subset of artificial intelligence focused on training models to make predictions."
]

# Build FAISS index
@st.cache_resource
def build_faiss_index():
    model = load_embedding_model()
    doc_embeddings = np.array([model.encode(doc) for doc in documents])
    dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(doc_embeddings)
    return index, documents

# Retrieve top_k context using FAISS
def retrieve_context(query, model, index, documents, top_k=2):
    query_embedding = model.encode(query)
    distances, indices = index.search(np.array([query_embedding]), top_k)
    return " ".join([documents[i] for i in indices[0]])

# OpenAI response function
def get_openai_response(prompt, temperature=0.7, max_tokens=800):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an educational assistant that helps students learn."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"Error: {e}"

# === Feature: Topic Explainer ===
if choice == "Topic Explainer":
    st.header("📚 Topic Explainer")
    topic = st.text_input("Enter a topic you'd like to understand better:", placeholder="e.g., What is gradient descent?")
    
    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to use this feature.")
    elif topic:
        openai.api_key = api_key
        with st.spinner("Retrieving context and generating explanation..."):
            embedding_model = load_embedding_model()
            index, docs = build_faiss_index()
            context = retrieve_context(topic, embedding_model, index, docs)
            prompt = f"Here is some background information:\n{context}\n\nNow explain this topic: {topic}"
            explanation = get_openai_response(prompt)
            st.markdown(explanation)

# === Feature: Study Plan Generator ===
elif choice == "Study Plan Generator":
    st.header("📅 Study Plan Generator")
    subject = st.text_input("Enter a topic or subject you want a study plan for:", placeholder="e.g., Machine Learning")

    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to use this feature.")
    elif subject:
        openai.api_key = api_key
        with st.spinner("Generating your study plan..."):
            prompt = f"Create a detailed 7-day study plan to learn: {subject}"
            plan = get_openai_response(prompt)
            st.markdown(plan)

# === Feature: Quiz Generator ===
elif choice == "Quiz Generator":
    st.header("📝 Quiz Generator")
    quiz_topic = st.text_input("Enter a topic for the quiz:", placeholder="e.g., Photosynthesis")

    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to use this feature.")
    elif quiz_topic:
        openai.api_key = api_key
        with st.spinner("Generating quiz questions..."):
            prompt = f"Generate 5 multiple-choice questions (with 4 options and correct answers) about: {quiz_topic}"
            quiz = get_openai_response(prompt)
            st.markdown(quiz)

# === Feature: Text Summarizer ===
elif choice == "Text Summarizer":
    st.header("📄 Text Summarizer")
    long_text = st.text_area("Paste a long passage you want to summarize:", height=250)

    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to use this feature.")
    elif long_text:
        openai.api_key = api_key
        with st.spinner("Summarizing text..."):
            prompt = f"Summarize the following text in a clear and concise way:\n\n{long_text}"
            summary = get_openai_response(prompt)
            st.markdown(summary)
