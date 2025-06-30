# 🎓 EduGenius – AI-Powered Learning Assistant

EduGenius is a Streamlit web application designed to help students grasp difficult concepts using the power of OpenAI's large language models. With features like topic explanation, study planning, quiz generation, and text summarization, EduGenius aims to simplify learning through AI.

## 🚀 Features

- **📚 Topic Explainer**: Understand complex topics with context-aware AI explanations.
- **📅 Study Plan Generator**: Create a personalized and manageable study schedule.
- **📝 Quiz Generator**: Test your knowledge with AI-generated quizzes (coming soon).
- **📄 Text Summarizer**: Condense lengthy texts into digestible summaries (coming soon).

## 🔧 How It Works

1. User enters a topic or text.
2. App retrieves contextual information using FAISS and embeddings (if needed).
3. OpenAI API generates a relevant explanation or output based on the prompt.

## 📦 Tech Stack

- **Frontend**: Streamlit
- **LLM Backend**: OpenAI GPT (via `openai` API)
- **Embeddings (if used)**: `sentence-transformers`
- **Retrieval**: FAISS
- **Language**: Python

## 📂 Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/WakeelDev/edugenius.git
   cd edugenius
   
🔗 Live Demo
👉 Try EduGenius on Streamlit

📄 License
This project is open-source and available under the MIT License.
