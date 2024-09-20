import streamlit as st
import PyPDF2
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import nltk
import re

# NLTK indirmeleri
import nltk
nltk.download('punkt')
nltk.download('nonbreaking_prefixes')


# Başlık
st.title("Türkçe RAG Uygulaması")

# PDF Yükleme
uploaded_file = st.file_uploader("Bir PDF dosyası yükleyin", type=["pdf"])

if uploaded_file is not None:
    # PDF'den Metin Çıkarma
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()

    # Metni Temizleme
    text = text.replace('\n', ' ').replace('\r', '')
    text = re.sub(' +', ' ', text)

    # Metni Cümlelere Bölme (Regex kullanarak)
    sentences = re.split(r'(?<=[.!?]) +', text)

    # Embedding Modeli Yükleme
    embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedder = SentenceTransformer(embedding_model_name)

    # Cümle Embedding'leri Oluşturma
    embeddings = embedder.encode(sentences)

    # FAISS İndeksi Oluşturma
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    st.write("PDF içeriği başarıyla işlendi.")

    # Soru Alma
    question = st.text_input("Sorunuzu girin")

    if question:
        # Soru Embedding'i Oluşturma
        question_embedding = embedder.encode([question])

        # Benzer Cümleleri Arama
        k = 5  # En benzer 5 cümleyi getir
        distances, indices = index.search(np.array(question_embedding), k)
        retrieved_sentences = [sentences[idx] for idx in indices[0]]

        # Bağlam Oluşturma
        context = " ".join(retrieved_sentences)

        # QA Modelini Yükleme
        qa_model_name = "dbmdz/bert-base-turkish-cased"
        tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
        qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

        # Cevabı Alma
        qa_input = {
            'question': question,
            'context': context
        }
        result = qa_pipeline(qa_input)

        st.write("**Cevap:**")
        st.write(result['answer'])


#streamlit run turkishRAG.py