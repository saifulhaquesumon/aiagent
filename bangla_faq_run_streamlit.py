import streamlit as st
import bangla_faq as bf
import textwrap

st.title("Bangla FAQ Bot Demo")

question = st.text_area("আপনার প্রশ্ন লিখুন:")
if st.button("প্রশ্ন জিজ্ঞাসা করুন"):
    if question:
        with st.spinner("প্রশ্নের উত্তর খুঁজছি..."):
            category = bf.detect_category_llm(question)
            response = bf.ask_faq_bot(question, category)
            st.write("উত্তর:", response)
    else:
        st.warning("দয়া করে একটি প্রশ্ন লিখুন।")
