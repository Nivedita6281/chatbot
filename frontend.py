import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from backend import load_vector_store

def main():
    st.title("Q&A Chatbot for EA_Sports🤖")
    st.write("Explore the knowledge hidden in the documents! Ask a question to start your journey!")

    # Load FAISS vector store
    vector_store = load_vector_store()

    # Set up LangChain QA chain
    retriever = vector_store.as_retriever()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = load_qa_chain(llm, chain_type="stuff")

    # Q&A Section
    question = st.text_input("Your Question:")
    if question:
        with st.spinner("Fetching the answer..."):
            # Retrieve relevant documents and get answer
            docs = retriever.get_relevant_documents(question)
            result = qa_chain.run(input_documents=docs, question=question)
            st.write("### Answer:")
            st.success(result)

if __name__ == "__main__":
    main()
