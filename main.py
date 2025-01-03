from fastapi import FastAPI, Body
from pydantic import BaseModel
from backend import load_vector_store
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

app = FastAPI()

retriever = load_vector_store()
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = load_qa_chain(llm, chain_type="stuff")

class QuestionRequest(BaseModel):
    question: str

@app.post("/qa")
def post_answer(request: QuestionRequest):
    question = request.question
    docs = retriever.get_relevant_documents(question)
    answer = qa_chain.run(input_documents=docs, question=question)
    return {"question": question, "answer": answer}
