import streamlit as st
import uuid
import json, os
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
import g4f
from langchain.llms.base import LLM
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document

chat_history_file = "chat_history.json"


def process_pdfs():
    # Load PDF files using PyPDFLoader
    pdf_loader_1 = PyPDFLoader("docs/RIL_IAR_2024.pdf")
    pdf_loader_2 = PyPDFLoader("docs/Nagarro-Q3-2024-EN.pdf")
    pdf_loader_3 = PyPDFLoader("docs/tsla-20240723-gen.pdf")

    # Load documents with metadata (file names, pages, etc.)
    documents_1 = pdf_loader_1.load()
    documents_2 = pdf_loader_2.load()
    documents_3 = pdf_loader_3.load()

    # Adding custom metadata (you can add more as needed)
    for doc in documents_1:
        doc.metadata = {"source": "path_to_pdf_file_1.pdf", "file_name": "file_1", "page_count": len(doc.page_content.split("\n")),"users_access":["A"]}

    for doc in documents_2:
        doc.metadata = {"source": "path_to_pdf_file_2.pdf", "file_name": "file_2", "page_count": len(doc.page_content.split("\n")),"users_access":["B"]}

    for doc in documents_3:
        doc.metadata = {"source": "path_to_pdf_file_2.pdf", "file_name": "file_2", "page_count": len(doc.page_content.split("\n")),"users_access":["A","C"]}

    # Now documents_1 and documents_2 contain the PDFs along with metadata
    return documents_1 + documents_2 + documents_3

# Custom LLM Wrapper for g4f
class G4FLLM(LLM):
    def _call(self, prompt: str, stop=None):
        try:
            response = g4f.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "system", "content": "You are a helpful Q&A asistant on companies earnings. Respond in always polite English. If you dont have supporting data to respond, reply as current user don't have access to asked company's info else say out of scope query."},{"role": "user", "content": prompt}]
            )
            if isinstance(response, str):
                return response
            elif isinstance(response, dict) and "choices" in response:
                return response["choices"][0]["message"]["content"]
            else:
                raise ValueError("Unexpected response format from g4f.")
        except Exception as e:
            return f"Error calling GPT-4: {str(e)}"

    @property
    def _llm_type(self) -> str:
        return "g4f"


# Memory and Storage Initialization
if "chat_data" not in st.session_state:
    st.session_state["chat_data"] = {}

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# Save chat history to a JSON file
def save_chat_history(chat_history):
    with open(chat_history_file, "w") as file:
        json.dump(chat_history, file, indent=4)

def get_mock_documents():
    # docs = process_pdfs()
    # return docs
    return [
        Document(
            page_content="Nagarro had a strong Q4 with revenue increasing by 20%.",
            metadata={"source": "pdf1", "user_access":["A"]}),
        Document(
            page_content="Reliance's expenses rose sharply in Q4, impacting profit margins. And earning is down by 10%",
            metadata={"source": "pdf2", "user_access":["B"]}),
        Document(
            page_content="Tesla had a successful product launch, generating a lot of buzz in the market. Tesla had very expensive quarter becoming loss of 500 Crore.",
            metadata={"source": "pdf3", "user_access":["A","C"]}),
        
    ]


# Load existing chat history
def load_chat_history():
    if os.path.exists(chat_history_file):
        with open(chat_history_file, "r") as file:
            try:
                chat_history = json.load(file)
                if isinstance(chat_history, list):  # Check if it's a list
                    return chat_history
                else:
                    return []  # Return an empty list if data is not in expected format
            except json.JSONDecodeError:
                return []  # Return empty list if file content is invalid
    return []  # If file doesn't exist, return an empty list


def initialize_chain(documents, user_id):
    # Setup your embedding model
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Extract texts and metadata from documents
    doc_texts = [doc.page_content for doc in documents if user_id in doc.metadata["user_access"]]
    doc_metadatas = [doc.metadata for doc in documents if user_id in doc.metadata["user_access"]]
    
    # Create FAISS index with texts and metadata
    faiss_index = FAISS.from_texts(
        texts=doc_texts, 
        embedding=embeddings, 
        metadatas=doc_metadatas
    )
    
    # Create memory to store conversation context
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Create the Conversational Retrieval Chain with FAISS and metadata filtering
    chain = ConversationalRetrievalChain.from_llm(
        llm=G4FLLM(),  # Use the custom g4f LLM
        retriever=faiss_index.as_retriever(search_kwargs={"filters": {"users_access": [user_id]}}),
        memory=memory
    )
    
    return chain

# def initialize_chain(doc_texts):
#     # Setup your embedding model (Sentence Transformers for embedding)
#     embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

#     # Use FAISS for vector store (you can replace it with your own document retrieval method)
#     faiss_index = FAISS.from_texts(doc_texts, embeddings)
    
#     # Create memory to store conversation context
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
#     # Create the Conversational Retrieval Chain with g4f LLM, FAISS, and memory
#     chain = ConversationalRetrievalChain.from_llm(
#         llm=G4FLLM(),  # Use the custom g4f LLM
#         retriever=faiss_index.as_retriever(),
#         memory=memory
#     )
    
#     return chain


st.title("Multi-user Conversational Q&A System with Chat ID")

# Dropdown for selecting user ID
st.sidebar.header("User Selection")
user_id = st.sidebar.selectbox("Select User ID:", options=["A", "B", "C"], key="user_id_dropdown")

# Display existing chat history
st.subheader("Existing Chats")
existing_chats = load_chat_history()
if isinstance(existing_chats, list):
    for chat in existing_chats:
        if isinstance(chat, dict) and 'chat_id' in chat and 'question' in chat and 'answer' in chat:
            st.markdown(f"**Chat ID:** {chat['chat_id']}")
            st.markdown(f"**Question:** {chat['question']}")
            st.markdown(f"**Answer:** {chat['answer']}")

            # Allow for a follow-up question on the current chat_id
            follow_up = st.text_input(f"Ask a follow-up for Chat ID {chat['chat_id']} (User: {user_id}):", key=f"followup_input_{chat['chat_id']}")
            
            if st.button(f"Submit Follow-Up for {chat['chat_id']}", key=f"followup_submit_{chat['chat_id']}"):
                # Pass chat ID context to LLM for follow-up
                chain = initialize_chain(get_mock_documents(), user_id)  # Replace with actual docs
                follow_up_prompt = (
                    follow_up
                    + f"\nGiven the very first context as:\nQuestion- {chat['question']} and Answer- {chat['answer']}."
                    + f"\nAdditional conversations in this context: {chat.get('follow_ups', [])}"
                )

                # Invoke the LLM chain with context
                response = chain({"question": follow_up_prompt})  #, "user_id": user_id Include user ID
                
                # Save the follow-up response in the chat history
                chat.setdefault("follow_ups", []).append({"user_id": user_id, "question": follow_up, "answer": response['answer']})
                
                # Save updated chat history
                save_chat_history(existing_chats)

                # Display the follow-up response
                st.success(f"Response for Chat ID {chat['chat_id']}: {response['answer']}")

else:
    st.warning("No valid chat history found.")

# New Question Input
st.markdown("---")
st.subheader("Ask a New Question")
new_question = st.text_input(f"Enter your question (User: {user_id}):", key="new_question_input")

if st.button("Submit New Question", key="new_question_submit"):
    # Simulate generating a new chat ID
    new_chat_id = str(uuid.uuid4())

    # Simulate invoking LLM for the new question
    chain = initialize_chain(get_mock_documents(), user_id)  # Replace with actual docs
    response = chain({"question": new_question})  #, "user_id": user_id Include user ID

    # Create a new chat entry
    new_chat = {
        "chat_id": new_chat_id,
        "user_id": user_id,
        "question": new_question,
        "answer": response['answer'],
        "follow_ups": []
    }

    # Load the current chat history, add the new chat, and save it back
    existing_chats.append(new_chat)
    save_chat_history(existing_chats)

    # Display the new question and answer
    st.success(f"Response for New Question: {response['answer']}")


# Main Streamlit UI
# st.title("ChatGPT-Style Conversational Q&A System with Chat ID")

# # Display existing chat history
# st.subheader("Existing Chats")
# existing_chats = load_chat_history()
# if isinstance(existing_chats, list):
#     for chat in existing_chats:
#         # Ensure chat is a dictionary and contains the necessary keys
#         if isinstance(chat, dict) and 'chat_id' in chat and 'question' in chat and 'answer' in chat:
#             st.markdown(f"**Chat ID:** {chat['chat_id']}")
#             st.markdown(f"**Question:** {chat['question']}")
#             st.markdown(f"**Answer:** {chat['answer']}")

#             print("chat----",chat)

#             # Allow for a follow-up question on the current chat_id
#             follow_up = st.text_input(f"Ask a follow-up for Chat ID {chat['chat_id']}:", key=f"followup_input_{chat['chat_id']}")
#             print("follow_up----",follow_up)

#             if st.button(f"Submit Follow-Up for {chat['chat_id']}", key=f"followup_submit_{chat['chat_id']}"):
#                 # Pass chat ID context to LLM for follow-up
#                 chain = initialize_chain(get_mock_documents())  # Replace with actual docs
#                 follow_up_prompt = follow_up + "\n given the very first context as: \n Question- {question} and Answer- {answer} and some additional conversations in this context itself are : {sub_follow_ups}".format(question=chat["question"], answer=chat["answer"], sub_follow_ups=str(chat["follow_ups"]))
#                 # follow_up_prompt = f"Chat ID: {chat['chat_id']}\n{follow_up}"
#                 print("follow_up_prompt-----",follow_up_prompt)

#                 # Invoke the LLM chain with context
#                 response = chain({"question": follow_up_prompt}) #, "chat_id": chat["chat_id"]

#                 # Save the follow-up response in the chat history
#                 chat["follow_ups"].append({"question":follow_up, "answer":response['answer']})
#                 # chat['follow_up_question'] = follow_up
#                 # chat['follow_up_answer'] = response['answer']

#                 # Save updated chat history
#                 save_chat_history(existing_chats)

#                 # Display the follow-up response
#                 st.success(f"Response for Chat ID {chat['chat_id']}: {response['answer']}")

# else:
#     st.warning("No valid chat history found.")


# # New Question Input
# st.markdown("---")
# st.subheader("Ask a New Question")
# new_question = st.text_input("Enter your question:", key="new_question_input")

# # New question submit button with a unique key
# if st.button("Submit New Question", key="new_question_submit"):
#     # Simulate generating a new chat ID (in a real app, use UUID or a unique identifier)
#     new_chat_id = str(uuid.uuid4())

#     # Simulate invoking LLM for the new question
#     chain = initialize_chain(get_mock_documents())  # Replace with actual docs
#     response = chain({"question": new_question})

#     # Create a new chat entry
#     new_chat = {
#         "chat_id": new_chat_id,
#         "question": new_question,
#         "answer": response['answer'],
#         "follow_ups":[]
#     }

#     # Load the current chat history, add the new chat, and save it back
#     existing_chats.append(new_chat)
#     save_chat_history(existing_chats)

#     # Display the new question and answer
#     st.success(f"Response for New Question: {response['answer']}")
