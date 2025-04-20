from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOllama
chroma_db_path = "vector_store"

def get_bot():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")       #lightweight msmaller, memory-efficient model
        
        vectordb = Chroma(
            persist_directory=chroma_db_path,
            embedding_function=embeddings
        )

        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        #Use a lighter LLM-Mistral
        llm = ChatOllama(model="mistral")

        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory
        )
        return qa
    
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        return None



# Load once
chatbot = get_bot()

def get_bot_response(user_query):
    if chatbot:
        try:
            return chatbot.run(user_query)
        except Exception as e:
            return f"error during response: {e}"
    else:
        return "Chatbot failed to initialize."
