import gradio
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate                                    # To format prompts
from langchain_core.output_parsers import StrOutputParser                            # to transform the output of an LLM into a more usable format
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough          # Required by LCEL (LangChain Expression Language)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import torch

username = "aisha-partha"
my_repo = "RAG-PDF-Chatbot"





def generate_query_response(question):
    llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",       # Model card: https://huggingface.co/HuggingFaceH4/zephyr-7b-beta
    task="text-generation",
    max_new_tokens = 512,
    top_k = 30,
    temperature = 0.1,
    repetition_penalty = 1.03,
    )
    template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Always say "thanks for asking!" at the end of the answer.
        {context}
        Question: {question}
        Helpful Answer:"""

    QA_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
    
    embedding =  HuggingFaceEmbeddings(
    model_name="mixedbread-ai/mxbai-embed-large-v1",                             # Provide the pre-trained model's path
    model_kwargs={'device': "cuda" if torch.cuda.is_available() else "cpu"},     # Pass the model configuration options
    encode_kwargs={'normalize_embeddings': False}                                # Pass the encoding options
    )

    vectordb = Chroma(persist_directory = 'docs/chroma/',
                  embedding_function = embedding
    )
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 7, "fetch_k":15})

    retrieval = RunnableParallel(
    {
        "context": RunnablePassthrough(context= lambda x: x["question"] | retriever),
        "question": RunnablePassthrough()
        }
    )
    
    rag_chain = (retrieval                     # Retrieval
             | QA_PROMPT                   # Augmentation
             | llm                         # Generation
             | StrOutputParser()
             )
    
    response = rag_chain.invoke({"question": question})
    
    return response

# Input from user
in_query = gradio.Textbox(lines=10, placeholder=None, value="Query", label='Enter Query')

# Output prediction
out_response = gradio.Textbox(type="text", label='Response')


# Gradio interface to generate UI
ragbot = gradio.Interface(fn = generate_query_response,
                         inputs = [in_query],
                         outputs = [out_response],
                         title = "PDF Inference",
                         description = "RAG with Open Source Model",
                         allow_flagging = 'never')

ragbot.launch(share=True)
