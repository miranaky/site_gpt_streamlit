import os

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import SitemapLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

folders = [
    "./.cache",
    "./.cache/files",
]
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)


openai_api_key = None

st.set_page_config(page_title="Cloudflare Site GPT", page_icon="🖥️")


def load_llm(openai_api_key):
    return ChatOpenAI(
        temperature=0.1,
        model="gpt-4o-mini",
        api_key=openai_api_key,
        streaming=True,
    )


answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm

    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]

    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


def parse_page(soup):
    header = soup.find("header")
    navs = soup.find_all("nav")
    footer = soup.find("footer")
    footer_link = soup.find("div", {"class": "items-center flex flex-wrap"})
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    for nav in navs:
        nav.decompose()
    if footer_link:
        footer_link.decompose()
    return str(soup.get_text()).replace("\n", " ").replace("Skip to content", "")


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=2000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        parsing_function=parse_page,
        filter_urls=[
            r"^(.*\/ai-gateway\/).*",
            r"^(.*\/vectorize\/).*",
            r"^(.*\/workers-ai\/).*",
        ],
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings(api_key=openai_api_key))
    return vector_store.as_retriever()


def save_message(message, role):
    st.session_state.messages.append({"role": role, "message": message})


def send_message(message: str, role: str, save: bool = True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state.messages:
        send_message(
            message=message["message"],
            role=message["role"],
            save=False,
        )


st.title("Cloudflare Site GPT")

st.markdown(
    """            
    Ask questions about the content of a Cloudflare.
    Start by writing the openai api key in the sidebar.
"""
)

with st.sidebar:
    st.write("github repo: https://github.com/miranaky/site_gpt_streamlit")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    st.write("")

if openai_api_key:
    llm = load_llm(openai_api_key)
    retriever = load_website("https://developers.cloudflare.com/sitemap-0.xml")
    send_message("I'm ready to answer for cloudflare", role="ai", save=False)
    paint_history()

    query = st.chat_input("Ask a question to the website.")
    if query:
        send_message(message=query, role="human")

        chain = (
            {
                "docs": retriever,
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(get_answers)
            | RunnableLambda(choose_answer)
        )
        response = chain.invoke(query)
        send_message(response.content, role="ai")

else:
    st.session_state.messages = []
