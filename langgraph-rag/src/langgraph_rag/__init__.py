import bs4
import requests

from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter

from functools import lru_cache
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings

from langchain.tools import tool

from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState

# # deepseek-r1:1.5b

# saytlardan metni cixarmaq ucun istifade olunur
def load_web_page(url: str, bs_kwargs: dict | None = None) -> list[Document]:
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    soup = bs4.BeautifulSoup(response.text, "html.parser", **(bs_kwargs or {}))
    return [Document(page_content=soup.get_text(), metadata={"source": url})]


urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

# saytin nav/script hissesini atib yalniz post metnini goturur
_POST_ONLY = {
    "parse_only": bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
}

docs = [load_web_page(url, bs_kwargs=_POST_ONLY) for url in urls]

docs_list = [item for sublist in docs for item in sublist]

# cixariln textleri split edib vector database vurmaq ucun istifade olunur

# chunk_size kicik olsa 500+ chunk yaranir; Ollama her chunk ucun ayri TCP
# baglantisi acdigina gore Windows-da ephemeral port bitir ve embed 400 verir
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=200,
)
doc_splits = text_splitter.split_documents(docs_list)

# # vector database e textleri add edir

@lru_cache(maxsize=1)
def _get_retriever():
    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits,
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
    )
    return vectorstore.as_retriever()


# # vector db dan query e esasen melumat geri qaytarir istifadeciye donur

@tool
def retrieve_blog_posts(query: str) -> str:
    """Search and return information about Lilian Weng blog posts."""
    retriever = _get_retriever()
    retrieved_docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in retrieved_docs])


retriever_tool = retrieve_blog_posts

retriever_tool.invoke({"query": "types of reward hacking"})


# # llm query yazacaq yoxsa söhbət edəcək onu seçir

# model_provider olmasa "deepseek-" prefiksine gore DeepSeek cloud API secilir,
# bize ise lokal Ollama lazimdir
response_model = init_chat_model(
    "deepseek-r1:1.5b",
    model_provider="ollama",
    temperature=0,
)

def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}


input = {
    "messages": [
        {
            "role": "user",
            "content": "Salam üzeyir hacıbəyli haqqında nə bilirsən?",
        }
    ]
}
generate_query_or_respond(input)["messages"][-1].pretty_print()
