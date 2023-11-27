from cgitb import text
from tabnanny import verbose
from venv import create
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from dotenv import load_dotenv
from dotenv import dotenv_values
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# (TODO) what is embedding?
from langchain.embeddings.openai import OpenAIEmbeddings


load_dotenv()  # environment variable, instead of passing them explicitly
# print("OpenAI API key:", dotenv_values(".env")["OPENAI_API_KEY"])


def generate_pet_name(animal, color):
    llm = OpenAI(temperature=0.2)
    prompt_template = PromptTemplate(
        input_variables=["animal"],
        template="Give me five pet names for a {animal}, with color {color}.",
    )
    name_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="name")

    return name_chain({"animal": animal, "color": color})


def langchain_agent():
    llm = OpenAI(temperature=0.2)
    tools = load_tools(["wikipedia", "llm-math"], llm=llm)
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    result = agent.run(
        "What is the average age of a dog?, how many human years does 1 dog year map to?"
    )


def create_vector_db_from_youtube(url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, OpenAIEmbeddings())
    return db


def get_response_from_query(db, query, k=4):
    db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in db.documents])


if __name__ == "__main__":
    # test the function
    # print(generate_pet_name("dog", "brown"))
    # print(langchain_agent())
    print(create_vector_db_from_youtube("https://www.youtube.com/watch?v=byYlC2cagLw"))
