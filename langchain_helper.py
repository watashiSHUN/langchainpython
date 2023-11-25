from tabnanny import verbose
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from dotenv import load_dotenv
from dotenv import dotenv_values


load_dotenv() # environment variable, instead of passing them explicitly
#print("OpenAI API key:", dotenv_values(".env")["OPENAI_API_KEY"])

def generate_pet_name(animal, color):
    llm = OpenAI(temperature=0.2)
    prompt_template = PromptTemplate(
        input_variables=["animal"], 
        template="Give me five pet names for a {animal}, with color {color}.")
    name_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="name")

    return name_chain({'animal': animal, 'color': color})

def langchain_agent():
    llm = OpenAI(temperature=0.2)
    tools = load_tools(["wikipedia","llm-math"], llm=llm)
    agent = initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)
    result = agent.run("What is the average age of a dog?, how many human years does 1 dog year map to?")

if __name__ == "__main__":
    # test the function
    print(generate_pet_name("dog", "brown"))
    print(langchain_agent())