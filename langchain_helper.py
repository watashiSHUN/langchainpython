from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from dotenv import dotenv_values

load_dotenv() # environment variable, instead of passing them explicitly
#print("OpenAI API key:", dotenv_values(".env")["OPENAI_API_KEY"])

def generate_pet_name(animal, color):
    llm = OpenAI(temperature=0.2)
    prompt_template = PromptTemplate(
        input_variables=["animal"], 
        template="Give me five pet names for a {animal}, with color {color}.")
    name_chain = LLMChain(llm=llm, prompt=prompt_template)

    return name_chain({'animal': animal, 'color': color})

if __name__ == "__main__":
    # test the function
    print(generate_pet_name("dog", "brown"))