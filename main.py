import langchain_helper as lch
import streamlit as st

st.title("Pet Name Generator")
animal_type = st.sidebar.selectbox("Select your pet type", ["dog", "cat", "horse", "bird", "fish", "reptile", "rodent", "insect"])

# the string is not dynamic
animal_color = st.sidebar.text_area("What is the color of your animal?", max_chars=20)

if st.sidebar.button("Generate"):
    openai_response = lch.generate_pet_name(animal_type, animal_color)
    st.write(openai_response["name"])