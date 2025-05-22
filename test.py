from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

import streamlit as st
import os
import pathlib
import textwrap
from PIL import Image
import google.generativeai as genai
import base64
from io import BytesIO
import json
import requests
import time

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# local run
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# runpod_key = os.getenv("Runpod_Key")
# runpod_id = os.getenv("runpod_id")


# deployment
genai.configure(api_key=st.secrets['GOOGLE_API_KEY'])
runpod_key = st.secrets['runpod_key']
runpod_id = st.secrets['runpod_id']


def get_gemini_response(question):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(question)
    return response.text

def get_gemini_response_image(input,image):
    model = genai.GenerativeModel('gemini-1.5-flash')
    if input!="":
       response = model.generate_content([input,image])
    else:
       response = model.generate_content(image)
    return response.text

def qwen_response(input, encoded_image, api_key, runpod_id):
    client = OpenAI(
    api_key=api_key,
    base_url= f"https://api.runpod.ai/v2/{runpod_id}/openai/v1")
  
    encoded_image_text = encoded_image.decode("utf-8")
    # print(encoded_image_text)
    base64_qwen = f"data:image;base64,{encoded_image_text}"
    chat_response = client.chat.completions.create(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        messages=[
            {"role": "system", "content": "Your goal is analyze documents in polish lenguage and write down all information included"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base64_qwen
                        },
                    },
                    {"type": "text", "text": input},
                ],
            },
            
        ],
        max_tokens= 2000,
        temperature=0.2
    )

    return chat_response.choices[0].message.content


with st.sidebar:
    st.header("Mindbox documents processing")
    # text_input_prompt =st.text_input("Enter the prompt: ",key="input")
    # st.markdown("<h1 style='text-align: center;'>(or)</h1>", unsafe_allow_html=True)
    option = st.selectbox("Select llm model:",
                          ("gemini-1.5-flash", "qwen2.5-vl-7B [open_source]")),
    img_input_prompt =st.text_input("Enter the prompt: ",key="input1")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    image="" 
    if img_input_prompt=="":
                img_input_prompt = "What is wirtten on the document. Structure it in JSON output."
    submit=st.button("Generate response")


if submit:
    if uploaded_file:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            im_file = BytesIO()
            image.save(im_file, format="png")
            im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    
            im_b64 = base64.b64encode(im_bytes)
            # print(base64.b64encode(uploaded_file).decode('utf-8'))
            st.image(image, caption="Uploaded Image.", use_column_width=True)
            st.subheader("Generated response:")
            print(option)
          
            if "gemini-1.5-flash" in option:
                response=get_gemini_response_image(img_input_prompt,image)
            else:
                response = qwen_response(input=img_input_prompt, encoded_image=im_b64,api_key=runpod_key, runpod_id=runpod_id)

        st.write(response)
    else:
        st.write("Please upload a document")