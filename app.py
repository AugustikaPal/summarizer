# import validators,streamlit as st
# from langchain.prompts import PromptTemplate
# from langchain_groq import ChatGroq
# from langchain.chains.summarize import load_summarize_chain
# from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
# from langchain_huggingface import HuggingFaceEndpoint


# ## sstreamlit APP
# st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
# st.title("🦜 LangChain: Summarize Text From YT or Website")
# st.subheader('Summarize URL')



# ## Get the Groq API Key and url(YT or website)to be summarized
# with st.sidebar:
#     hf_api_key=st.text_input("Huggingface API Token",value="",type="password")

# generic_url=st.text_input("URL",label_visibility="collapsed")

# ## Gemma Model USsing Groq API
# ##llm =ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)
# repo_id="mistralai/Mistral-7B-Instruct-v0.3"
# llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=150,temperature=0.7,token=hf_api_key)

# prompt_template="""
# Provide a summary of the following content in 300 words:
# Content:{text}

# """
# prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

# if st.button("Summarize the Content from YT or Website"):
#     ## Validate all the inputs
#     if not hf_api_key.strip() or not generic_url.strip():
#         st.error("Please provide the information to get started")
#     elif not validators.url(generic_url):
#         st.error("Please enter a valid Url. It can may be a YT video utl or website url")

#     else:
#         try:
#             with st.spinner("Waiting..."):
#                 ## loading the website or yt video data
#                 if "youtube.com" in generic_url:
#                     loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
#                 else:
#                     loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
#                                                  headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
#                 docs=loader.load()

#                 ## Chain For Summarization
#                 chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
#                 output_summary=chain.run(docs)

#                 st.success(output_summary)
#         except Exception as e:
#             st.exception(f"Exception:{e}")
import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEndpoint

# Streamlit App Configuration
st.set_page_config(
    page_title="LangChain: Summarize YT or Website Content",
    page_icon="🦜",
    layout="wide"
)

# Title and Subtitle
st.title("🦜 LangChain: Summarize Content from YouTube or Websites")
st.subheader("Quickly get the key insights from any URL!")

# Sidebar for API Key Input
with st.sidebar:
    st.header("🔑 API Configuration")
    hf_api_key = st.text_input(
        "Huggingface API Token", value="", type="password", help="Enter your Huggingface API token here."
    )

# Main Content
st.write("Enter a URL (YouTube or website) below to get a concise summary.")

# URL Input Section with Placeholder and Clear Label
generic_url = st.text_input(
    "Enter YouTube or Website URL",
    placeholder="https://example.com",
    label_visibility="visible"
)

# Model Configuration
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=150, temperature=0.7, token=hf_api_key)

# Prompt Template
prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Button to Trigger Summarization
if st.button("🔍 Summarize Content"):
    # Validate Inputs
    if not hf_api_key.strip():
        st.error("Please provide your Huggingface API Token.")
    elif not generic_url.strip():
        st.error("Please enter a URL.")
    elif not validators.url(generic_url):
        st.error("Invalid URL! Please enter a valid YouTube or website URL.")
    else:
        try:
            with st.spinner("Summarizing content... Please wait."):
                # Load data from the URL
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                docs = loader.load()

                # Summarization Chain
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                # Display the Summary
                st.success("Summary generated successfully!")
                st.write(output_summary)
                
                # Copy to Clipboard Button
                st.button("📋 Copy Summary", on_click=lambda: st.write("Summary copied!"))  # Placeholder

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Footer or Additional Information
st.markdown("---")
st.markdown("Powered by [LangChain](https://www.langchain.com) | Built with Streamlit")
