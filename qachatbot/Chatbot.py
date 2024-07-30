import streamlit as st
import pandas as pd
import sys
# sys.path.insert(1, r'D:\Navya\Coding\PythonNew\pythonProject\Machine Learning')
from QA_Chatbot import get_response

# try:
#     from QA_Chatbot import Chatbot, get_response
#     print("Module imported successfully.")
# except ImportError as e:
#     print(f"Error importing module: {e}")


def toggle_notes():
    st.session_state.notes_open = not st.session_state.notes_open


with st.sidebar:
    st.header('InfoHub', divider='rainbow')
    if 'notes_open' not in st.session_state:
        st.session_state.notes_open = False


    # option = st.selectbox(
    #     "Mode:",
    #     ("Dark Mode", "Light Mode"))

from dataclasses import dataclass
from typing import Literal

@dataclass
class Message:
    origin:Literal["human", "ai"]
    message:str


def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history=[]
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    # if 'qa_chain' not in st.session_state:
    #     st.session_state.Chatbot_obj = Chatbot()
    #     st.session_state.Chatbot_obj.setup_model()
    #     st.session_state.qa_chain = st.session_state.Chatbot_obj.qa_chain()
    # if "token_count" not in st.session_state:
    #     st.session_state.token_count=0
    # if "conversation" not ni st.session_state:
    #     llm=OpenAI(
    #         temperature=0,
    #         open_api_key=st.secrets["openai_api_key"],
    #         model_name="text-davinci-003"
    #     )
def on_click_callback():
    human_prompt=st.session_state.human_prompt
    # llm_response=st.session_state.history.append(human_prompt)
    if human_prompt:  # Ensure there is input before processing
        st.session_state.history.append(human_prompt)
        st.session_state.past.append(f"You: {human_prompt}")

        # Simulating a bot response (replace this with actual bot logic)
        # llm_response = "This is a simulated response."
        # import pdb; pdb.set_trace()
        # qa_chain = st.session_state.qa_chain
        llm_response = get_response(human_prompt)
        st.session_state.generated.append(llm_response)
        st.session_state.past.append(f"Bot: {llm_response}")

        # Clear the input field after processing
        st.session_state.human_prompt = ""
    # st. session_state.history.append(
    # Message ("human", human_prompt)
    # )
    # st.session_state.history.append(Message("ai",llm_response))


initialize_session_state()

# llm = OpenAI (
#     temperature=0,
#     openai_api_key=st. secrets ["openai_api_key"], model_name="text-davinci-003")
# conversation = ConversationChain (
#     llm=llm,
#     memory=ConversationSummaryMemory(llm=llm),
# )

st.title("Hello! Welcome to InfoHub!")
chat_placeholder=st.container()
prompt_placeholder=st.form("chat_form")
credit_card_placeholder=st.empty()

# with chat_placeholder:
#     for chat in st.session_state.history:
#         st.markdown(chat)

with prompt_placeholder:
    st.markdown("**Chat** - _press Enter to submit_ ")
    cols=st.columns((6,1))
    cols[0].text_input(
        "Chat",
        value="Hello bot",
        label_visibility="collapsed",
        key="human_prompt",
    )
    cols[1].form_submit_button(
        "Submit",
        type="primary",
        on_click=on_click_callback
        )


def clear_text():
    st.session_state["human_prompt"] = ""
st.button("Clear Text", on_click=clear_text)
# credit_card_placeholder.caption(f"""
# Used {st.session_state.token_count}tokens""")


#prompt = st.chat_input("Say something")
# if prompt:
#     st.write(f"User has sent the following prompt: {prompt}")


if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])):
        st.markdown(f"You: {st.session_state['history'][i]}")
        st.markdown(f"Bot: {st.session_state['generated'][i]}")

with st.sidebar:
    col1, col2 = st.columns(2)

    with col1:
        st.button("Notes", on_click=toggle_notes)
    if st.session_state.notes_open:
        st.text_area("Enter your notes here:", "")


    with col2:
        if st.button("New Chat"):
            # Clear the chat history when "New Chat" button is clicked
            st.session_state.history.clear()
            st.session_state.generated.clear()
            st.session_state.past.clear()
            # Rerun the script to update the app state
            st.experimental_rerun()



    st.subheader("Search History")
    i = 1
    for chat in st.session_state.history:
        st.text(f"{i}. {chat}")
        i += 1



