# run---> streamlit run streamlit_chat.py


import time
import streamlit as st
from streamlit import session_state
from src.config.core_config import settings
from src.chatbot.agents.agent_openai_tools import CampusManagementOpenAIToolsAgent
from src.chatbot.tools.utils.tool_helpers import visited_links
from src.chatbot_log.chatbot_logger import logger
from pages.utils import initialize_session_sate

# create an instance of the agent executor
# TODO every time the users interacts with the chatbot, all the script  re-runS. This is not efficient. CACHE THE AGENT EXECUTOR??? (solved singltone pattern)
# TODO The id of the agent executor object is different every time the script runs. This is not efficient. CACHE THE AGENT EXECUTOR???
# TODO shall I cache this object and only recreate it when the input changes? the only thing that changes is the language (prompt)


initialize_session_sate()


greeting_message = session_state["_"]("How can I help you?")


agent_executor = CampusManagementOpenAIToolsAgent.run()

# App title
st.set_page_config(
    page_title="Campus Management Chatbot",
    page_icon="app/static/Icon-chatbot.png",
    layout="centered",
    initial_sidebar_state="collapsed",
)


with open("./pages/static/style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)


if session_state.get("show_warning", True):
    st.warning(
        session_state["_"](
            "The responses from the chat assistant may be incorrect - therefore, please verify the answers for their correctness."
        )
    )
    if st.button(session_state["_"]("I understand")):
        session_state["show_warning"] = False
        st.rerun()


st.title("Campus Management")

# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "avatar": "./static/Icon-chatbot.svg",
            "content": greeting_message,
        }
    ]

# Display chat messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"], avatar=message["avatar"]):
        st.write(message["content"])

# User-provided prompt
if prompt := st.chat_input(placeholder=session_state["_"]("Message")):
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "avatar": "./static/Icon-User.svg"}
    )
    with st.chat_message("user", avatar="./static/Icon-User.svg"):
        st.write(prompt)


def get_feedback(user_query, response, time_taken, rate):
    feedback = {
        "user_query": user_query,
        "response": response,
        "time_taken": time_taken,
        "rate": rate,
    }
    logger.info(f"Feedback= {feedback}")


# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant", avatar="./static/Icon-chatbot.svg"):
        with st.spinner(session_state["_"]("Generating response...")):

            logger.info(f"User's query: {prompt}")
            # start time
            start_time = time.time()
            settings.time_request_sent = start_time

            # chunks = []
            # for chunk in agent_executor._agent_executor.stream({"input": prompt}):
            #     chunks.append(chunk)
            #     st.write(chunk)

            # streaming is done in CallbackHandlerStreaming
            response = agent_executor(prompt)
            # response = agent.invoke({"input": prompt})
            # end time
            end_time = time.time()
            time_taken = end_time - start_time
            logger.info(
                f"Time taken to serve whole answer to the user: {time_taken} seconds"
            )

            # TODO if pdf is too long to display, provide a download link
            # check if the response contains a PDF file

            if visited_links():
                with st.expander("Sources"):
                    for link in visited_links():
                        st.write("- " + link)
                visited_links.clear()

            # if "sources" in response:
            #     if response["sources"]:
            #         with st.expander("Sources"):
            #             for source in response["sources"]:
            #                 st.write(f"- {source}")

            # Comment out the following code block to disable the PDF download feature
            # pdf_content, pdf_file_name = extract_pdf_with_timeout(
            #     response["output"], 15
            # )
            # if pdf_content:

            #     st.download_button(
            #         label=f"Download: {pdf_file_name}",
            #         data=pdf_content,
            #         file_name=pdf_file_name,
            #         mime="application/pdf",
            #     )

    col1, col2, col3, col4 = st.columns([3, 3, 0.5, 0.5])

    with col3:
        with st.container(key="FeedbackLike"):
            st.button(
                label=":thumbsup:",
                type="primary",
                on_click=get_feedback,
                key="like-button",
                args=[prompt, response["output"], time_taken, "like"],
                help=session_state["_"]("Click here if you like the response"),
            )

    with col4:
        with st.container(key="FeedbackDislike"):
            st.button(
                ":thumbsdown:",
                on_click=get_feedback,
                type="primary",
                key="dislike-button",
                args=[prompt, response["output"], time_taken, "dislike"],
                help=session_state["_"]("Click here if you dislike the response"),
            )

    message = {
        "role": "assistant",
        "content": response["output"],
        "avatar": "./static/Icon-chatbot.svg",
    }
    st.session_state.messages.append(message)