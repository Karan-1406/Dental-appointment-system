import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Load Streamlit Cloud secrets
try:
    secrets = st.secrets

    if secrets.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = secrets["GOOGLE_API_KEY"]

    if secrets.get("MODEL_NAME"):
        os.environ["MODEL_NAME"] = secrets["MODEL_NAME"]

    if secrets.get("TEMPERATURE") is not None:
        os.environ["TEMPERATURE"] = str(secrets["TEMPERATURE"])

except Exception:
    pass

# Validate API key before importing the agent
if not os.getenv("GOOGLE_API_KEY"):
    st.error("Gemini API key is missing. Please configure GOOGLE_API_KEY in Streamlit Secrets.")
    st.stop()

from langchain_core.messages import HumanMessage, AIMessage
from dental_agent.agent import dental_graph


st.set_page_config(
    page_title="Dental Appointment Assistant",
    page_icon="🦷",
    layout="centered"
)

st.title("🦷 Dental Appointment Assistant")
st.caption("Powered by Google Gemini and LangGraph")

st.markdown("""
Welcome! I can help you:
- Check available doctors and appointment slots
- Book dental appointments
- Cancel existing appointments
- Reschedule appointments
""")

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []


def get_text(content):
    """Convert Gemini text blocks into readable text."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)

    return str(content)


# Display previous conversation
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(get_text(message.content))

    elif isinstance(message, AIMessage):
        content = get_text(message.content)
        if content.strip() and not message.tool_calls:
            with st.chat_message("assistant"):
                st.markdown(content)


# Chat input
if prompt := st.chat_input("How can I help you today?"):

    user_message = HumanMessage(content=prompt)
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                result = dental_graph.invoke(
                    {"messages": st.session_state.messages},
                    config={"recursion_limit": 20}
                )

            st.session_state.messages = result["messages"]

            # Display the final AI response
            final_response = None

            for message in reversed(st.session_state.messages):
                if isinstance(message, AIMessage) and not message.tool_calls:
                    text = get_text(message.content)
                    if text.strip():
                        final_response = text
                        break

            if final_response:
                st.markdown(final_response)
            else:
                st.info("I couldn't generate a response. Please try again.")

        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.session_state.messages.pop()

