import streamlit as st

from agent import chat

st.set_page_config(
    page_title="Financial AI Agent",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Financial AI Agent")

st.write("Ask anything about your financial data.")

# Session history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Ask your question...")

if prompt:

    # User message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = chat(prompt)

            st.markdown(response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
        }
    )