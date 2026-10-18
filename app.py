import streamlit as st
from agent import chat, resume_chat

st.set_page_config(
    page_title="Financial AI Agent",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Financial AI Agent")
st.write("Ask anything about your financial data.")

THREAD_ID = "default"  # swap for a per-user/session id if you have one

# --------------------------------------------------
# Session state
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None  # holds payload dict while waiting on approval

# --------------------------------------------------
# Display previous messages
# --------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --------------------------------------------------
# Helper: unpack chat()/resume_chat() result into history
# --------------------------------------------------
def handle_result(result: dict):
    if result["status"] == "interrupted":
        st.session_state.pending_interrupt = result["payload"]
    else:
        st.session_state.pending_interrupt = None
        st.session_state.messages.append(
            {"role": "assistant", "content": result["text"]}
        )

# --------------------------------------------------
# CASE 1: waiting on human approval — show approve/reject, block chat_input
# --------------------------------------------------
if st.session_state.pending_interrupt is not None:
    payload = st.session_state.pending_interrupt

    with st.chat_message("assistant"):
        st.warning(payload.get("message", "Approval required"))
        st.json(payload)

        col1, col2 = st.columns(2)
        approve = col1.button("✅ Approve", key="approve_btn")
        reject = col2.button("❌ Reject", key="reject_btn")

    if approve or reject:
        st.session_state.messages.append(
            {"role": "user", "content": "Approved ✅" if approve else "Rejected ❌"}
        )

        with st.spinner("Processing..."):
            result = resume_chat(bool(approve), thread_id=THREAD_ID)

        handle_result(result)
        st.rerun()

# --------------------------------------------------
# CASE 2: normal chat input
# --------------------------------------------------
else:
    prompt = st.chat_input("Ask your question...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = chat(prompt, thread_id=THREAD_ID)

                if result["status"] == "interrupted":
                    st.session_state.pending_interrupt = result["payload"]
                    st.warning(result["payload"].get("message", "Approval required"))
                    st.json(result["payload"])
                else:
                    st.markdown(result["text"])

        handle_result(result)
        st.rerun()