import streamlit as st
from groq import Groq

from agents.router_agent import RouterAgent
from agents.research_agent import ResearchAgent
from agents.critique_agent import CritiqueAgent
from agents.messages import ResearchRequest
from rag.retriever import Retriever

st.set_page_config(page_title="AI Fitness & Lifestyle Research Assistant", page_icon="🏋️")


@st.cache_resource
def load_pipeline():
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    retriever = Retriever()
    return (
        RouterAgent(client),
        ResearchAgent(client, retriever),
        CritiqueAgent(client),
    )


router, researcher, critic = load_pipeline()

st.title("🏋️ AI Fitness & Lifestyle Research Assistant")
with st.sidebar:
    st.header("📜 History")
    if "history" in st.session_state and st.session_state.history:
        # group into (question, answer) pairs
        pairs = []
        for i in range(0, len(st.session_state.history) - 1, 2):
            if st.session_state.history[i]["role"] == "user":
                pairs.append((st.session_state.history[i]["content"],
                               st.session_state.history[i + 1]["content"]
                               if i + 1 < len(st.session_state.history) else ""))
        for i, (q, a) in enumerate(reversed(pairs), 1):
            with st.expander(f"{len(pairs) - i + 1}. {q[:40]}{'...' if len(q) > 40 else ''}"):
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {a}")
        if st.button("🗑️ Clear history"):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("No questions asked yet this session.")
        
st.caption(
    "Ask a research-backed question about training, nutrition, sleep, or "
    "recovery. Answers are grounded in a curated document corpus — this "
    "is not a substitute for medical advice."
)

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

user_query = st.chat_input("Ask a question...")

if user_query:
    st.session_state.history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Routing query..."):
            label = router.route(user_query)

        if label == "SMALL_TALK":
            answer = "Hi! Ask me a fitness, nutrition, sleep, or recovery question and I'll dig into the research for you."
        elif label == "OUT_OF_SCOPE":
            answer = (
                "That's outside what I can help with — I'm scoped to "
                "fitness, nutrition, sleep, and recovery research, and I "
                "can't provide individual medical diagnosis or treatment."
            )
        else:
            with st.spinner("Researching..."):
                request = ResearchRequest(query=user_query)
                draft = researcher.handle(request)
            with st.spinner("Double-checking the answer..."):
                final = critic.review(draft)
            answer = final.answer
            if final.sources:
                answer += "\n\n**Sources:** " + ", ".join(final.sources)

        st.markdown(answer)
        st.session_state.history.append({"role": "assistant", "content": answer})