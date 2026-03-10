from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector_qna import retriever
import streamlit as st
from multimodal_handler import MultimodalHandler
import sys
import os

# To enable importing from the same dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

multimodal_handler = MultimodalHandler(
    google_maps_api_key="AIzaSyADjmzRSlR1fl31rdWcdaZVYO3kJUkM3GE"
)

template = """
You are an expert assistant for PolyU (Hong Kong Polytechnic University) freshmen.
Your task is to answer questions based on the provided FAQ information.

RELEVANT FAQ ENTRIES:
{answers}

USER'S QUESTION: {question}

QUERY TYPE: {query_type}
EXTRACTED ENTITY: {entity}

INSTRUCTIONS:
1. Answer the question using ONLY the information from the relevant FAQ entries above
2. If the answer can be found in the FAQ entries, provide a clear and helpful answer
3. If the answer is NOT in the FAQ entries, say: "I don't have specific information about that in my knowledge base. You may want to check the official PolyU website or contact the relevant department."
4. Include specific contact emails, phone numbers, or URLs when mentioned in the FAQ
5. Keep your answer concise and focused on the user's question
6. If this is a location query, make sure to mention the location name clearly in your answer.


ANSWER:
"""
# Specify model
model = OllamaLLM(model="deepseek-r1")

prompt = ChatPromptTemplate.from_template(template)

# Invoke the entire chain
chain = prompt | model


def format_retrieved_documents(docs):
    """Format the retrieved documents for the prompt"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        # Extract content from Document object
        content = doc.page_content
        formatted.append(f"FAQ {i}: {content}")
    return "\n\n".join(formatted)


def get_chatbot_response(question):
    # 1. Fast regex detection
    query_type = multimodal_handler.detect_query_type(question)
    entity = None

    # 2. If regex says text, double-check with tiny LLM
    if query_type == "text":
        fallback_type = multimodal_handler.classify_with_tiny_llm(question)
        if fallback_type == "location":
            query_type = "location"
            print("Tiny LLM overrode to location")

    # 3. For location, extract entity (regex first, then tiny LLM)
    if query_type == "location":
        entity = multimodal_handler.extract_location_entity(question)
        if not entity:
            entity = multimodal_handler.extract_with_tiny_llm(question)
            print("Tiny LLM extracted entity")

    # 4. Retrieve FAQs and generate answer (same as before)
    answers = retriever.invoke(question)
    if not answers:
        return {"text": "Sorry, no relevant information found.", "type": "text"}, []

    formatted_answers = format_retrieved_documents(answers)
    result = chain.invoke({
        "answers": formatted_answers,
        "question": question,
        "query_type": query_type,
        "entity": entity if entity else ""
    }).strip()

    response = {"text": result, "type": query_type}
    if query_type == "location" and entity:
        map_html, coords = multimodal_handler.get_map_embed(entity)
        response["map_html"] = map_html
        response["coordinates"] = coords
        response["location_name"] = entity

    return response, answers


"""
def cli_interface():
    print("=" * 60)
    print("POLYTECHNIC UNIVERSITY OF HONG KONG - ASSISTANT")
    print("=" * 60)
    print("\nI can help with questions about:")
    print("- Student ID cards and entry visas")
    print("- Academic programs and registration")
    print("- Course enrollment and schedules")
    print("- PolyU systems and resources etc.")
    print("\nType 'q' to quit at any time.\n")

    while True:
        print("\n" + "-" * 40)
        question = input("Ask your question: ").strip()

        if question.lower() == 'q':
            print("\nThank you for using the PolyU Assistant. Good luck with your studies!")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            response, answers = get_chatbot_response(question, "text", "")

            print("\n" + "=" * 40)
            print("ANSWER:")
            print("=" * 40)
            print(response["text"])

            # Show number of sources used
            if answers:
                print(f"\n[Based on {len(answers)} relevant FAQ entries]")

        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please try again or contact technical support.")
"""


def streamlit_interface():
    st.set_page_config(page_title="PolyU Assistant", layout="wide")

    with st.sidebar:
        st.title("PolyU Assistant")
        st.markdown("...")  # keep your existing sidebar content
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your PolyU assistant. How can I help you today?"}
        ]

    st.title("PolyU Assistant")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("map_html"):
                st.components.v1.html(message["map_html"], height=400)

    # Chat input – NO KEY ARGUMENT
    if prompt := st.chat_input("Ask about PolyU (e.g., 'Where is the library?')"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, answers = get_chatbot_response(prompt)
            st.markdown(response["text"])
            if response.get("type") == "location" and response.get("map_html"):
                st.components.v1.html(response["map_html"], height=400)

            if answers:
                with st.expander(f"Based on {len(answers)} FAQ entries"):
                    for i, doc in enumerate(answers, 1):
                        st.markdown(f"**FAQ {i}:** {doc.page_content[:200]}...")

        assistant_msg = {"role": "assistant", "content": response["text"]}
        if response.get("map_html"):
            assistant_msg["map_html"] = response["map_html"]
        st.session_state.messages.append(assistant_msg)


if __name__ == "__main__":
    streamlit_interface()
