"""
streamlit_app.py
Mobile-friendly vegetable scanner: take/upload a photo, classify it with a
fine-tuned EfficientNetB0, then chat with Gemini 3.1 Flash-Lite (via
LangChain) about the identified vegetable.

Run locally:
    streamlit run app/streamlit_app.py
"""

from PIL import Image
import streamlit as st

from llm_chain import get_response
from utils import (
    is_confident,
    load_class_indices,
    load_model,
    predict_top_k,
    preprocess_image,
)

# ----------------------------------------------------------------------
# Page config + light mobile-friendly styling
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Veggie Scanner",
    page_icon="🥕",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        /* Bigger touch targets for mobile */
        .stButton > button {
            width: 100%;
            padding: 0.75rem;
            font-size: 1.05rem;
            border-radius: 10px;
        }
        /* Keep content readable on narrow screens */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 560px;
        }
        div[data-testid="stCameraInput"] video {
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🥕 Veggie Scanner")
st.caption("Scan or upload a photo of a vegetable, then ask me anything about it.")

# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
if "image_file" not in st.session_state:
    st.session_state.image_file = None
if "current_vegetable" not in st.session_state:
    st.session_state.current_vegetable = None
if "top_predictions" not in st.session_state:
    st.session_state.top_predictions = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------------------------------------------------
# Image input: camera (mobile) or file upload (fallback)
# ----------------------------------------------------------------------
if st.session_state.image_file is None:
    tab_camera, tab_upload = st.tabs(["📷 Camera", "🖼️ Upload"])

    with tab_camera:
        camera_image = st.camera_input("Take a photo of the vegetable", label_visibility="collapsed")
        if camera_image is not None:
            st.session_state.image_file = camera_image

    with tab_upload:
        uploaded_image = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            st.session_state.image_file = uploaded_image

    if st.session_state.image_file:
        st.rerun()
# ----------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------
if st.session_state.image_file is not None:
    pil_image = Image.open(st.session_state.image_file)

    st.image(pil_image, caption="Scanned image", width='stretch')

    with st.spinner("Identifying..."):
        model = load_model()
        idx_to_class = load_class_indices()
        img_array = preprocess_image(pil_image)
        top_preds = predict_top_k(model, img_array, idx_to_class, k=3)

    if is_confident(top_preds):
        best_label, best_prob = top_preds[0]

        # Reset chat if a *new* vegetable was scanned
        if st.session_state.current_vegetable != best_label:
            st.session_state.current_vegetable = best_label
            st.session_state.messages = []

        st.success(f"**{best_label}** ({best_prob * 100:.1f}% confidence)")

        with st.expander("See top 3 predictions"):
            for label, prob in top_preds:
                st.write(f"- {label}: {prob * 100:.1f}%")

        if st.button("🔄 Scan a different vegetable"):
            st.session_state.image_file = None
            st.session_state.current_vegetable = None
            st.session_state.messages = []
            st.rerun()

    else:
        st.session_state.current_vegetable = None
        st.warning(
            "I'm not confident this is one of the vegetables I know. "
            "Try a closer, well-lit photo of a single vegetable."
        )

        if st.button("🔄 Scan a different vegetable"):
            st.session_state.image_file = None
            st.session_state.current_vegetable = None
            st.session_state.messages = []
            st.rerun()

# ----------------------------------------------------------------------
# Chat about the identified vegetable
# ----------------------------------------------------------------------
if st.session_state.current_vegetable:
    st.divider()
    st.subheader(f"Ask me about {st.session_state.current_vegetable}")

    for turn in st.session_state.messages:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    user_input = st.chat_input("e.g. Is this good for weight loss? How do I store it?")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    reply = get_response(
                        st.session_state.current_vegetable,
                        st.session_state.messages[:-1],
                        user_input,
                    )
                except Exception as e:
                    reply = f"Sorry, something went wrong reaching the LLM: {e}"
                st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
