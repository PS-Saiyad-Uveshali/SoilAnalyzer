import streamlit as st
import fitz  # PyMuPDF
from model import llm
from prompt import get_chain

st.set_page_config(page_title="Soil Analyzer", layout="centered")
st.title("🌱 Soil Report Analyzer")

uploaded_file = st.file_uploader("Upload a Soil Test Report (PDF)", type=["pdf"])

if uploaded_file:
    with st.spinner("Reading PDF..."):
        pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        soil_text = "\n".join([page.get_text() for page in pdf])

    st.subheader("📄 Extracted Text Preview")
    st.text_area("PDF Content", soil_text, height=200)

    if st.button("🔍 Analyze Soil Report"):
        with st.spinner("Analyzing with Gemini..."):
            try:
                chain = get_chain(llm)
                result = chain.invoke(soil_text)

                st.success("✅ Analysis Completed!")
                st.subheader("🧪 Soil Quality")
                st.write(result.quality)

                st.subheader("🌾 Recommended Crops")
                st.write(", ".join(result.recommended_crops))

                st.subheader("💡 Suggestions")
                st.write(result.suggestions)

            except Exception as e:
                st.error(f"Error: {str(e)}")
