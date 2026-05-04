import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="AI Data Analyst (OpenRouter)", layout="wide")
st.title("📊 SMART AI DATA ANALYST DASHBOARD (OpenRouter)")

# ---------------- OPENROUTER API KEY ----------------
OPENROUTER_API_KEY = ""

# ---------------- AI FUNCTION ----------------
def ask_ai(prompt):
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "AI Data Analyst App"
            },
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional data analyst. Give short, clear and useful answers."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        )

        # SAFE CHECK (IMPORTANT)
        result = response.json()

        if "choices" in result:
            return result["choices"][0]["message"]["content"]
        else:
            return f"❌ API Error: {result}"

    except Exception as e:
        return f"❌ Request Failed: {str(e)}"


# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:

    try:
        df = pd.read_csv(uploaded_file, encoding="latin1")

        # clean types
        df = df.convert_dtypes()

        st.subheader("📄 Data Preview")
        st.dataframe(df)

        # ---------------- CLEANING ----------------
        st.subheader("🧹 Data Cleaning Report")

        col1, col2 = st.columns(2)

        with col1:
            st.write("Missing Values")
            st.write(df.isnull().sum())

        with col2:
            st.write("Duplicate Rows")
            st.write(df.duplicated().sum())

        # ---------------- NUMERIC DATA ----------------
        numeric_df = df.select_dtypes(include=["number"])

        # ---------------- STATS ----------------
        st.subheader("❓ Quick Statistics")

        if not numeric_df.empty:

            choice = st.selectbox("Select:", ["None", "Average", "Max", "Min", "Summary"])

            if choice == "Average":
                st.write(numeric_df.mean())

            elif choice == "Max":
                st.write(numeric_df.max())

            elif choice == "Min":
                st.write(numeric_df.min())

            elif choice == "Summary":
                st.write(df.describe(include="all"))

        else:
            st.warning("No numeric data found")

        # ---------------- VISUALIZATION ----------------
        st.subheader("📈 Data Visualization")

        if not numeric_df.empty:

            col1, col2 = st.columns(2)

            with col1:
                st.write("Correlation Heatmap")
                fig, ax = plt.subplots()
                sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)

            with col2:
                st.write("Distribution Charts")
                fig2, ax2 = plt.subplots()
                numeric_df.hist(ax=ax2)
                plt.tight_layout()
                st.pyplot(fig2)

        # ---------------- AI CHAT ----------------
        st.subheader("🤖 AI Data Analyst Chatbot")

        user_question = st.text_input("Ask anything about your dataset:")

        if user_question:

            summary = df.describe(include="all").to_string()

            prompt = f"""
Dataset:
{summary}

User Question:
{user_question}

Give simple and accurate answer.
"""

            with st.spinner("AI is thinking..."):
                answer = ask_ai(prompt)

            st.success("AI Response:")
            st.write(answer)

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Upload a CSV file to start analysis")
