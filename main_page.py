import streamlit as st

st.set_page_config(page_title="nlp-notebook", page_icon="🌎")
# sidebar --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
st.sidebar.info("Select a page above.")

# main page --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
st.header("📚 NLP Notebook",divider = "rainbow")

"""

I hope to post little projects along with my **nlp study** journey. Feel free to fork my website!

#### :blue[Menu]
- `🎬 Sentiment Analysis`: Movie sentiment analysis using **Naive Bayes**
"""


# column --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

st.divider()

st.write("Also, check out my ***personal portofolio!***")
st.link_button("Personal Portofolio", "https://harrycheng.streamlit.app")




