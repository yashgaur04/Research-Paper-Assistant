import streamlit as st
import pandas as pd
import os
from arxivFetch import fetch_top_arxiv_papers, store_papers_in_vector_db, setup_langchain_qa, format_citation
import openai
import time


st.title("arxiv Research Assistant")

# User inputs
query = st.text_input("Enter Research Topic:")
year_range = st.slider("Select Year Range:", 2000, 2025, (2018, 2023))
top_n = st.selectbox("Select Number of Top Papers:", [5, 10, 25])

# API Key input
api_key = st.text_input("Enter OpenAI API Key:", type="password")
os.environ["OPENAI_API_KEY"] = api_key if api_key else os.getenv("OPENAI_API_KEY")

# Model selection
target_model = st.selectbox("Select AI Model:", ["gpt-4-turbo"]) # WIP: more models to be added in next version "llama", "mistral", "claude"])

if st.button("Retrieve Research Papers"):
    with st.spinner("Fetching top research papers from arXiv..."):
        papers = fetch_top_arxiv_papers(query, year_range, top_n)
        vector_db = store_papers_in_vector_db(papers)
        qa_chain, papers = setup_langchain_qa(vector_db, papers)
        st.session_state["qa_chain"] = qa_chain
        st.session_state["papers"] = papers
        st.success("Papers Retrieved Successfully!")
    
    # Convert papers to DataFrame for export
    df = pd.DataFrame([paper.dict() for paper in papers])
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Research Papers", data=csv, file_name="research_papers.csv", mime="text/csv")
    
    # Search within retrieved papers
    search_query = st.text_input("Chat with these research papers:")
    filtered_papers = [paper for paper in papers if search_query.lower() in paper.title.lower() or search_query.lower() in paper.authors.lower()]
    
    # Filters for journal and authors
    unique_journals = sorted(set(paper.journal for paper in papers if paper.journal))
    selected_journal = st.selectbox("Filter by Journal:", ["All"] + unique_journals)
    if selected_journal != "All":
        filtered_papers = [paper for paper in filtered_papers if paper.journal == selected_journal]
    
    unique_authors = sorted(set(paper.authors for paper in papers if paper.authors))
    selected_author = st.selectbox("Filter by Author:", ["All"] + unique_authors)
    if selected_author != "All":
        filtered_papers = [paper for paper in filtered_papers if paper.authors == selected_author]
    
    # Paginate research paper display
    st.subheader("Research Papers")
    page_size = 5
    num_pages = len(filtered_papers) // page_size + (1 if len(filtered_papers) % page_size > 0 else 0)
    page_num = st.slider("Select Page:", 1, max(num_pages, 1), 1)
    start_idx = (page_num - 1) * page_size
    end_idx = start_idx + page_size
    
    for paper in filtered_papers[start_idx:end_idx]:
        st.markdown(f"**{paper.title}** ({paper.year})")
        st.markdown(f"*{paper.authors}* - {paper.journal}")
        st.markdown(f"[Read Paper]({paper.url})")
        st.markdown("---")

if "qa_chain" in st.session_state:
    st.subheader("Chat with AI")
    user_query = st.text_input("Ask a question about the research topic:")
    if st.button("Get Answer"):
        with st.spinner("Generating response..."):
            response = st.session_state["qa_chain"].run(user_query)
            formatted_response = format_citation(response, st.session_state["papers"])
            st.write(formatted_response, unsafe_allow_html=True)
