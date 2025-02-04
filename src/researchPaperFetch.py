from scholarly import scholarly
from scholarly import ProxyGenerator
from typing import List, Dict
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import LlamaCpp, OpenAI, Anthropic
import re

class ResearchPaper(BaseModel):
    title: str
    authors: str
    year: int
    abstract: str
    citations: int
    url: str
    journal: str
    publisher: str
    paper_type: str

def fetch_top_scholar_papers(query: str, year_range: tuple, top_n: int, search_query: str = "", selected_journal: str = "All", selected_author: str = "All", sort_by: str = "citations", selected_paper_type: str = "All") -> List[ResearchPaper]:
    """
    Fetch top cited research papers from Google Scholar based on a user-specified query.
    """
    pg = ProxyGenerator()
    pg.FreeProxies()  # Use free proxies
    scholarly.use_proxy(pg)
    search_results = scholarly.search_pubs(query)
    papers = []
    
    for paper in search_results:
        bib = paper.get('bib', {})
        pub_year = int(bib.get('year', 0))
        citation_count = paper.get('num_citations', 0)
        paper_type = "Journal" if "journal" in bib else "Conference"
        
        # Filter based on year range
        if year_range[0] <= pub_year <= year_range[1]:
            research_paper = ResearchPaper(
                title=bib.get('title', 'N/A'),
                authors=bib.get('author', 'N/A'),
                year=pub_year,
                abstract=bib.get('abstract', 'No abstract available'),
                citations=citation_count,
                url=paper.get('pub_url', 'N/A'),
                journal=bib.get('journal', 'N/A'),
                publisher=bib.get('publisher', 'N/A'),
                paper_type=paper_type
            )
            papers.append(research_paper)
    
    # Apply search and filter conditions
    if search_query:
        papers = [p for p in papers if search_query.lower() in p.title.lower() or search_query.lower() in p.authors.lower()]
    if selected_journal != "All":
        papers = [p for p in papers if p.journal == selected_journal]
    if selected_author != "All":
        papers = [p for p in papers if p.authors == selected_author]
    if selected_paper_type != "All":
        papers = [p for p in papers if p.paper_type == selected_paper_type]
    
    # Sort by selected criteria
    if sort_by == "relevance":
        top_papers = sorted(papers, key=lambda x: (x.year, x.citations), reverse=True)[:top_n]
    else:
        top_papers = sorted(papers, key=lambda x: x.citations, reverse=True)[:top_n]
    
    return top_papers

def store_papers_in_vector_db(papers: List[ResearchPaper]):
    """ Store research papers in a FAISS vector database for retrieval. """
    documents = [Document(page_content=paper.abstract, metadata={
        "title": paper.title,
        "authors": paper.authors,
        "year": paper.year,
        "citations": paper.citations,
        "url": paper.url,
        "journal": paper.journal,
        "publisher": paper.publisher,
        "paper_type": paper.paper_type
    }) for paper in papers]
    
    embeddings = HuggingFaceEmbeddings()
    vector_db = FAISS.from_documents(documents, embeddings)
    return vector_db

def setup_langchain_qa(vector_db, papers: List[ResearchPaper], model_choice: str = "gpt-4-turbo"):
    """ Set up a retrieval-based QA chatbot using LangChain with multi-model support. """
    if model_choice == "llama":
        llm = LlamaCpp(model_path="path/to/llama/model.bin")
    elif model_choice == "mistral":
        llm = OpenAI(model_name="mistral-7b")
    elif model_choice == "claude":
        llm = Anthropic(model_name="claude-2")
    else:
        llm = ChatOpenAI(model_name="gpt-4-turbo")
    
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_db.as_retriever())
    return qa_chain, papers

# Example Usage
if __name__ == "__main__":
    topic = "Artificial Intelligence in Healthcare"
    year_range = (2018, 2023)
    top_n = 5
    search_query = "Deep Learning"
    selected_journal = "Nature"
    selected_author = "John Doe"
    sort_by = "relevance"
    selected_paper_type = "Journal"
    model_choice = "llama"  # Options: llama, mistral, claude, gpt-4-turbo
    
    results = fetch_top_scholar_papers(topic, year_range, top_n, search_query, selected_journal, selected_author, sort_by, selected_paper_type)
    vector_db = store_papers_in_vector_db(results)
    qa_chain, papers = setup_langchain_qa(vector_db, results, model_choice)
    
    user_query = "What are the latest advancements in AI for healthcare?"
    response = qa_chain.run(user_query)
    
    print("Chatbot Response:", response)
    
    for idx, paper in enumerate(results, start=1):
        print(f"{idx}. {paper.title} ({paper.year}) - Citations: {paper.citations}")
        print(f"   Journal: {paper.journal}, Publisher: {paper.publisher}, Type: {paper.paper_type}")
        print(f"   URL: {paper.url}\n")
