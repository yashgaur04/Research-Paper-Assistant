import arxiv
import heapq
import os
import unittest
from typing import List, Dict
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_community.chat_models import AzureChatOpenAI
from langchain_community.llms import OpenAI
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

def get_openai_api_key():
    """ Prompt user for OpenAI API key if not set in environment variables. """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Enter your OpenAI API Key: ")
        os.environ["OPENAI_API_KEY"] = api_key
    return api_key

def fetch_top_arxiv_papers(query: str, year_range: tuple, top_n: int, search_query: str = "", selected_journal: str = "All", selected_author: str = "All", sort_by: str = "relevance", selected_paper_type: str = "All") -> List[ResearchPaper]:
    """
    Fetch top research papers from arXiv based on a user-specified query.
    """
    try:
        search_results = arxiv.Search(
            query=query,
            max_results=top_n,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
    except Exception as e:
        print(f"Error fetching papers: {e}")
        return []
    
    papers = []
    
    for paper in search_results.results():
        pub_year = paper.published.year if paper.published else 0
        paper_type = "Preprint"  # arXiv is mostly preprints
        
        # Filter based on year range
        if year_range[0] <= pub_year <= year_range[1]:
            research_paper = ResearchPaper(
                title=paper.title,
                authors=", ".join(author.name for author in paper.authors),
                year=pub_year,
                abstract=paper.summary,
                citations=0,  # arXiv does not provide citation counts
                url=paper.entry_id,
                journal="arXiv",
                publisher="arXiv",
                paper_type=paper_type
            )
            papers.append(research_paper)
    
    return papers[:top_n]

def store_papers_in_vector_db(papers: List[ResearchPaper]):
    """ Store research papers in a FAISS vector database for retrieval. """
    if not papers:
        return None
    documents = [Document(page_content=paper.abstract, metadata=paper.dict()) for paper in papers]
    embeddings = HuggingFaceEmbeddings()
    vector_db = FAISS.from_documents(documents, embeddings)
    return vector_db

def setup_langchain_qa(vector_db, papers: List[ResearchPaper]):
    """ Set up a retrieval-based QA chatbot using LangChain with OpenAI. """
    api_key = get_openai_api_key()
    llm = AzureChatOpenAI(
        deployment_name="gpt-4-turbo",
        model_name="gpt-4",
        openai_api_base="https://your-azure-endpoint.com/",
        openai_api_version="2023-05-15",
        openai_api_key=api_key
    )
    
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_db.as_retriever())
    return qa_chain, papers

def format_citation(text: str, papers: List[ResearchPaper]) -> str:
    """ Format the chatbot's response to include superscript citations with hyperlinks. """
    for i, paper in enumerate(papers, start=1):
        citation = f'<sup><a href="{paper.url}" title="{paper.title} ({paper.year}) - {paper.journal}">[{i}]</a></sup>'
        text = re.sub(f'\b{re.escape(paper.title[:10])}.*?\b', f'\g<0>{citation}', text, flags=re.IGNORECASE)
    return text

class TestResearchAssistant(unittest.TestCase):
    def test_fetch_top_arxiv_papers(self):
        papers = fetch_top_arxiv_papers("AI", (2018, 2023), 5)
        self.assertTrue(len(papers) <= 5)
        for paper in papers:
            self.assertTrue(2018 <= paper.year <= 2023)

    def test_store_papers_in_vector_db(self):
        papers = [ResearchPaper(title="Sample Paper", authors="Author Name", year=2022, abstract="This is a test.", citations=0, url="http://test.com", journal="arXiv", publisher="arXiv", paper_type="Preprint")]
        vector_db = store_papers_in_vector_db(papers)
        self.assertIsNotNone(vector_db)
    
    def test_format_citation(self):
        papers = [ResearchPaper(title="Test Paper", authors="Author Name", year=2022, abstract="", citations=0, url="http://test.com", journal="arXiv", publisher="arXiv", paper_type="Preprint")]
        text = "This is a reference to Test Paper."
        formatted_text = format_citation(text, papers)
        self.assertIn("<sup><a href=\"http://test.com\"", formatted_text)

if __name__ == "__main__":
    unittest.main()
