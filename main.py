import os
import argparse
from pathlib import Path
from typing import List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings               
from langchain_chroma import Chroma
from langchain_community.document_loaders import (                     
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser



## ─────────────────────────  CONFIG  ─────────────────────────
DOCS_DIR         = Path("docs")                      
PERSIST_DIR      = Path("chroma_db")                 
EMBEDDING_MODEL  = "text-embedding-3-small"
CHAT_MODEL       = "gpt-4o"                          
CHUNK_SIZE       = 400
CHUNK_OVERLAP    = 40

## ─────────────────────────────────────────────────────────────

def ingest_docs() -> None:
    """
    1) Load all .md files from DOCS_DIR
    2) Split into semantic chunks
    3) Embed & persist to Chroma
    """
    loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
    )
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    docs: List = splitter.split_documents(raw_docs)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=str(PERSIST_DIR),
    )

    print(f"Ingested {len(docs)} chunks into {PERSIST_DIR}")


def build_chain() -> ConversationalRetrievalChain:
    """Return a ConversationalRetrievalChain wired to the persisted Chroma store."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectordb   = Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
    )

    retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "lambda_mult": 0.5, "score_threshold": 0.8},
)


    llm = ChatOpenAI(
        model_name=CHAT_MODEL,
        temperature=0.3,
    )
    system_prompt = SystemMessagePromptTemplate.from_template(
        """You are a helpful chatbot for a project management system that handles Leads and Jobs for Repair, Tiling, and Excavation work.

CRITICAL RULE: ONLY answer questions about the project, management system based on the provided context documents. If the question is not related to the system, respond with: "I can only help with questions about the project management system. Please ask about leads, jobs, equipment, or system processes.", when the quesions are about the system or the website or the project, it's probably related to and it's asking about this project.

BE A SMART CHATBOT:
- Give SHORT, intelligent answers (2-4 sentences max)
- Be conversational and friendly
- Infer information intelligently from context
- Answer what users actually want to know
- NEVER answer general knowledge questions

EQUIPMENT RULES (infer from context):
- Machines: Can be used in excavation and tiling jobs
- Vehicles: Only in excavation jobs  
- Trailers: Only in excavation jobs
- No equipment in repair jobs or any leads

RESPONSE STYLE:
- Keep answers brief and direct
- Use "You can..." or "To do this..."
- Give quick steps if needed: 1. Do X 2. Do Y
- Don't explain obvious things
- Focus on what matters most

CONTEXT DOCUMENTS:
{context}

Be smart, brief, and helpful like a real chatbot conversation ONLY about the project management system."""
    )
    human_prompt = HumanMessagePromptTemplate.from_template("{question}")

    prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt}, 
        verbose=False,
    )

def split_question_into_parts(question: str) -> List[str]:
    """Use GPT to split a compound question into multiple single-topic ones while preserving context."""
    llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0)
    prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are an expert at analyzing and decomposing user queries. Your task is to split compound questions into separate, standalone questions.

CRITICAL GUIDELINES:
- Split compound questions based on conjunctions (and, also, plus, additionally, furthermore, moreover)
- Each split question must be completely self-contained and include necessary context
- If the original question mentions a specific subject (like "tiling lead"), include that context in EVERY split question
- Ensure each split question can be answered independently
- If there's only one main topic, return it as a single question


Input: {question}

Return ONLY the questions as a comma-separated list with no extra text:"""
    )
    parser = CommaSeparatedListOutputParser()
    chain = prompt | llm | parser
    result = chain.invoke({"question": question})
    
    if not result or len(result) == 1:
        parts = question.split(" and ")
        if len(parts) > 1:
            base_context = ""
            if "tiling" in parts[0].lower():
                base_context = "tiling "
            elif "repair" in parts[0].lower():
                base_context = "repair "
            elif "excavation" in parts[0].lower():
                base_context = "excavation "
            
            split_questions = []
            for i, part in enumerate(parts):
                if i == 0:
                    split_questions.append(part.strip())
                else:
                    part = part.strip()
                    if base_context and base_context.strip() not in part.lower():
                        if part.startswith("how to"):
                            part = part.replace("how to", f"how to {base_context}", 1)
                        elif part.startswith("what"):
                            if "convert" in part:
                                part = part.replace("convert it to", f"convert {base_context}lead to")
                        elif part.startswith("which"):
                            if "can do" in part:
                                part = part.replace("this steps", f"the {base_context}lead steps")
                    split_questions.append(part)
            
            return split_questions
    
    return result

def enhance_response_authority(response):
    """Enhance the response to be more customer-friendly, natural, and comprehensive."""
    
    if "cannot answer" in response.lower() or "not available" in response.lower() or "don't have information" in response.lower():
        return response
    

    formal_phrases = [
        "According to the documentation, ",
        "This is explicitly stated in the documentation. ",
        "The system documentation specifies that ",
        "As documented in the system, ",
        "The documentation indicates ",
        "As outlined in the documentation, ",
        "The official procedure is: ",
        "Documented steps are: ",
        "System requirements include: ",
        "These are the documented requirements",
        "This process is fully documented",
        "This information is documented in the official system documentation."
    ]
    
    for phrase in formal_phrases:
        response = response.replace(phrase, "")
    
    response = response.replace("**", "")
    response = response.replace("- ", "")
    response = response.replace("* ", "")
    
    response = response.replace("The system requires", "You need to")
    response = response.replace("The documentation specifies", "")
    response = response.replace("official system procedures", "steps")
    response = response.replace("documented procedures", "steps")
    response = response.replace("cannot be added", "can't be used in")
    response = response.replace("can only be added", "can only be used in")
    

    response = response.strip()
    

    if any(keyword in response.lower() for keyword in ['equipment', 'machine', 'vehicle', 'trailer']):
        pass
    elif "can" in response.lower() and len(response.split()) < 10:
        pass
    
    verbose_phrases = [
        "you need to follow these steps:",
        "the process is as follows:",
        "the way to do this is:"
    ]
    
    for phrase in verbose_phrases:
        response = response.replace(phrase, "")
    
    lines = response.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if line and not any(skip_phrase in line.lower() for skip_phrase in [
            "that's it!", "you've successfully", "once you've filled"
        ]):
            cleaned_lines.append(line)
    
    response = '\n'.join(cleaned_lines)
    
    if not response:
        return "I can help you with that based on the system documentation."
    
    if not any(response.lower().startswith(starter) for starter in [
        "to", "you", "here", "first", "steps", "simply", "just", "when", "if", "1.", "yes", "no", "there are", "the system"
    ]):
        if any(word in response.lower()[:30] for word in ["click", "press", "go to", "navigate"]):
            response = "To do this: " + response
        elif response.lower().startswith("1."):
            response = "Steps:\n" + response
        elif any(word in response.lower()[:20] for word in ["machine", "vehicle", "trailer", "equipment"]):
            response = "Here's how equipment works: " + response
    
    return response


def chat(question: str) -> None:
    """
    Handles compound questions by splitting and answering each part independently.
    """
    if not PERSIST_DIR.exists():
        raise RuntimeError(
            f"No vector store found at {PERSIST_DIR}. Run `python main.py ingest` first."
        )

    chain = build_chain()

    try:
        sub_questions = split_question_into_parts(question)
    except Exception as e:
        print(f"[!] Failed to split question. Falling back to original: {e}")
        sub_questions = [question]

    for i, q in enumerate(sub_questions, 1):
        print(f"\n🔹 Q{i}: {q}")
        response = chain.invoke({"question": q, "chat_history": []})
        enhanced_response = enhance_response_authority(response["answer"])
        print("→",enhanced_response)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Docs-powered chatbot utility")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub_ingest = sub.add_parser("ingest", help="Load markdown docs & build vector DB")

    sub_chat = sub.add_parser("chat", help="Ask a test question")
    sub_chat.add_argument("--question", "-q", required=True, help="Prompt to send")

    args = parser.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")

    if args.cmd == "ingest":
        ingest_docs()
    elif args.cmd == "chat":
        chat(args.question)
