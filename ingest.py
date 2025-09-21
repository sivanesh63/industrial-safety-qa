import os
import re
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# Set up the tools
model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="safety_db")
collection = client.get_or_create_collection(name="safety_docs")

def clean_text(text):
    """Clean text by removing extra spaces."""
    if not text:
        return ""
    clean = re.sub(r'\s+', ' ', text.strip())
    return clean

def split_into_chunks(text, chunk_size=300):
    """Split text into smaller chunks."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) > 50:
            chunks.append(chunk)
    
    return chunks

def process_one_pdf(pdf_file_path, document_id):
    """Process a single PDF file."""
    
    try:
        reader = PdfReader(pdf_file_path)
        all_text = ""
        
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                all_text += page_text + "\n"
        
        if not all_text.strip():
            return 0
        
        clean = clean_text(all_text)
        chunks = split_into_chunks(clean)
        
        if not chunks:
            print(f"No chunks created from {pdf_file_path}")
            return 0
        
        
        chunk_ids = []
        chunk_embeddings = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document_id}_chunk_{i}"
            chunk_ids.append(chunk_id)
            embedding = model.encode(chunk).tolist()
            chunk_embeddings.append(embedding)
        
        collection.add(
            ids=chunk_ids,
            documents=chunks,
            embeddings=chunk_embeddings,
            metadatas=[{"source": document_id, "chunk_number": i} for i in range(len(chunks))]
        )
        
        return len(chunks)
        
    except Exception as error:
        return 0

def process_all_pdfs():
    """Process all PDF files in the folder."""
    pdf_folder = "data\industrial-safety-pdfs"
    
    if not os.path.exists(pdf_folder):
        print(f"Folder '{pdf_folder}' not found!")
        return
    
    pdf_files = []
    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith('.pdf'):
            pdf_files.append(filename)
    
    if not pdf_files:
        return
    
    
    total_chunks = 0
    successful_files = 0
    
    for i, filename in enumerate(pdf_files):
        
        full_path = os.path.join(pdf_folder, filename)
        document_id = f"doc_{i+1}"
        
        chunks_created = process_one_pdf(full_path, document_id)
        
        if chunks_created > 0:
            successful_files += 1
            total_chunks += chunks_created
    

    
    if successful_files < len(pdf_files):
        failed = len(pdf_files) - successful_files

if __name__ == "__main__":
    try:
        process_all_pdfs()
        
    except Exception as error:
        print(f"Program failed: {error}")