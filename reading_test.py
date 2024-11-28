from langchain_community.document_loaders import PyMuPDFLoader

# Specify the path to your PDF file
pdf_path = "./pdf/cai.pdf"

# Create a PDFPlumberLoader instance
loader = PyMuPDFLoader(pdf_path)

try:
    # Load and parse the PDF file
    documents = loader.load()
    
    # Print the content of the documents
    for doc in documents:
        print("Content:", doc.page_content)
        print("Metadata:", doc.metadata)
        print("-" * 50)

except Exception as e:
    print(f"Error loading PDF: {e}")