import os
import json
import faiss
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def create_faiss_index_from_xml(raw_data_dir, index_output_dir, model_name="all-MiniLM-L6-v2"):
    """
    Process XML files and create a FAISS index directly without an intermediate JSON file.

    Args:
        raw_data_dir: Path to directory containing raw XML data folders
        index_output_dir: Directory to save the FAISS index and metadata
        model_name: Name of the sentence transformer model to use for encoding

    Returns:
        Tuple of (faiss_index, documents, document_embeddings)
    """
    import glob
    import xml.etree.ElementTree as ET

    # Create output directory
    os.makedirs(index_output_dir, exist_ok=True)

    # Get all XML files recursively
    xml_files = []
    for folder in os.listdir(raw_data_dir):
        folder_path = os.path.join(raw_data_dir, folder)
        if os.path.isdir(folder_path):
            xml_files.extend(glob.glob(os.path.join(folder_path, "*.xml")))

    print(f"Found {len(xml_files)} XML files")

    # Extract documents and QA pairs
    documents = []
    qa_data = []

    for xml_file in tqdm(xml_files, desc="Processing XML files"):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Get the focus (topic) of the document
            focus = root.find("Focus").text

            # Process each QA pair
            qa_pairs_element = root.find("QAPairs")
            if qa_pairs_element is not None:
                for qa_pair in qa_pairs_element.findall("QAPair"):
                    question_element = qa_pair.find("Question")
                    answer_element = qa_pair.find("Answer")

                    if question_element is not None and answer_element is not None:
                        question_text = question_element.text.strip()
                        answer_text = answer_element.text.strip()

                        # Create prompt without "Answer:" suffix for document index
                        document = f"Topic: {focus}\n\nQuestion: {question_text}"
                        documents.append(document)

                        # Store full QA pair for retrieval
                        qa_data.append({
                            "prompt": f"{document}\n\nAnswer:",
                            "completion": answer_text
                        })
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")

    print(f"Extracted {len(documents)} documents for indexing")

    # Load sentence transformer model for encoding
    print(f"Loading sentence transformer model: {model_name}")
    model = SentenceTransformer(model_name)

    # Encode documents
    print("Encoding documents...")
    batch_size = 32
    document_embeddings = []

    for i in tqdm(range(0, len(documents), batch_size), desc="Encoding batches"):
        batch = documents[i:i + batch_size]
        embeddings = model.encode(batch, convert_to_tensor=True)
        document_embeddings.append(embeddings)

    # Concatenate all embeddings
    document_embeddings = torch.cat(document_embeddings, dim=0)
    document_embeddings_np = document_embeddings.cpu().numpy()

    # Create FAISS index
    dimension = document_embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)

    # Add vectors to the index
    index.add(document_embeddings_np)
    print(f"Created FAISS index with {index.ntotal} vectors of dimension {dimension}")

    # Save the index, documents and metadata
    faiss.write_index(index, os.path.join(index_output_dir, "qa_index.faiss"))

    # Save documents for retrieval
    with open(os.path.join(index_output_dir, "documents.json"), 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    # Save QA data mapping
    with open(os.path.join(index_output_dir, "qa_mapping.json"), 'w', encoding='utf-8') as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=2)

    print(f"Saved FAISS index and metadata to {index_output_dir}")

    return index, documents, document_embeddings_np


# Example usage
if __name__ == "__main__":
    create_faiss_index_from_xml(
        "data/raw",
        "data/faiss_index"
    )