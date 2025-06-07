import os
import json
import faiss
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import glob
import xml.etree.ElementTree as ET

def create_faiss_index_from_xml(raw_data_dir, index_output_dir, model_name="all-MiniLM-L6-v2"):
    # Tạo thư mục đầu ra
    os.makedirs(index_output_dir, exist_ok=True)

    target_folders = ['1_CancerGov_QA']
    xml_files = []

    for folder in os.listdir(raw_data_dir):
        if folder in target_folders:
            folder_path = os.path.join(raw_data_dir, folder)
            if os.path.isdir(folder_path):
                xml_files.extend(glob.glob(os.path.join(folder_path, "*.xml")))

    print(f"Found {len(xml_files)} XML files")

    # Trích xuất tài liệu và cặp QA
    documents = []
    qa_data = []

    for xml_file in tqdm(xml_files, desc="Processing XML files"):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Lấy trọng tâm (chủ đề) của tài liệu
            focus = root.find("Focus").text

            # Xử lý từng cặp QA
            qa_pairs_element = root.find("QAPairs")
            if qa_pairs_element is not None:
                for qa_pair in qa_pairs_element.findall("QAPair"):
                    question_element = qa_pair.find("Question")
                    answer_element = qa_pair.find("Answer")

                    if question_element is not None and answer_element is not None:
                        question_text = question_element.text.strip()
                        answer_text = answer_element.text.strip()

                        # Tạo lời nhắc không có hậu tố "Trả lời:" cho chỉ mục tài liệu
                        document = f"Topic: {focus}\n\nQuestion: {question_text}"
                        documents.append(document)

                        # Lưu trữ cặp QA đầy đủ để truy xuất
                        qa_data.append({
                            "prompt": f"{document}\n\nAnswer:",
                            "completion": answer_text
                        })
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")

    print(f"Extracted {len(documents)} documents for indexing")

    # Tải mô hình chuyển đổi câu để mã hóa
    print(f"Loading sentence transformer model: {model_name}")
    model = SentenceTransformer(model_name)

    # Mã hóa tài liệu
    print("Encoding documents...")
    batch_size = 32
    document_embeddings = []

    for i in tqdm(range(0, len(documents), batch_size), desc="Encoding batches"):
        batch = documents[i:i + batch_size]
        embeddings = model.encode(batch, convert_to_tensor=True)
        document_embeddings.append(embeddings)

    # Nối tất cả các nhúng
    document_embeddings = torch.cat(document_embeddings, dim=0)
    document_embeddings_np = document_embeddings.cpu().numpy()

    # Tạo chỉ mục FAISS
    dimension = document_embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)

    # Thêm vectơ vào chỉ mục
    index.add(document_embeddings_np)
    print(f"Created FAISS index with {index.ntotal} vectors of dimension {dimension}")

    # Lưu chỉ mục, tài liệu và siêu dữ liệu
    faiss.write_index(index, os.path.join(index_output_dir, "qa_index1.faiss"))

    # Lưu tài liệu để truy xuất
    with open(os.path.join(index_output_dir, "documents1.json"), 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    # Lưu dữ liệu QA ánh xạ
    with open(os.path.join(index_output_dir, "qa_mapping1.json"), 'w', encoding='utf-8') as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=2)

    print(f"Saved FAISS index and metadata to {index_output_dir}")

    return index, documents, document_embeddings_np


# Cách sử dụng ví dụ
if __name__ == "__main__":
    create_faiss_index_from_xml(
        "data/raw",
        "data/faiss_index1"
    )
