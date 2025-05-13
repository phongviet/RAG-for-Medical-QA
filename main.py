import os
import json
import faiss
import torch
import argparse
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


class MedicalQASystem:
    def __init__(self, index_dir, model_name="Qwen/Qwen2.5-0.5B-Instruct",
                 encoder_name="all-MiniLM-L6-v2", top_k=3):
        """
        Initialize the Medical QA system with RAG components

        Args:
            index_dir: Directory containing the FAISS index and document mappings
            model_name: LLM to use for generating answers
            encoder_name: Sentence encoder model for embedding queries
            top_k: Number of relevant documents to retrieve
        """
        self.index_dir = index_dir
        self.top_k = top_k

        # Load document mappings
        with open(os.path.join(index_dir, "documents.json"), "r") as f:
            self.documents = json.load(f)

        with open(os.path.join(index_dir, "qa_mapping.json"), "r") as f:
            self.qa_mapping = json.load(f)

        # Load FAISS index
        self.index = faiss.read_index(os.path.join(index_dir, "qa_index.faiss"))
        print(f"Loaded FAISS index with {self.index.ntotal} vectors")

        # Load sentence encoder model
        print(f"Loading sentence encoder: {encoder_name}")
        self.encoder = SentenceTransformer(encoder_name)

        # Load language model for answer generation
        print(f"Loading language model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        print("Medical QA system initialized")

    def retrieve(self, query):
        """Retrieve relevant documents for a query"""
        # Encode the query
        query_embedding = self.encoder.encode([query], convert_to_tensor=True)
        query_embedding_np = query_embedding.cpu().numpy()

        # Search in FAISS index
        distances, indices = self.index.search(query_embedding_np, self.top_k)

        # Get the corresponding documents
        retrieved_docs = [self.documents[idx] for idx in indices[0]]
        retrieved_answers = [self.qa_mapping[idx]["completion"] for idx in indices[0]]

        return list(zip(retrieved_docs, distances[0], retrieved_answers))

    def format_prompt(self, query, retrieved_docs):
        """Format prompt with retrieved context"""
        # Prepare context from retrieved documents in a structured way
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            # Extract topic from document
            topic_part = doc[0].split("\n\n")[0].replace("Topic: ", "")
            # Include retrieved answer as reference
            context_parts.append(f"Reference {i + 1} on {topic_part}:\n{doc[2]}")

        context_str = "\n\n".join(context_parts)

        prompt = f"""You are a highly knowledgeable medical assistant providing accurate and professional information to healthcare professionals.
Use the reference information below to answer the medical question comprehensively.

Reference Information:
{context_str}

Question: {query}

Instructions:
1. Provide a clear, concise, and well-organized medical answer.
2. Use appropriate medical terminology and maintain a professional tone.
3. Base your answer strictly on the provided reference information.
4. Do not include any citation numbers, references, or footnote markers (e.g., [1], [2], etc.).
5. Do not mention the references explicitly in your answer.
6. Avoid adding phrases like 'End of Medical Answer' or similar concluding statements.

Always conclude your answer with this exact disclaimer:
"Please consult with a qualified healthcare professional for accurate diagnosis and personalized medical advice."

Answer:"""
        return prompt

    def generate_answer(self, prompt):
        """Generate answer using LLM"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.5,
                do_sample=True,
                top_p=0.85,
                num_return_sequences=1
            )

        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()

    def answer_question(self, query):
        """End-to-end question answering"""
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query)

        # Format prompt with retrieved context
        prompt = self.format_prompt(query, retrieved_docs)

        # Generate answer
        answer = self.generate_answer(prompt)

        return {
            "query": query,
            "retrieved_documents": [doc[0] for doc in retrieved_docs],
            "answer": answer
        }


def main():
    parser = argparse.ArgumentParser(description="Medical QA System")
    parser.add_argument("--index_dir", default="data/faiss_index", help="Directory with FAISS index")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="LLM model to use (default: Qwen/Qwen2.5-0.5B-Instruct)")
    parser.add_argument("--query", type=str, help="Question to answer (if not provided, runs in interactive mode)")

    args = parser.parse_args()

    # Initialize QA system
    qa_system = MedicalQASystem(
        index_dir=args.index_dir,
        model_name=args.model
    )

    # If no query argument provided, run in interactive mode by default
    if args.query:
        result = qa_system.answer_question(args.query)
        print("\n" + "-" * 80)
        print(f"QUERY: {result['query']}")
        print("\nRETRIEVED DOCUMENTS:")
        for i, doc in enumerate(result['retrieved_documents']):
            print(f"{i + 1}. {doc[:100]}...")
        print("\nANSWER:")
        print(result['answer'])
        print("-" * 80)
    else:
        # Interactive mode is the default behavior now
        print("Medical QA System (type 'exit' to quit)")
        print(f"Using model: {args.model}")
        print(f"Using index: {args.index_dir}")
        print()

        while True:
            query = input("\nEnter your medical question: ")
            if query.lower() == "exit":
                break

            try:
                result = qa_system.answer_question(query)
                print("\n" + "-" * 80)
                print("RETRIEVED DOCUMENTS:")
                for i, doc in enumerate(result['retrieved_documents']):
                    print(f"{i + 1}. {doc[:100]}...")
                print("\nANSWER:")
                print(result['answer'])
                print("-" * 80)
            except Exception as e:
                print(f"Error processing query: {e}")

if __name__ == "__main__":
    main()