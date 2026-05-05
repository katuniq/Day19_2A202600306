import json
import os
import time
from neo4j import GraphDatabase
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

# ---------------------------------------------------------
# CẤU HÌNH KẾT NỐI
# ---------------------------------------------------------
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Đường dẫn dữ liệu
CORPUS_PATH = "data/corpus.json"
TRIPLES_PATH = "data/triples.json"

# ---------------------------------------------------------
# BƯỚC 2: XÂY DỰNG ĐỒ THỊ (Graph Construction với Neo4j)
# ---------------------------------------------------------
class KnowledgeGraphManager:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def build_graph_from_triples(self, triples_file):
        """Xóa đồ thị cũ và chèn toàn bộ triples vào Neo4j"""
        print("[Neo4j] Đang đọc file triples...")
        with open(triples_file, 'r', encoding='utf-8') as f:
            triples = json.load(f)
        
        start_time = time.time()
        with self.driver.session() as session:
            # Xóa sạch dữ liệu hiện tại
            print("[Neo4j] Xóa đồ thị cũ...")
            session.run("MATCH (n) DETACH DELETE n")
            
            print("[Neo4j] Bắt đầu Import các node và quan hệ...")
            for idx, t in enumerate(triples):
                head = t['head']
                # Định dạng lại tên quan hệ cho chuẩn Neo4j (viết hoa, gạch dưới)
                rel = t['relation'].upper().replace(' ', '_').replace('-', '_')
                tail = t['tail']
                
                query = f"""
                MERGE (h:Entity {{name: $head}})
                MERGE (t:Entity {{name: $tail}})
                MERGE (h)-[:`{rel}`]->(t)
                """
                session.run(query, head=head, tail=tail)
                
                if (idx + 1) % 100 == 0:
                    print(f"  Đã import {idx + 1} triples...")
                    
        elapsed = time.time() - start_time
        print(f"[Neo4j] Xây dựng đồ thị hoàn tất! Mất {elapsed:.2f} giây.")

    def get_context_for_entity(self, entity_name):
        """Lấy ngữ cảnh xung quanh 1 node (2-hop)"""
        with self.driver.session() as session:
            # 1-hop
            query1 = """
            MATCH (start:Entity)-[r]-(end:Entity)
            WHERE toLower(start.name) CONTAINS toLower($entity_name)
            RETURN start.name as e1, type(r) as rel, end.name as e2
            LIMIT 20
            """
            res1 = session.run(query1, entity_name=entity_name)
            context = [f"{rec['e1']} {rec['rel']} {rec['e2']}" for rec in res1]
            
            # 2-hop
            query2 = """
            MATCH (start:Entity)-[r1]-(mid:Entity)-[r2]-(end:Entity)
            WHERE toLower(start.name) CONTAINS toLower($entity_name) AND start <> end
            RETURN start.name as e1, type(r1) as rel1, mid.name as e2, type(r2) as rel2, end.name as e3
            LIMIT 20
            """
            res2 = session.run(query2, entity_name=entity_name)
            context.extend([f"{rec['e1']} {rec['rel1']} {rec['e2']} and {rec['e2']} {rec['rel2']} {rec['e3']}" for rec in res2])
            
            return list(set(context))

# ---------------------------------------------------------
# BƯỚC 3 & 4: FLAT RAG VÀ GRAPHRAG (Truy vấn)
# ---------------------------------------------------------
def setup_flat_rag(corpus_file):
    print("[FlatRAG] Đang tạo vector store bằng ChromaDB...")
    with open(corpus_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    docs = []
    for item in data:
        chunks = item['text'].split('\n\n')
        for chunk in chunks:
            if len(chunk.strip()) > 20:
                docs.append(Document(page_content=chunk, metadata={"company": item['company']}))
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    # Khởi tạo VectorStore (tạo collection tạm trong memory hoặc thư mục)
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, collection_name="flat_rag_test")
    print("[FlatRAG] Tạo vector store hoàn tất.")
    return vectorstore

def answer_with_flat_rag(vectorstore, llm, question):
    # Lấy top 3 chunks liên quan nhất
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n---\n".join([d.page_content for d in docs])
    
    prompt = f"""Bạn là một trợ lý ảo thông minh. Dựa vào thông tin văn bản dưới đây:
{context}

Hãy trả lời câu hỏi: {question}
Nếu không có thông tin, hãy nói "Tôi không biết dựa trên văn bản được cung cấp"."""
    
    return llm.invoke(prompt).content

def answer_with_graph_rag(kg_manager, llm, question, key_entities):
    """
    key_entities: Danh sách thực thể chính rút trích từ câu hỏi. 
    (Trong thực tế sẽ dùng LLM để trích xuất tự động, ở đây truyền sẵn mảng string cho bài lab)
    """
    context_lines = []
    for entity in key_entities:
        paths = kg_manager.get_context_for_entity(entity)
        context_lines.extend(paths)
    
    context_str = "\n".join(context_lines[:50]) # Giới hạn số lượng path
    
    prompt = f"""Bạn là một trợ lý ảo phân tích đồ thị tri thức (Knowledge Graph).
Dưới đây là các mối quan hệ đồ thị liên quan đến thực thể trong câu hỏi:
{context_str}

Hãy trả lời câu hỏi: {question}
Trả lời bằng ngôn ngữ tự nhiên và chỉ dựa vào thông tin đồ thị trên."""
    
    return llm.invoke(prompt).content

# ---------------------------------------------------------
# CHẠY ĐÁNH GIÁ (EVALUATION)
# ---------------------------------------------------------
def run_evaluation():
    # 1. Khởi tạo LLM
    try:
        llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0)
    except Exception as e:
        print("Lỗi khởi tạo OpenAI. Bạn đã điền API Key chưa?")
        return

    # 2. Xây dựng Knowledge Graph trên Neo4j
    kg = KnowledgeGraphManager(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        kg.build_graph_from_triples(TRIPLES_PATH)
    except Exception as e:
        print(f"Lỗi kết nối Neo4j: {e}")
        print("Vui lòng kiểm tra lại Username/Password Neo4j Desktop.")
        return
    
    # 3. Xây dựng Flat RAG
    vectorstore = setup_flat_rag(CORPUS_PATH)

    # 4. Danh sách 5 câu hỏi benchmark để test
    test_cases = [
        {
            "q": "Ai là những người đồng sáng lập của OpenAI và công ty này được định giá bao nhiêu vào năm 2025?",
            "entities": ["OpenAI", "2025"]
        },
        {
            "q": "Sundar Pichai giữ chức vụ gì ở Google và Alphabet?",
            "entities": ["Sundar Pichai", "Google", "Alphabet"]
        },
        {
            "q": "Microsoft đã mua lại những công ty nào trong quá khứ?",
            "entities": ["Microsoft"]
        },
        {
            "q": "Meta Platforms sở hữu những ứng dụng mạng xã hội nào?",
            "entities": ["Meta Platforms"]
        },
        {
            "q": "Apple được định giá 4 nghìn tỷ đô la vào thời điểm nào và CEO hiện tại là ai?",
            "entities": ["Apple", "4 nghìn tỷ"]
        }
    ]

    print("\n" + "="*50)
    print("BẮT ĐẦU SO SÁNH FLAT RAG VÀ GRAPHRAG")
    print("="*50)
    
    for i, case in enumerate(test_cases, 1):
        question = case["q"]
        entities = case["entities"]
        print(f"\n[Câu hỏi {i}]: {question}")
        print(f"  -> Các thực thể nhận diện (để Graph tìm): {entities}")
        
        # Flat RAG
        flat_ans = answer_with_flat_rag(vectorstore, llm, question)
        print(f"\n--- [Flat RAG Output] ---\n{flat_ans}")
        
        # Graph RAG
        graph_ans = answer_with_graph_rag(kg, llm, question, entities)
        print(f"\n--- [Graph RAG Output] ---\n{graph_ans}")
        print("-" * 50)
        
    kg.close()

if __name__ == "__main__":
    print("Lưu ý: Để chạy được script này, cần có:")
    print("1. Neo4j Desktop đang mở (port 7687)")
    print("2. OPENAI_API_KEY hợp lệ")
    print("\nĐang bắt đầu pipeline...")
    run_evaluation()
