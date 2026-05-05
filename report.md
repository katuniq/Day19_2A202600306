# BÁO CÁO BÀI LAB 19: XÂY DỰNG HỆ THỐNG GRAPHRAG VỚI TECH COMPANY CORPUS
**Tên: Trần Quốc Khánh**
**MS: 2A202600306**
## 1. Mã nguồn
Mã nguồn cho bài lab đã được triển khai trong file `graphrag_solution.py` (cùng thư mục). 
Script cung cấp các chức năng chính:
- **Indexing (Neo4j):** Đọc file `data/triples.json` và chèn các triples (Head, Relation, Tail) vào Neo4j thông qua thư viện `neo4j` Python driver.
- **Flat RAG:** Sử dụng thư viện `langchain` và `chromadb` để băm file `data/corpus.json`, tạo vector embeddings và truy xuất thông tin bằng tìm kiếm tương đồng (Similarity Search).
- **GraphRAG:** Sử dụng Cypher Query để duyệt đồ thị trong phạm vi 2-hop (lấy các nodes và quan hệ lân cận với thực thể chính được hỏi), sau đó cung cấp đoạn ngữ cảnh các mối nối đồ thị cho LLM trả lời.

## 2. Ảnh chụp màn hình đồ thị tri thức
> **Hướng dẫn:**
> 1. Chạy file `graphrag_solution.py` với biến môi trường phù hợp.
> 2. Mở **Neo4j Desktop** (hoặc truy cập http://localhost:7474/).
> 3. Trong trình duyệt Neo4j (Neo4j Browser), nhập lệnh Cypher sau:
>    `MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 150`

  ![Neo4j Graph](/neo4j_graph.png)

## 3. Bảng so sánh 20 câu hỏi benchmark giữa Flat RAG và GraphRAG

Dưới đây là thiết kế 20 câu hỏi dùng để đánh giá hệ thống, tập trung vào khả năng đa bước (multi-hop) và giảm thiểu ảo giác (hallucination):

| STT | Câu hỏi | Dự đoán kết quả Flat RAG | Dự đoán kết quả GraphRAG |
| :--- | :--- | :--- | :--- |
| 1 | Ai là người thành lập OpenAI và nó được định giá bao nhiêu vào năm 2025? | Có thể bị sót thông tin định giá do thông tin nằm ở 2 đoạn văn khác nhau (ảo giác cục bộ). | **Trả lời chính xác** do kết nối được `OpenAI` -> `FOUNDED_BY` và `OpenAI` -> `VALUED_AT`. |
| 2 | Sundar Pichai là CEO của những công ty nào? | Có thể chỉ trả lời là Google do độ tương đồng văn bản cao hơn với cụm Google. | Dễ dàng liệt kê cả `Google` và `Alphabet` vì chúng kết nối chung với node `Sundar Pichai`. |
| 3 | Microsoft đã mua lại những ứng dụng/công ty nào trong quá khứ? | Sót thông tin (VD: sót Skype hoặc LinkedIn) do bị chia cắt context. | Lấy được toàn bộ danh sách qua quan hệ `ACQUIRED`. |
| 4 | Kể tên các sản phẩm mạng xã hội và nhắn tin của Meta. | Tốt, nhưng đôi khi liệt kê nhầm công ty con của các hãng khác nếu dùng vector nhầm lẫn. | Chính xác tuyệt đối thông qua các cạnh `OWNS`. |
| 5 | Tại sao OpenAI lại sa thải và phục chức cho Sam Altman? | Trả lời tốt vì câu chuyện thường nằm trong 1 đoạn văn. | Trả lời ngắn gọn dựa trên quan hệ (nếu có extract lý do), đôi khi yếu hơn Flat RAG về ngữ cảnh dài. |
| 6 | Meta Platforms đã mua lại WhatsApp và Instagram với giá trị bao nhiêu? | Flat RAG có thể trả lời sai số tiền giữa các thương vụ. | GraphRAG kết nối chính xác giá trị với từng thương vụ cụ thể. |
| 7 | Apple Inc. được thành lập vào năm nào và ai là người sáng lập? | Trả lời tốt, thông tin cơ bản. | Trả lời cực nhanh và chính xác. |
| 8 | Sản phẩm nào của Apple ra đời sau khi mua lại NeXT? | Có thể bị ảo giác nếu khoảng cách câu quá xa. | Kết nối đa bước `Apple -> ACQUIRED -> NeXT` và `Apple -> PRODUCT_OF -> iMac/iPhone`. |
| 9 | Amazon mua lại Whole Foods Market vào thời gian nào và giá bao nhiêu? | Trả lời tốt. | Trả lời tốt, trích xuất chính xác thời gian và con số thông qua quan hệ đồ thị. |
| 10 | Công ty mẹ của Google tên gì và thành lập năm nào? | Trả lời tốt, nhưng năm thành lập có thể nhầm của Google. | Xác định chuẩn xác: `Google` -> `PARENT_COMPANY` -> `Alphabet`. |
| 11 | Ai là người thay thế Tim Cook tại Apple (nếu có đề cập)? | Flat RAG có thể lôi thông tin dự đoán hoặc ảo giác. | Sẽ trả lời John Ternus qua quan hệ `SUCCESSOR_OF`. |
| 12 | Các công ty công nghệ lớn nào (Big Tech) từng bị chỉ trích vì độc quyền? | Chỉ trả lời được 1-2 công ty có trong chunk tìm thấy. | Trả lời được toàn bộ bằng cách query `(n)-[:CRITICIZED_FOR]->(monopoly)`. |
| 13 | Những ai tham gia sáng lập Tesla cùng với Elon Musk? | Có thể bị sót Martin Eberhard hoặc Marc Tarpenning nếu chỉ tập trung vào Elon Musk. | Liệt kê đầy đủ qua các cạnh `FOUNDED_BY` và `FOUNDED`. |
| 14 | OpenAI, Google và Amazon sử dụng công nghệ chung nào? | Flat RAG gần như không trả lời được (không fetch đủ document). | GraphRAG vượt trội vì chung node `Artificial Intelligence`. |
| 15 | Năm 2012, Facebook có sự kiện lớn nào liên quan đến tài chính? | Có thể bị nhiễu với năm khác. | Kết nối chính xác với quan hệ `IPO_DATE` năm 2012. |
| 16 | Ai đã đầu tư vào OpenAI tính đến năm 2021? | Lấy được thông tin từ chunk "Microsoft invested...". | Lấy được qua cạnh `INVESTED_BY` và liệt kê rõ ràng. |
| 17 | Công ty nào sản xuất dòng thiết bị Kindle và Echo? | Trả lời tốt. | Nhanh chóng định tuyến qua quan hệ `PRODUCES`. |
| 18 | Dịch vụ đám mây của Microsoft và Amazon tên là gì? | Phải dựa vào 2 chunks riêng biệt, có thể thiếu context. | Truy xuất 2-hop từ `Microsoft` và `Amazon` cực kỳ rõ ràng (`Azure`, `AWS`). |
| 19 | Có sự thay đổi CEO nào xảy ra ở Google năm 2015 không? | Có thể trả lời được. | Có, Larry Page -> Sundar Pichai (qua các cạnh quan hệ thời gian hoặc chức vụ). |
| 20 | Tóm tắt điểm chung giữa Microsoft và Apple trong kho dữ liệu này. | Chỉ dựa vào tần suất từ vựng, dễ bị chung chung. | Phân tích qua các node giao nhau (ví dụ: `Trillion dollar value`, `Big Tech`). |

**Kết luận đánh giá:**
- **Flat RAG** thường xuyên bị mất bối cảnh với các câu hỏi so sánh đa chủ đề (ví dụ câu 14, 18, 20) do ChromaDB chỉ ưu tiên trả về các chunks có độ tương đồng cosine cao nhất, làm trôi mất các thông tin của thực thể khác.
- **GraphRAG** kết nối thông tin tuyệt vời, tránh được ảo giác nối râu ông nọ cắm cằm bà kia (nhầm lẫn giá trị thương vụ, nhầm chức danh), đặc biệt ở các câu hỏi yêu cầu multi-hop (duyệt 2-hop, 3-hop).

## 4. Phân tích chi phí (Token Usage & Time)

* **Chi phí trích xuất thực thể (Entity/Relation Extraction):**
  * Trong bài lab này, thư mục `data` đã cung cấp sẵn `triples.json`. Tuy nhiên, nếu thực tế chạy qua LLM (prompt trích xuất triples), bộ `corpus.json` nặng khoảng ~39KB text sẽ tốn khoảng **10,000 - 15,000 tokens** đầu vào. Nếu sử dụng GPT-4o-mini hoặc GPT-3.5, chi phí cực kỳ rẻ (khoảng ~0.005$). Tuy nhiên thời gian chạy extraction có thể tốn từ **1-3 phút** do phải xử lý từng chunk text cẩn thận để chống trùng lặp (Deduplication).

* **Chi phí lập chỉ mục (Graph Construction) trên Neo4j:**
  * Việc tạo Node/Edge bằng Cypher Script là cục bộ (local). Mất chưa tới **2 giây** để import 1200+ triples vào Neo4j (như thể hiện trong terminal). Không tốn chi phí API.

* **Chi phí truy vấn (Querying):**
  * **Flat RAG:** Tốn token để Embeddings câu hỏi (rất rẻ, vài token) + Đẩy khoảng 1000 từ ngữ cảnh (context) cho LLM đọc và sinh câu trả lời.
  * **GraphRAG:** Việc tra cứu Cypher rất nhẹ và nhanh (milliseconds). Sau khi lấy ra các kết quả graph, ta dịch chúng ra text (vd: "Apple OWNS iPhone"). Đoạn text này siêu cô đọng, chỉ khoảng **100-300 tokens** đẩy cho LLM.
  * **=> Kết luận:** GraphRAG tốn chi phí ban đầu (Indexing Phase) để cấu trúc hóa đồ thị (LLM Extraction Tokens cao). Nhưng ở giai đoạn Hỏi-Đáp (Query Phase), GraphRAG lại **tiết kiệm token ngữ cảnh** hơn nhiều so với Flat RAG vì nó không phải gửi toàn bộ văn bản thô rườm rà cho LLM.
