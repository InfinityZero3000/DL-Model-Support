I. Dataset 
1. Dữ liệu Fluency Scoring (Đánh giá độ trôi chảy)
Nguồn dữ liệu gốc:
•	CoLA (Corpus of Linguistic Acceptability): Đánh giá tính chuẩn xác của ngôn ngữ 
•	GLUE SST-2: Dữ liệu phân loại Sentiment, được dùng như prompt proxy cho fluency 
•	Dữ liệu Local: Tập wi_locness (văn bản của người học tiếng Anh thực tế).
2. Dữ liệu Grammar Correction (Sửa lỗi ngữ pháp)
Nguồn dữ liệu gốc:
•	Nguồn chính: Tập dữ liệu wi_locness (kết hợp BEA-2019 Shared Task) có sẵn ở local.
•	Nguồn bổ sung (được đề cập định hướng): CoNLL-2014, FCE Corpus, và các lỗi được tạo ra bằng phương pháp Synthetic errors.
3. Dữ liệu Vocabulary Classification (Phân loại từ vựng)
Nguồn dữ liệu gốc:
•	Simple Wikipedia: Bách khoa toàn thư phiên bản tiếng Anh đơn giản (tương đương A2-B1)
•	AG News: Các bài báo tin tức (tương đương độ khó B2)
•	SNLI Dataset: Tập dữ liệu suy luận ngôn ngữ đa dạng độ khó
4. Dữ liệu Dialogue Generation (Sinh hội thoại)
Nguồn dữ liệu gốc:
•	OpenOrca: Dataset tập trung vào khả năng làm theo chỉ dẫn (Instruction Following)
•	DialogSum: Tập dữ liệu tóm tắt và xử lý hội thoại
•	Anthropic HH-RLHF: Tập dữ liệu rèn luyện tính hữu ích và an toàn (Helpful & Harmless)


nyu-mll/glue-cola	https://huggingface.co/datasets/nyu-mll/glue/viewer/cola
Dùng load_dataset("nyu-mll/glue", "cola"); split chính: train, validation, test. HF ghi CoLA là subset của nyu-mll/glue, có khoảng 10.7k rows. (Hugging Face)

nyu-mll/glue-sst2	https://huggingface.co/datasets/nyu-mll/glue/viewer/sst2
Dùng load_dataset("nyu-mll/glue", "sst2"); split: train, validation, test. Đây là bản SST-2 trong GLUE, nhãn 0 = negative, 1 = positive. (Hugging Face)

wikipedia - 20220301.simple	Bản hiện tại chuẩn trên HF: https://huggingface.co/datasets/wikimedia/wikipedia
Nếu không bắt buộc đúng snapshot cũ, dùng load_dataset("wikimedia/wikipedia", "20231101.simple"). HF hiện liệt kê 20231101.simple; còn 20220301.simple là snapshot cũ, nên nếu cần đúng phiên bản này thì tải từ Wikimedia dump thay vì mirror lạ. (Hugging Face)

ag_news	https://huggingface.co/datasets/fancyzhx/ag_news
Dùng load_dataset("fancyzhx/ag_news"); split: train 120k rows, test 7.6k rows. (Hugging Face)

sst	https://huggingface.co/datasets/stanfordnlp/sst
Dùng load_dataset("stanfordnlp/sst") nếu cần Stanford Sentiment Treebank bản đầy đủ. Nếu chỉ cần binary sentiment classification thì dùng nyu-mll/glue, config sst2. (Hugging Face)

Open-Orca/OpenOrca	https://huggingface.co/datasets/Open-Orca/OpenOrca
Dùng load_dataset("Open-Orca/OpenOrca"); dữ liệu instruction-tuning, có các cột như system_prompt, question, response. (Hugging Face)

knkarthick/dialogsum	HF: https://huggingface.co/datasets/knkarthick/dialogsum ; bản gốc: https://github.com/cylnlp/dialogsum
Dùng load_dataset("knkarthick/dialogsum"); split: train, validation, test; cột chính: dialogue, summary, topic. (Hugging Face)

Anthropic/hh-rlhf	HF: https://huggingface.co/datasets/Anthropic/hh-rlhf ; bản gốc: https://github.com/anthropics/hh-rlhf
Dùng load_dataset("Anthropic/hh-rlhf"); phù hợp cho RLHF/DPO/reward model, mỗi dòng thường có cặp chosen và rejected. (Hugging Face)



II. Data Processing
Dữ liệu thô từ các nguồn trên không được đưa ngay vào để training, mà đi qua một quy trình xử lý, làm sạch và định dạng chặt chẽ bằng cách sử dụng script (download_and_inspect_datasets.py):
1.	Làm sạch văn bản (Text Normalization):
o	Loại bỏ các khoảng trắng thừa, căn chỉnh chuẩn in thường (lowercasing).
o	Sử dụng hàm looks_english() để quét và loại trừ những văn bản "rác": nếu một sample có tỷ lệ ký tự non-ASCII trên 20%, nó sẽ bị loại bỏ.
2.	Cắt đoạn và chọn ngữ cảnh (Context Slicing):
o	Với các đoạn text quá dài (như Wikipedia), hàm split_sentences() và pick_context() được tự động sử dụng để trích xuất ngẫu nhiên số lượng câu phù hợp với min/max context (nhằm mô phỏng kích thước của câu hội thoại ngắn).
3.	Sửa lỗi ngữ pháp tự động (Grammar M2 Edits):
o	Với tập Grammar (như wi_locness), dữ liệu lưu lỗi ở định dạng chuẩn M2 Spans. Dữ liệu được đưa qua hàm apply_m2_edits() để duyệt thay thế đoạn text lỗi từ phải sang trái (right-to-left) thành văn bản đúng ngữ pháp hoàn chỉnh, chuẩn bị dữ liệu input (lỗi) và output (câu đúng).
4.	Ước lượng độ khó từ vựng (Vocabulary CEFR Estimation):
o	Hệ thống dùng một thuật toán heuristic (estimate_vocab_level) dựa trên độ dài trung bình của từ và tỷ lệ từ vựng dài (hơn 9 ký tự). Nếu tỷ lệ này cao, dữ liệu được gán nhãn CEFR B2 hoặc B1. Nếu từ vựng ngắn, nó được đánh giá là A2.
5.	Loại bỏ trùng lặp (Deduplication):
o	Sử dụng mã băm MD5 (stable_hash) trên các text đầu vào để đảm bảo không có bất kỳ câu nào bị lặp lại trong cùng một task hoặc lặp lại trên toàn cục (global deduplication).
6.	Kiểm duyệt cuối cùng (Quality Inspection):
o	Các sample quá ngắn (dưới 5 ký tự) bị loại bỏ.
o	Dữ liệu được lưu trữ tập trung vào các file .jsonl trong thư mục training_data_clean cùng với file clean_report.json báo cáo phân bổ task.



