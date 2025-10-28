# Công Cụ Phân Tích Cảm Xúc Đa Khía Cạnh Khách Sạn

Repo này chứa mã nguồn cho một ứng dụng web hỗ trợ phân tích cảm xúc đa khía cạnh (Aspect-Based Sentiment Analysis - ABSA) cho các đánh giá khách sạn. Ứng dụng bao gồm các công cụ chính: Tiền xử lý dữ liệu, Gán nhãn dữ liệu, và Phân loại câu đơn. 

Ngoài pre-processing tool, thì annotation tool có 2 loại. 
*  Manually Annotation Tool: gán nhãn thủ công được viết bằng html , js, css.
*  Semi-Annotation Tool: gán nhãn bán tự động sử dụng mô hình học máy

## Yêu Cầu

*   Python 3.8+
*   Các thư viện được liệt kê trong `requirements.txt`.

## Cài Đặt

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/tranminhvu945/DS102.git
    cd ABSA-Hotel-Tool
    ```

2.  **Cài đặt thư viện:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Mô Hình:**
    *   Phải đảm bảo bạn có file model `pipe.joblib` đã được huấn luyện trong root directory
    *   Đặt tệp này vào thư mục `./model/` (tạo thư mục `model` nếu chưa có).

## Chạy Ứng Dụng

Sử dụng Streamlit để chạy ứng dụng:

```bash
streamlit run main.py
```
