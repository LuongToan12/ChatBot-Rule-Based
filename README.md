# Rule-Based Chatbot (John) sử dụng NLTK

Dự án này triển khai một chatbot phản hồi tự động dựa trên tập quy tắc kịch bản cấu hình trước (Rule-Based Chatbot) thông qua mô-đun `nltk.chat.util.Chat` của thư viện xử lý ngôn ngữ tự nhiên NLTK. Chatbot sử dụng Biểu thức chính quy (Regular Expressions) để nhận diện ý định và trích xuất thực thể từ câu lệnh của người dùng.

## 📋 Mục lục
1. [Tổng Quan Kiến Trúc](#1-tổng-quan-kiến-trúc)
2. [Sơ Đồ & Quy Trình Thực Hiện](#2-quy-trình-thực-hiện)
3. [Chi Tiết Cấu Hợp Phần Dữ Liệu](#3-chi-tiết-cấu-hợp-phần-dữ-liệu)
4. [Hướng Dẫn Cài Đặt & Vận Hành](#4-hướng-dẫn-cài-đặt--vận-hành)
5. [Đánh Giá Ưu & Nhược Điểm](#5-đánh-giá-ưu--nhược-điểm)

---

## 1. Tổng Quan Kiến Trúc

Hệ thống hoạt động theo cơ chế **Pattern Matching (Khớp mẫu tuần tự)**:
* **Đầu vào (Input):** Văn bản thô từ người dùng nhập qua giao diện dòng lệnh (Console).
* **Bộ máy xử lý (Core Engine):** Sử dụng `nltk.chat.util.Chat` đối chiếu chuỗi ký tự dựa trên Regex.
* **Bộ chuyển đổi đại từ (Reflections):** Ánh xạ đảo ngược ngôi xưng hô (ví dụ: chuyển "tôi" thành "bạn") để câu phản hồi tự nhiên.
* **Đầu ra (Output):** Lựa chọn ngẫu nhiên một mẫu câu trả lời phù hợp trong danh mục được định nghĩa sẵn.

---

## 2. Quy Trình Thực Hiện (Data Flow)

Quy trình xử lý một lượt hội thoại (Turn-based conversation) diễn ra theo các bước tuần tự sau:
### Chi tiết các bước thực hiện trong mã nguồn:
1. **Khởi tạo hệ thống (Initialization):** Lớp `RuleBasedChatbot` nạp hai tham số chính là `pairs` (mẫu kịch bản) và `reflections` (bộ ánh xạ ngôi).
2. **Lắng nghe luồng đầu vào (Input Loop):** Hàm `chat_with_bot()` kích hoạt vòng lặp vô hạn `while True`, liên tục bắt chuỗi ký tự người dùng nhập vào.
3. **Phân tích điều kiện dừng:** Nếu người dùng gõ cụm từ hệ thống quy định (`thoát`), chương trình sẽ thực hiện lệnh `break` thoát khỏi luồng.
4. **So khớp biểu thức chính quy (Regex Evaluation):** Trình biên dịch duyệt danh sách `pairs` theo thứ tự từ trên xuống dưới:
   * Nếu người dùng nhập: *"Tôi muốn mua điện thoại"*, hệ thống sẽ khớp với mẫu `r"tôi muốn mua (.*)"`.
   * Cụm từ nằm trong dấu ngoặc `(.*)` là thực thể biến động, được trích xuất thành giá trị cụ thể (ở đây là `"điện thoại"`).
5. **Xử lý bộ phản chiếu (Reflections Mapping):** Nếu trong cụm biến động trích xuất chứa các từ khóa như *"mình"*, *"tôi"*, hệ thống tự động đổi thành *"bạn"* để chuẩn hóa ngữ nghĩa giao tiếp của Bot.
6. **Bơm dữ liệu & Trả kết quả (Response Generation):** Tham số sau khi xử lý được chèn vào vị trí cấu hình sẵn ký hiệu `%1` (hoặc `%2`, `%3` tùy theo thứ tự nhóm ngoặc đơn trong Regex) để tạo ra câu trả lời hoàn chỉnh.
7. **Xử lý ngoại lệ (Fallback Catch-all):** Nếu chuỗi nhập vào hoàn toàn lạ lẫm và không khớp với bất kỳ quy tắc đặc định nào, dòng lệnh cuối cùng `[r"(.*)", [...]]` sẽ kích hoạt để đưa ra phản hồi yêu cầu người dùng giải thích rõ ràng hơn.

---

## 3. Chi Tiết Cấu Hợp Phần Dữ Liệu

### A. Bộ dữ liệu ánh xạ đại từ (`reflections`)
Giúp chatbot đảo ngôi ngữ pháp một cách chính xác khi hội thoại:
| Từ khóa đầu vào (User) | Phản hồi tương ứng (Bot) |
| :--- | :--- |
| `tôi` / `mình` | `bạn` |
| `bạn` | `tôi` |
| `của tôi` / `của mình` | `của bạn` |
| `của bạn` | `của tôi` |

### B. Cấu trúc tập quy tắc kịch bản (`pairs`)
Danh sách này được thiết kế theo nguyên lý **Độ ưu tiên giảm dần**:
* **Nhóm 1: Chào hỏi & định danh** (`chào|hi|hey`, `tên tôi là (.*)`) -> Đặt ở trên cùng nhằm ưu tiên phản hồi xã giao lập tức.
* **Nhóm 2: Nghiệp vụ/Tính năng** (`tôi muốn mua (.*)`, `tôi muốn xem giá (.*)`) -> Bắt các từ khóa hành động thương mại và trích xuất tên sản phẩm để tư vấn.
* **Nhóm 3: Từ khóa thoát & Hệ thống** (`bye|exit|thoát`) -> Đóng phiên làm việc.
* **Nhóm 4: Mẫu Fallback** (`(.*)`) -> Nằm ở đáy danh sách, đảm bảo chatbot luôn luôn có phản hồi trong mọi tình huống.

---

## 4. Hướng Dẫn Cài Đặt & Vận Hành

### Điều kiện kiên quyết
* Máy tính đã cài đặt **Python 3.x**
* Đã cài đặt thư viện xử lý ngôn ngữ tự nhiên **NLTK**

```bash
pip install nltk
