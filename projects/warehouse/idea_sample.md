# Order Management Web App Requirements

## Project Goal
Xây dựng web app quản lý đơn hàng giúp nhân viên kho xử lý ít nhất 1.000 đơn/ngày với thời gian phản hồi trung bình ≤ 150 ms.

## Target Users
- Nhân viên kho
- Quản lý kho (khoảng 20–30 người dùng nội bộ)

## Core Features
- Dashboard tổng quan đơn hàng (lọc theo trạng thái, ngày tạo)
- Chức năng “xác nhận đơn” & “đóng gói”
- Thông báo email/SMS khi đơn thay đổi trạng thái
- Nhân viên có thể quan sát tình trạng đơn hàng và nhanh chóng cập nhật kho hàng

## Non-functional Constraints
- Thời gian phản hồi API ≤ 150 ms
- Khả năng chịu tải 500 requests/giây
- Tuân thủ bảo mật OAuth2
- Hệ thống phải đơn giản, nhỏ gọn, có thể chạy trên server cloud với tài nguyên chỉ 1 GB

## Success Metrics (KPIs)
- Giảm thời gian xử lý đơn từ 5 phút xuống còn ≤ 2 phút
- Duy trì uptime ≥ 99,9% hàng tháng

## Feasibility / Challenges
- Hệ thống cần tối ưu hiệu năng và dung lượng để phù hợp với tài nguyên hạn chế
- Hệ thống chỉ được phép chạy trên cloud server với RAM 1GB

## Risks / Dependencies
- Có thể có rủi ro khi danh sách hạng mục kho hàng chưa đầy đủ
- Có rủi ro trong khi danh sách item trong kho không được cập nhật chính xác

---

# Feature Suggestions by User Role

## Nhân viên kho
- **Dashboard tổng quan đơn hàng** (must-have): Cho phép lọc đơn hàng theo trạng thái và ngày tạo.
- **Xác nhận đơn hàng** (must-have): Chức năng để xác nhận các đơn hàng mới.
- **Đóng gói đơn hàng** (must-have): Chức năng để cập nhật trạng thái đơn hàng đã được đóng gói.
- **Thông báo trạng thái đơn hàng** (must-have): Gửi email/SMS thông báo khi đơn hàng thay đổi trạng thái.

## Quản lý kho
- **Theo dõi thời gian xử lý trung bình của từng nhân viên** (nice-to-have): Giúp giám sát hiệu suất làm việc của nhân viên.
- **Báo cáo tổng quan về kho hàng** (nice-to-have): Hiển thị tổng quan về số lượng và tình trạng đơn hàng.
- **Quản lý tài khoản người dùng** (nice-to-have): Chức năng để quản lý thông tin và quyền truy cập của nhân viên.