# Product Requirements Document (PRD)

## 1. Project Overview
Xây dựng web app quản lý đơn hàng giúp nhân viên kho xử lý ít nhất 1.000 đơn/ngày với thời gian phản hồi trung bình ≤ 150 ms.

## 2. Target Users
- Nhân viên kho
- Quản lý kho (20–30 người dùng nội bộ)

## 3. Core Features
- Dashboard tổng quan đơn hàng (lọc theo trạng thái, ngày tạo)
- Xác nhận đơn hàng
- Đóng gói đơn hàng
- Thông báo email/SMS khi đơn thay đổi trạng thái
- Quan sát và cập nhật tình trạng đơn hàng/kho hàng

## 4. Non-functional Requirements
- Thời gian phản hồi API ≤ 150 ms
- Khả năng chịu tải 500 requests/giây
- Tuân thủ bảo mật OAuth2
- Hệ thống nhỏ gọn, chạy trên cloud server với RAM 1GB

## 5. Success Metrics (KPIs)
- Giảm thời gian xử lý đơn từ 5 phút xuống còn ≤ 2 phút
- Duy trì uptime ≥ 99,9% hàng tháng

## 6. Feasibility / Challenges
- Tối ưu hiệu năng và dung lượng phù hợp tài nguyên hạn chế
- Chỉ được phép chạy trên cloud server với RAM 1GB

## 7. Risks / Dependencies
- Có thể có rủi ro khi danh sách hạng mục kho hàng chưa đầy đủ
- Có rủi ro khi danh sách item trong kho không được cập nhật chính xác

## 8. Feature Suggestions by User Role

### Nhân viên kho (Warehouse Staff)
- Dashboard tổng quan đơn hàng (must-have)
- Xác nhận đơn hàng (must-have)
- Đóng gói đơn hàng (must-have)
- Thông báo trạng thái đơn hàng (must-have)

### Quản lý kho (Warehouse Manager)
- Theo dõi thời gian xử lý trung bình của từng nhân viên (nice-to-have)
- Báo cáo tổng quan về kho hàng (nice-to-have)
- Quản lý tài khoản người dùng (nice-to-have)
