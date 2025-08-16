Template Ứng Dụng FastAPI, SQLAlchemy, PostgreSQLTemplate này cung cấp một cấu trúc ứng dụng mẫu tuân thủ các best practice về Rich Domain Model, Repository Pattern và Service Layer, phù hợp để xây dựng các API bền vững và dễ bảo trì với FastAPI, SQLAlchemy và PostgreSQL.Cấu trúc thư mụcyour_project_name/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── database.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── orm.py
│   ├── domain/
│   │   ├── __init__.py
│   │   └── entities.py
│   ├── repositories/
│   │   ├── __init__.py
│   │   └── crud.py
│   ├── services/
│   │   ├── __init__.py
│   │   └── business.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── pydantic.py
├── instructions.md
├── requirements.txt
└── .env.example
Nội dung các tệp tin1. your_project_name/requirements.txtĐây là danh sách các thư viện Python cần thiết cho dự án.fastapi
uvicorn[standard]
sqlalchemy
psycopg2-binary # Driver cho PostgreSQL
pydantic
python-dotenv
2. your_project_name/.env.exampleTệp này chứa ví dụ về cấu hình biến môi trường. Bạn cần sao chép nội dung này vào một tệp mới tên là .env và điền thông tin kết nối cơ sở dữ liệu PostgreSQL thực tế của bạn.DATABASE_URL="postgresql://user:password@host:5432/fastapi_db"
3. your_project_name/instructions.mdHướng dẫn chi tiết này sẽ giúp bạn hiểu cách các thành phần được tạo ra và cách sử dụng template.# Hướng Dẫn Tạo Cấu Trúc Ứng Dụng FastAPI với SQLAlchemy & PostgreSQL

Hướng dẫn này mô tả cách tạo cấu trúc ứng dụng dựa trên các định nghĩa lớp trong tệp JSON đầu vào, tuân thủ các nguyên tắc của Rich Domain Model, Repository Pattern, và Service Layer.

## Stack Công Nghệ

* **API Framework:** FastAPI
* **ORM:** SQLAlchemy
* **Cơ sở dữ liệu:** PostgreSQL (có thể dễ dàng chuyển sang các DB khác do Repository Pattern)

## Các Bước Tạo Code từ File JSON

Giả định AI nhận một file JSON định nghĩa các `classes` như `optimized_classes.json`.

### Bước 1: Cấu hình Môi trường

1.  **Tạo file `requirements.txt`**: Đặt nội dung như được cung cấp trong template. Chạy `pip install -r requirements.txt` để cài đặt các thư viện cần thiết.
2.  **Tạo file `.env`**: Sao chép nội dung từ `.env.example` và điền thông tin kết nối PostgreSQL của bạn (ví dụ: `DATABASE_URL="postgresql://myuser:mypassword@localhost:5432/mydatabase"`).

### Bước 2: Tạo SQLAlchemy ORM Models (`app/models/orm.py`)

Dựa vào các `attributes` của các `class_name` trong JSON (ví dụ: `order`, `order_processing_record`), tạo các lớp SQLAlchemy Model kế thừa từ `declarative_base()`.

* **Logic:**
    * Mỗi `class_name` trong JSON tương ứng với một bảng trong DB.
    * Các `attributes` trong JSON sẽ được ánh xạ thành `Column` trong SQLAlchemy.
    * Xác định `primary_key`, `nullable`, `unique`, `index` phù hợp. Ví dụ: `order_id` thường là `primary_key`. `datetime?` ám chỉ `nullable=True`.
    * Sử dụng kiểu dữ liệu SQLAlchemy phù hợp (e.g., `String`, `DateTime`, `Text` cho `details: dict` nếu muốn lưu dưới dạng chuỗi JSON hoặc `JSONB` cho PostgreSQL).

### Bước 3: Tạo Domain Classes (`app/domain/entities.py`)

Dựa vào các `class_name`, `attributes`, và `methods` trong JSON, tạo các lớp Domain Object. Các lớp này **chỉ chứa logic nghiệp vụ thuần túy** và không biết về DB hay `Session`.

* **Logic:**
    * Mỗi `class_name` là một lớp Domain.
    * Các `attributes` là các thuộc tính của lớp.
    * Các `methods` được định nghĩa là các phương thức của lớp, thực hiện logic nghiệp vụ bằng cách thay đổi trạng thái của chính đối tượng.

### Bước 4: Tạo Repositories (`app/repositories/crud.py`)

Dựa vào các `repository` class trong JSON (ví dụ: `order_repository`, `order_processing_repository`), tạo các lớp Repository. Mỗi Repository nhận một SQLAlchemy `Session` và chịu trách nhiệm ánh xạ giữa Domain Objects và ORM Models, thực hiện các thao tác CRUD.

* **Logic:**
    * Mỗi `repository` class là một lớp Repository.
    * Chúng có phương thức `__init__(self, session: Session)`.
    * Triển khai các phương thức được định nghĩa trong JSON (ví dụ: `get_by_status`, `get_by_id`, `save`, `fetch_all_completed`).
    * Bên trong các phương thức này, thực hiện logic chuyển đổi giữa Domain Object và ORM Model (`_to_domain`, `_to_model`) và sử dụng `self.session` để tương tác với DB.

### Bước 5: Tạo Services (`app/services/business.py`)

Dựa vào các `service` class trong JSON (ví dụ: `order_service`, `processing_metrics_service`, `reporting_service`), tạo các lớp Service. Các lớp này chứa logic phối hợp nghiệp vụ, sử dụng các Repository và các phương thức của Domain Object.

* **Logic:**
    * Mỗi `service` class là một lớp Service.
    * Chúng nhận các Repository cần thiết làm tham số khởi tạo (Dependency Injection).
    * Triển khai các phương thức được liệt kê trong JSON.
    * Bên trong các phương thức này, Service sẽ:
        1.  Sử dụng Repository để lấy Domain Object.
        2.  Gọi phương thức nghiệp vụ trên Domain Object.
        3.  Sử dụng Repository để lưu lại các thay đổi của Domain Object.
        4.  **Quan trọng:** Không gọi `commit()` hoặc `rollback()` trong Service Layer.

### Bước 6: Tạo Pydantic Schemas (`app/schemas/pydantic.py`)

Dựa vào các `attributes` của Domain Classes, tạo các Pydantic model cho request body và response body của API.

* **Logic:**
    * Mỗi Domain Class sẽ có ít nhất một Pydantic model tương ứng để trả về (response_model).
    * Các request body (ví dụ: khi tạo `Order`) cũng cần Pydantic model.
    * Sử dụng `from_attributes = True` (hoặc `orm_mode = True` cho Pydantic v1) để dễ dàng chuyển đổi từ Domain Object.

### Bước 7: Cấu hình DB và Dependency Injection (`app/database.py` và `app/config.py`)

Thiết lập kết nối PostgreSQL và dependency injection cho SQLAlchemy `Session`.

* **`app/config.py`**: Quản lý `DATABASE_URL` từ biến môi trường.
* **`app/database.py`**: Khởi tạo SQLAlchemy `engine`, `SessionLocal`, và hàm `get_db_session` cho Dependency Injection của FastAPI, cùng với `create_db_and_tables`.

### Bước 8: Tạo FastAPI App và Endpoints (`app/main.py`)

Kết nối mọi thứ lại bằng FastAPI.

* **Logic:**
    * Khởi tạo `FastAPI` app.
    * Sử dụng `get_db_session` dependency để có phiên DB cho mỗi endpoint.
    * Trong mỗi endpoint, khởi tạo các Repository và Service cần thiết.
    * Gọi các phương thức của **Service** để thực thi logic.
    * Thực hiện `db.commit()` nếu thành công hoặc `db.rollback()` nếu có lỗi.
    * Sử dụng `HTTPException` để xử lý các trường hợp lỗi (400, 404, 500).
    * Trả về Pydantic model làm response.

### Chạy ứng dụng

Sau khi tạo tất cả các tệp và thiết lập môi trường (bao gồm PostgreSQL và file `.env`), bạn có thể chạy ứng dụng bằng Uvicorn:

```bash
uvicorn app.main:app --reload
Sau đó, truy cập tài liệu API tự động tại http://127.0.0.1:8000/docs.
---

### 4. `your_project_name/app/__init__.py`

```python
# Tệp trống
5. your_project_name/app/main.pyfrom fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict

from app.database import get_db_session, create_db_and_tables
from app.repositories.crud import OrderRepository, OrderProcessingRepository
from app.services.business import OrderService, ReportingService, ProcessingMetricsService
from app.schemas.pydantic import OrderCreateRequest, OrderResponse, OrderProcessingRecordCreateRequest, OrderProcessingRecordResponse

# Khởi tạo ứng dụng FastAPI
app = FastAPI(title="Order Management API")

# Sự kiện khởi động ứng dụng: tạo các bảng DB nếu chưa tồn tại
@app.on_event("startup")
def on_startup():
    create_db_and_tables()

# Endpoint: Tạo đơn hàng mới
@app.post("/orders/", response_model=OrderResponse, status_code=status.HTTP_201_CREATED)
def create_order_endpoint(request: OrderCreateRequest, db: Session = Depends(get_db_session)):
    try:
        # Khởi tạo Repository và Service
        order_repo = OrderRepository(db)
        order_service = OrderService(order_repo)

        # Gọi phương thức tạo đơn hàng từ Service
        new_order = order_service.create_new_order(request.order_id, request.details)

        # Commit giao dịch và làm mới đối tượng để đảm bảo dữ liệu mới nhất từ DB
        db.commit()
        db.refresh(new_order)
        return new_order
    except Exception as e:
        # Rollback giao dịch nếu có lỗi
        db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create order: {e}")

# Endpoint: Xác nhận đơn hàng
@app.put("/orders/{order_id}/confirm", response_model=OrderResponse)
def confirm_order_endpoint(order_id: str, db: Session = Depends(get_db_session)):
    try:
        order_repo = OrderRepository(db)
        order_service = OrderService(order_repo)

        # Gọi phương thức xác nhận đơn hàng từ Service
        confirmed_order = order_service.confirm_order(order_id)

        if not confirmed_order:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Order not found")

        db.commit()
        db.refresh(confirmed_order)
        return confirmed_order
    except ValueError as ve: # Xử lý lỗi nghiệp vụ từ domain model (ví dụ: không thể xác nhận trạng thái hiện tại)
        db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to confirm order: {e}")

# Endpoint: Đóng gói đơn hàng
@app.put("/orders/{order_id}/package", response_model=OrderResponse)
def package_order_endpoint(order_id: str, db: Session = Depends(get_db_session)):
    try:
        order_repo = OrderRepository(db)
        order_service = OrderService(order_repo)

        # Gọi phương thức đóng gói đơn hàng từ Service
        packaged_order = order_service.package_order(order_id)

        if not packaged_order:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Order not found")

        db.commit()
        db.refresh(packaged_order)
        return packaged_order
    except ValueError as ve:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to package order: {e}")

# Endpoint: Lấy thông tin đơn hàng theo ID
@app.get("/orders/{order_id}", response_model=OrderResponse)
def get_order_by_id_endpoint(order_id: str, db: Session = Depends(get_db_session)):
    order_repo = OrderRepository(db)
    order = order_repo.get_by_id(order_id)
    if not order:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Order not found")
    return order

# Endpoint: Lấy danh sách đơn hàng theo trạng thái
@app.get("/orders/", response_model=List[OrderResponse])
def get_orders_by_status_endpoint(status_filter: str, page: int = 1, per_page: int = 10, db: Session = Depends(get_db_session)):
    order_repo = OrderRepository(db)
    orders = order_repo.get_by_status(status_filter, page, per_page)
    return orders

# Endpoint: Tạo bản ghi xử lý đơn hàng
@app.post("/order-processing-records/", response_model=OrderProcessingRecordResponse, status_code=status.HTTP_201_CREATED)
def create_order_processing_record_endpoint(request: OrderProcessingRecordCreateRequest, db: Session = Depends(get_db_session)):
    try:
        processing_repo = OrderProcessingRepository(db)
        # Tạo đối tượng Domain Record
        new_record = OrderProcessingRecord(
            order_id=request.order_id,
            employee_id=request.employee_id,
            start_timestamp=request.start_timestamp,
            end_timestamp=request.end_timestamp
        )
        processing_repo.add(new_record)
        db.commit()
        db.refresh(new_record) # Refresh để lấy ID nếu DB tự tạo (ví dụ: autoincrement)
        return new_record
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create processing record: {e}")

# Endpoint: Lấy báo cáo thời gian xử lý của nhân viên
@app.get("/reports/employee-processing-time", response_model=Dict[str, float])
def get_employee_processing_report_endpoint(db: Session = Depends(get_db_session)):
    try:
        processing_repo = OrderProcessingRepository(db)
        metrics_service = ProcessingMetricsService()
        reporting_service = ReportingService(processing_repo, metrics_service)

        report = reporting_service.generate_employee_time_report()
        return report
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to generate report: {e}")
6. your_project_name/app/config.pyfrom dotenv import load_dotenv
import os

load_dotenv() # Tải các biến môi trường từ tệp .env

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/fastapi_db")
7. your_project_name/app/database.pyfrom sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.models.orm import Base # Import Base từ ORM models
from app.config import DATABASE_URL # Import URL từ cấu hình

# Tạo SQLAlchemy Engine để kết nối tới DB
engine = create_engine(DATABASE_URL)

# Tạo SessionLocal class (mỗi instance là một "Unit of Work")
# autocommit=False: không tự động commit sau mỗi thao tác
# autoflush=False: không tự động flush (đồng bộ hóa) các thay đổi vào DB trước khi query
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency để cung cấp Session cho mỗi yêu cầu FastAPI
# Đảm bảo session được đóng sau khi yêu cầu được xử lý
def get_db_session():
    db = SessionLocal()
    try:
        yield db # Cung cấp session cho endpoint
    finally:
        db.close() # Đóng session khi hoàn thành hoặc có lỗi

# Hàm tiện ích để tạo tất cả các bảng trong DB dựa trên các ORM models đã định nghĩa
def create_db_and_tables():
    Base.metadata.create_all(engine)
8. your_project_name/app/models/__init__.py# Tệp trống
9. your_project_name/app/models/orm.pyfrom sqlalchemy import Column, String, DateTime, Text, BigInteger
from sqlalchemy.ext.declarative import declarative_base

# declarative_base() là cơ sở để định nghĩa các lớp ORM
Base = declarative_base()

# Lớp OrderModel ánh xạ tới bảng 'orders'
class OrderModel(Base):
    __tablename__ = 'orders'
    order_id = Column(String, primary_key=True, unique=True, index=True)
    status = Column(String, default="New", nullable=False)
    details = Column(Text, nullable=True) # Lưu trữ JSON dict dưới dạng chuỗi Text (hoặc JSONB cho PostgreSQL)
    confirmed_timestamp = Column(DateTime, nullable=True)
    packaged_timestamp = Column(DateTime, nullable=True)

# Lớp OrderProcessingRecordModel ánh xạ tới bảng 'order_processing_records'
class OrderProcessingRecordModel(Base):
    __tablename__ = 'order_processing_records'
    id = Column(BigInteger, primary_key=True, autoincrement=True) # ID tự tăng
    order_id = Column(String, nullable=False, index=True)
    employee_id = Column(String, nullable=False, index=True)
    start_timestamp = Column(DateTime, nullable=False)
    end_timestamp = Column(DateTime, nullable=False)
10. your_project_name/app/domain/__init__.py# Tệp trống
11. your_project_name/app/domain/entities.pyfrom datetime import datetime
from typing import Optional, Dict

# Lớp Domain Order
class Order:
    def __init__(self, order_id: str, status: str, details: Dict,
                 confirmed_timestamp: Optional[datetime] = None,
                 packaged_timestamp: Optional[datetime] = None):
        self.order_id = order_id
        self.status = status
        self.details = details
        self.confirmed_timestamp = confirmed_timestamp
        self.packaged_timestamp = packaged_timestamp

    # Phương thức nghiệp vụ: xác nhận đơn hàng
    def confirm(self):
        if self.status == "New":
            self.status = "Confirmed"
            self.confirmed_timestamp = datetime.now()
        else:
            raise ValueError(f"Order {self.order_id} cannot be confirmed from status '{self.status}'.")

    # Phương thức nghiệp vụ: đóng gói đơn hàng
    def package(self):
        if self.status == "Confirmed":
            self.status = "Packaged"
            self.packaged_timestamp = datetime.now()
        else:
            raise ValueError(f"Order {self.order_id} cannot be packaged from status '{self.status}'.")

    # Phương thức tiện ích để chuyển đổi sang dict (hữu ích cho việc trả về API nếu không dùng Pydantic Config)
    def to_dict(self):
        return {
            "order_id": self.order_id,
            "status": self.status,
            "details": self.details,
            "confirmed_timestamp": self.confirmed_timestamp.isoformat() if self.confirmed_timestamp else None,
            "packaged_timestamp": self.packaged_timestamp.isoformat() if self.packaged_timestamp else None
        }

# Lớp Domain OrderProcessingRecord
class OrderProcessingRecord:
    def __init__(self, order_id: str, employee_id: str,
                 start_timestamp: datetime, end_timestamp: datetime, record_id: Optional[int] = None):
        self.id = record_id
        self.order_id = order_id
        self.employee_id = employee_id
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp

    # Phương thức nghiệp vụ: tính thời lượng xử lý
    def duration(self) -> float:
        return (self.end_timestamp - self.start_timestamp).total_seconds()
12. your_project_name/app/repositories/__init__.py# Tệp trống
13. your_project_name/app/repositories/crud.pyfrom sqlalchemy.orm import Session
from typing import List, Optional, Dict
import json # Để xử lý details là dict

from app.models.orm import OrderModel, OrderProcessingRecordModel
from app.domain.entities import Order, OrderProcessingRecord

# Repository cho Order
class OrderRepository:
    def __init__(self, session: Session):
        self.session = session

    # Phương thức chuyển đổi từ ORM Model sang Domain Object
    def _to_domain(self, model: OrderModel) -> Order:
        return Order(
            order_id=model.order_id,
            status=model.status,
            details=json.loads(model.details) if model.details else {}, # Giải mã JSON string
            confirmed_timestamp=model.confirmed_timestamp,
            packaged_timestamp=model.packaged_timestamp
        )

    # Phương thức chuyển đổi từ Domain Object sang ORM Model (cho việc lưu trữ/cập nhật)
    def _to_model(self, domain: Order) -> OrderModel:
        model = self.session.query(OrderModel).filter_by(order_id=domain.order_id).first()
        if not model: # Nếu chưa tồn tại, tạo mới
            model = OrderModel(order_id=domain.order_id)

        # Cập nhật các thuộc tính của ORM Model từ Domain Object
        model.status = domain.status
        model.details = json.dumps(domain.details) # Mã hóa dict thành JSON string
        model.confirmed_timestamp = domain.confirmed_timestamp
        model.packaged_timestamp = domain.packaged_timestamp
        return model

    # Lấy danh sách đơn hàng theo trạng thái
    def get_by_status(self, status: str, page: int = 1, per_page: int = 10) -> List[Order]:
        offset = (page - 1) * per_page
        models = self.session.query(OrderModel).filter_by(status=status).offset(offset).limit(per_page).all()
        return [self._to_domain(model) for model in models]

    # Lấy đơn hàng theo ID
    def get_by_id(self, order_id: str) -> Optional[Order]:
        model = self.session.query(OrderModel).filter_by(order_id=order_id).first()
        return self._to_domain(model) if model else None

    # Thêm đơn hàng mới
    def add(self, order: Order):
        order_model = self._to_model(order)
        self.session.add(order_model) # Thêm vào session để theo dõi

    # Cập nhật đơn hàng hiện có
    def update(self, order: Order):
        order_model = self._to_model(order)
        self.session.add(order_model) # session.add() sẽ tự động merge/cập nhật nếu đối tượng đã tồn tại

    # Xóa đơn hàng
    def delete(self, order_id: str):
        order_model = self.session.query(OrderModel).filter_by(order_id=order_id).first()
        if order_model:
            self.session.delete(order_model)

# Repository cho OrderProcessingRecord
class OrderProcessingRepository:
    def __init__(self, session: Session):
        self.session = session

    # Phương thức chuyển đổi từ ORM Model sang Domain Object
    def _to_domain(self, model: OrderProcessingRecordModel) -> OrderProcessingRecord:
        return OrderProcessingRecord(
            record_id=model.id,
            order_id=model.order_id,
            employee_id=model.employee_id,
            start_timestamp=model.start_timestamp,
            end_timestamp=model.end_timestamp
        )

    # Phương thức chuyển đổi từ Domain Object sang ORM Model
    def _to_model(self, domain: OrderProcessingRecord) -> OrderProcessingRecordModel:
        # Nếu domain.id đã được thiết lập (cho bản ghi hiện có), gán nó
        # Nếu là bản ghi mới, DB sẽ tự tạo ID
        model = OrderProcessingRecordModel(
            order_id=domain.order_id,
            employee_id=domain.employee_id,
            start_timestamp=domain.start_timestamp,
            end_timestamp=domain.end_timestamp
        )
        if domain.id is not None:
            model.id = domain.id
        return model

    # Lấy tất cả các bản ghi đã hoàn thành
    def fetch_all_completed(self) -> List[OrderProcessingRecord]:
        models = self.session.query(OrderProcessingRecordModel).all()
        return [self._to_domain(model) for model in models]

    # Thêm bản ghi xử lý mới
    def add(self, record: OrderProcessingRecord):
        record_model = self._to_model(record)
        self.session.add(record_model)
14. your_project_name/app/services/__init__.py# Tệp trống
15. your_project_name/app/services/business.pyfrom typing import List, Dict, Optional
from app.domain.entities import Order, OrderProcessingRecord
from app.repositories.crud import OrderRepository, OrderProcessingRepository # Import Repository
from datetime import datetime

# Service cho Order
class OrderService:
    def __init__(self, order_repo: OrderRepository):
        self.order_repo = order_repo

    def create_new_order(self, order_id: str, details: Dict) -> Order:
        order = Order(order_id=order_id, status="New", details=details)
        self.order_repo.add(order) # Yêu cầu Repository thêm vào session
        return order

    def fetch_new_orders(self, page: int = 1, per_page: int = 10) -> List[Order]:
        return self.order_repo.get_by_status("New", page, per_page)

    def confirm_order(self, order_id: str) -> Optional[Order]:
        order = self.order_repo.get_by_id(order_id) # Lấy Domain Object
        if order:
            order.confirm() # Gọi logic nghiệp vụ trên Domain Object
            self.order_repo.update(order) # Yêu cầu Repository cập nhật
        return order

    def fetch_confirmed_orders(self, page: int = 1, per_page: int = 10) -> List[Order]:
        return self.order_repo.get_by_status("Confirmed", page, per_page)

    def package_order(self, order_id: str) -> Optional[Order]:
        order = self.order_repo.get_by_id(order_id)
        if order:
            order.package() # Gọi logic nghiệp vụ trên Domain Object
            self.order_repo.update(order) # Yêu cầu Repository cập nhật
        return order

# Service cho các chỉ số xử lý
class ProcessingMetricsService:
    def average_time_by_employee(self, records: List[OrderProcessingRecord]) -> Dict[str, float]:
        employee_durations: Dict[str, List[float]] = {}
        for record in records:
            employee_durations.setdefault(record.employee_id, []).append(record.duration())

        avg_times = {
            emp_id: sum(durations) / len(durations)
            for emp_id, durations in employee_durations.items()
        }
        return avg_times

# Service cho báo cáo
class ReportingService:
    def __init__(self, processing_repo: OrderProcessingRepository, metrics_service: ProcessingMetricsService):
        self.processing_repo = processing_repo
        self.metrics_service = metrics_service

    def generate_employee_time_report(self) -> Dict[str, float]:
        records = self.processing_repo.fetch_all_completed() # Lấy dữ liệu thô từ Repository
        report = self.metrics_service.average_time_by_employee(records) # Xử lý bằng Metrics Service
        return report
16. your_project_name/app/schemas/__init__.py# Tệp trống
17. your_project_name/app/schemas/pydantic.pyfrom pydantic import BaseModel, Field
from typing import Optional, Dict
from datetime import datetime

# Schema cho request tạo đơn hàng
class OrderCreateRequest(BaseModel):
    order_id: str
    details: Dict = Field(default_factory=dict) # Mặc định là dict rỗng nếu không cung cấp

# Schema cho response của Order
class OrderResponse(BaseModel):
    order_id: str
    status: str
    details: Dict
    confirmed_timestamp: Optional[datetime] = None
    packaged_timestamp: Optional[datetime] = None

    class Config:
        from_attributes = True # Cho phép Pydantic tạo instance từ thuộc tính của các đối tượng khác (ví dụ: Domain Objects)

# Schema cho request tạo bản ghi xử lý đơn hàng
class OrderProcessingRecordCreateRequest(BaseModel):
    order_id: str
    employee_id: str
    start_timestamp: datetime
    end_timestamp: datetime

# Schema cho response của OrderProcessingRecord
class OrderProcessingRecordResponse(BaseModel):
    id: Optional[int] # ID có thể được sinh tự động bởi DB, nên là Optional khi tạo mới
    order_id: str
    employee_id: str
    start_timestamp: datetime
    end_timestamp: datetime

    class Config:
        from_attributes = True
