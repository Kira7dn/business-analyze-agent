"""
memory_creator.py - Công cụ tạo lệnh cho Cascade AI Assistant trong Windsurf
"""

import sys
import io
import json
from typing import Dict, Any, List, Optional


def create_command(tool_name: str, params: Dict[str, Any]) -> str:
    """
    Tạo lệnh cho AI Assistant dưới dạng chuỗi JSON trong thẻ XML

    Args:
        tool_name (str): Tên công cụ (sẽ trở thành tên thẻ XML)
        params (Dict[str, Any]): Các tham số của lệnh dưới dạng dictionary

    Returns:
        str: Chuỗi lệnh đã được format
    """
    # Chuyển đổi thành JSON và thêm vào thẻ XML
    json_str = json.dumps(params, ensure_ascii=False, indent=2)
    command = f"""<{tool_name}>
{json_str}
</{tool_name}>"""

    return command


def create_memory(title: str, content: str, tags: Optional[List[str]] = None) -> str:
    """
    Tạo lệnh create_memory

    Args:
        title (str): Tiêu đề của memory
        content (str): Nội dung chi tiết
        tags (List[str], optional): Danh sách các tags. Mặc định là None.

    Returns:
        str: Chuỗi lệnh đã được format
    """
    if tags is None:
        tags = []

    memory_data = {
        "Title": title,
        "Content": content,
        "Tags": tags,
        "CorpusNames": ["c:/Workspace/business-analyze-agent"],
        "Action": "create",
        "UserTriggered": True,
    }

    return create_command("create_memory", memory_data)


def create_demand(commands: List[Dict[str, Any]]) -> None:
    """
    Tạo và in ra các lệnh theo định dạng AI CODING - DEMAND

    Args:
        commands (List[Dict[str, Any]]): Danh sách các lệnh cần thực thi
            Mỗi lệnh là một dictionary với các key:
            - 'tool': Tên công cụ (bắt buộc)
            - 'params': Tham số cho công cụ (bắt buộc)
            - 'description': Mô tả ngắn (tùy chọn)
    """

    # Thiết lập stdout để hỗ trợ UTF-8
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("=" * 50)
    print("AI CODING - DEMAND:")
    print("=" * 50 + "\n")

    for i, cmd in enumerate(commands, 1):
        tool_name = cmd.get("tool")
        params = cmd.get("params", {})
        description = cmd.get("description", f"Command {i}")

        if not tool_name or not isinstance(params, dict):
            print(f"Invalid command format at index {i}")
            continue

        command = create_command(tool_name, params)
        print(f"{description}:")
        print(command)
        print("\n" + "-" * 50 + "\n")


# Example usage
if __name__ == "__main__":
    create_demand(
        [
            {
                "tool": "create_memory",
                "description": "Create system login information",
                "params": {
                    "Title": "Thông tin đăng nhập hệ thống",
                    "Content": "Thông tin truy cập hệ thống quản trị:\n- URL: https://admin.example.com\n- Username: admin\n- Password: ********",
                    "Tags": ["credentials", "admin"],
                    "CorpusNames": ["c:/Workspace/business-analyze-agent"],
                    "Action": "create",
                    "UserTriggered": True,
                },
            }
        ]
    )
