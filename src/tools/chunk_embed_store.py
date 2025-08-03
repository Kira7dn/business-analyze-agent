"""
Quản lý embedding: load, split, embedding, lưu vào database.
Tuân thủ SOLID, dễ mở rộng/test.
"""

import time

from .utils import (
    add_documents_to_supabase,
    default_tokenizer,
    get_supabase_client,
    smart_chunk_markdown_from_text,
)


class ChunkEmbedStore:
    """
    Quản lý quy trình load, split, embedding, lưu vào Supabase.
    """

    def __init__(self, table_name: str):
        self.supabase = get_supabase_client()
        self.table_name = table_name
        self.tokenizer = default_tokenizer
        self.chunker = smart_chunk_markdown_from_text

    def process(self, file_path: str, max_tokens: int = 400) -> int:
        """
        Chunk file markdown local và lưu embedding chuẩn RAG vào Supabase.
        Args:
            file_path (str): Đường dẫn file markdown.
            max_tokens (int): Số token tối đa mỗi chunk.
        Returns:
            int: Số chunk lưu thành công.
        """
        # Đọc file một lần
        with open(file_path, "r", encoding="utf-8") as f:
            document_content = f.read()
        # Chunk nội dung từ bộ nhớ
        chunks = self.chunker(document_content, self.tokenizer, max_tokens=max_tokens)

        def extract_section_info(chunk):
            return {
                "word_count": len(chunk.split()),
                "first_50_chars": chunk[:50],
            }

        metadatas = []
        for i, chunk in enumerate(chunks):
            meta = extract_section_info(chunk)
            meta["chunk_qty"] = len(chunks)
            meta["chunk_index"] = i
            meta["filename"] = file_path
            meta["created_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            metadatas.append(meta)
        # Lưu vào Supabase
        return add_documents_to_supabase(
            self.supabase,
            contents=chunks,
            document_content=document_content,
            metadatas=metadatas,
            table_name=self.table_name,
        )
