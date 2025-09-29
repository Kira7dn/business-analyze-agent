"""
Quản lý embedding: load, split, embedding, lưu vào database.
Tuân thủ SOLID, dễ mở rộng/test.
"""

import re
import os
import shutil
import time
import logging
import concurrent.futures
from typing import Any, Dict, List, Tuple, Callable

import openai
from tiktoken import get_encoding
from supabase import create_client, Client


def html_to_markdown(
    input_path: str,
    output_path: str = None,
    images_folder_rename: str = None,
) -> None:
    """
    Convert an HTML file to Markdown, optionally rewrite image folder names/paths, and save as UTF-8.
    If output_path is None, save to the same folder as input_path, replacing .html with .md.
    Args:
        input_path (str): Path to the HTML file.
        output_path (str, optional): Path to save the Markdown file. If None, uses input_path with .md extension.
        images_folder_rename (str, optional): If provided, rewrite image folder names/paths and rename folder on disk.
    Returns:
        None
    Raises:
        IOError: If file operations fail.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        html = f.read()
    try:
        from markdownify import markdownify as md
    except ImportError as exc:
        raise ImportError(
            "markdownify package is required. Install with: pip install markdownify"
        ) from exc
    markdown = md(html)
    # Auto-generate folder name if not provided
    if images_folder_rename is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        safe_base = re.sub(r"[^A-Za-z0-9]+", "_", base)
        images_folder_rename = f"{safe_base}_files"
    if images_folder_rename:
        # Replace folder names in image paths
        markdown = re.sub(
            r"!\[(.*?)\]\(([^)]+?)\)",
            lambda m: f"![{m.group(1)}]({images_folder_rename}/"
            + os.path.basename(m.group(2))
            + ")",
            markdown,
        )
        # Physically rename the folder if it exists
        input_dir = os.path.dirname(input_path)
        orig_folder = os.path.splitext(os.path.basename(input_path))[0] + "_files"
        orig_folder_path = os.path.join(input_dir, orig_folder)
        new_folder_path = os.path.join(input_dir, images_folder_rename)
        if os.path.exists(orig_folder_path) and orig_folder != images_folder_rename:
            shutil.move(orig_folder_path, new_folder_path)
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + ".md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    return output_path


def get_supabase_client() -> Client:
    """
    Get a Supabase client with the URL and key from environment variables.

    Returns:
        Supabase client instance
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_KEY must be set in environment variables"
        )

    return create_client(url, key)


def default_tokenizer(text: str) -> List[int]:
    """
    Tokenize text using tiktoken cl100k_base.
    Args:
        text (str): Input text.
    Returns:
        List[int]: Token ids.
    """
    return get_encoding("cl100k_base").encode(text)


def add_documents_to_supabase(
    client: Client,
    contents: List[str],
    metadatas: List[Dict[str, Any]],
    document_content: str,
    table_name: str,
    batch_size: int = 20,
) -> int:
    """
    Add documents/chunks to Supabase in batches. Returns the number of successfully inserted records.
    Optimized for clarity and efficiency.
    """

    logger = logging.getLogger("vectorDB.utils")
    total_inserted = 0
    n = len(contents)
    for i in range(0, n, batch_size):
        batch_contents = contents[i : i + batch_size]
        batch_metadatas = metadatas[i : i + batch_size]

        # Prepare contextual content in parallel
        process_args = [(document_content, content) for content in batch_contents]
        contextual_contents = [None] * len(batch_contents)
        contextual_flags = [False] * len(batch_contents)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [
                    executor.submit(generate_contextual_embedding, *arg)
                    for arg in process_args
                ]
                for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                    result, success = future.result()
                    contextual_contents[idx] = result
                    contextual_flags[idx] = success
        except Exception as e:
            logger.warning(
                f"Error in contextual embedding batch: {e}, using original chunks."
            )
            contextual_contents = batch_contents[:]
            contextual_flags = [False] * len(batch_contents)

        # Fallback if any None
        for idx, val in enumerate(contextual_contents):
            if val is None:
                contextual_contents[idx] = batch_contents[idx]

        # Update metadata with contextual embedding flag
        for idx, flag in enumerate(contextual_flags):
            if flag:
                batch_metadatas[idx]["contextual_embedding"] = True

        # Batch embedding
        batch_embeddings = create_embeddings_batch(contextual_contents)
        # Prepare data for Supabase
        batch_data = [
            {
                "content": contextual_contents[j],
                "metadata": {
                    "chunk_size": len(contextual_contents[j]),
                    **batch_metadatas[j],
                },
                "embedding": batch_embeddings[j],
            }
            for j in range(len(contextual_contents))
        ]

        # Insert batch, with retries
        max_retries = 3
        for retry in range(max_retries):
            try:
                client.table(table_name).insert(batch_data).execute()
                total_inserted += len(batch_data)
                break
            except Exception as e:
                logger.warning(
                    f"Error inserting batch (attempt {retry+1}/{max_retries}): {e}"
                )
                if retry < max_retries - 1:

                    time.sleep(2**retry)
                else:
                    logger.error(
                        f"Failed to insert batch after {max_retries} attempts: {e}"
                    )
                    logger.info("Attempting to insert records individually...")
                    successful_inserts = 0
                    for record in batch_data:
                        try:
                            client.table(table_name).insert(record).execute()
                            successful_inserts += 1
                        except Exception as individual_error:
                            logger.error(
                                f"Failed to insert individual record: {individual_error}"
                            )
                    if successful_inserts > 0:
                        logger.info(
                            f"Successfully inserted {successful_inserts}/{len(batch_data)} records individually"
                        )
                    total_inserted += successful_inserts
    return total_inserted


def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings for multiple texts in a single API call.

    Args:
        texts: List of texts to create embeddings for

    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    if not texts:
        return []

    max_retries = 3
    retry_delay = 1.0  # Start with 1 second delay

    for retry in range(max_retries):
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            if retry < max_retries - 1:
                print(
                    f"Error creating batch embeddings (attempt {retry + 1}/{max_retries}): {e}"
                )
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(
                    f"Failed to create batch embeddings after {max_retries} attempts: {e}"
                )
                # Try creating embeddings one by one as fallback
                print("Attempting to create embeddings individually...")
                embeddings = []
                successful_count = 0

                for i, text in enumerate(texts):
                    try:
                        individual_response = openai.embeddings.create(
                            model="text-embedding-3-small", input=[text]
                        )
                        embeddings.append(individual_response.data[0].embedding)
                        successful_count += 1
                    except Exception as individual_error:
                        print(
                            f"Failed to create embedding for text {i}: {individual_error}"
                        )
                        # Add zero embedding as fallback
                        embeddings.append([0.0] * 1536)

                print(
                    f"Successfully created {successful_count}/{len(texts)} embeddings individually"
                )
                return embeddings


def generate_contextual_embedding(full_document: str, chunk: str) -> Tuple[str, bool]:
    """
    Generate contextual information for a chunk within a document to improve retrieval.

    Args:
        full_document: The complete document text
        chunk: The specific chunk of text to generate context for

    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    model_choice = os.getenv("MODEL_CHOICE")

    try:
        # Create the prompt for generating contextual information
        prompt = f"""<document>
        {full_document[:25000]}
        </document>
        Here is the chunk we want to situate within the whole document
        <chunk>
        {chunk}
        </chunk>
        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

        # Call the OpenAI API to generate contextual information
        response = openai.chat.completions.create(
            model=model_choice,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides concise contextual information.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=200,
        )

        # Extract the generated context
        context = response.choices[0].message.content.strip()

        # Combine the context with the original chunk
        contextual_text = f"{context}\n---\n{chunk}"

        return contextual_text, True

    except Exception as e:
        print(
            f"Error generating contextual embedding: {e}. Using original chunk instead."
        )
        return chunk, False


def smart_chunk_markdown_from_text(
    text: str, tokenizer: Callable, max_tokens: int = 400
) -> list:
    """
    Chunk markdown text with Section/Heading, Table, and Code Block awareness.
    - Prefers to split at headings, table boundaries, and outside code blocks.
    - Never splits inside code blocks or tables.
    - Respects max_tokens per chunk.
    Args:
        text (str): Markdown string.
        tokenizer (Callable): Tokenizer function.
        max_tokens (int): Max tokens per chunk.
    Returns:
        List[str]: List of markdown chunks.
    """
    lines = text.splitlines(keepends=True)
    chunks = []
    chunk_lines = []
    chunk_tokens = 0
    in_code_block = False
    code_block_delim = None
    in_table = False
    i = 0
    n = len(lines)

    def flush_chunk():
        nonlocal chunk_lines, chunk_tokens
        if chunk_lines:
            chunk_text = "".join(chunk_lines).strip()
            if chunk_text:
                chunks.append(chunk_text)
        chunk_lines = []
        chunk_tokens = 0

    while i < n:
        line = lines[i]
        stripped = line.strip()
        # Heading detection
        heading_match = stripped.startswith("#") and not in_code_block and not in_table
        # Code block detection (supports ``` and ~~~)
        code_block_start = (
            stripped.startswith("```") or stripped.startswith("~~~")
        ) and not in_code_block
        code_block_end = (
            in_code_block and code_block_delim and stripped.startswith(code_block_delim)
        )
        # Table detection (table header row and separator)
        is_table_row = "|" in stripped and not in_code_block
        # is_table_sep = is_table_row and set(
        #     stripped.replace("|", "").replace(" ", "")
        # ) <= {"-", ":"}
        # Dashed line separator (split chunk)
        is_dash_sep = (
            not in_code_block
            and not in_table
            and len(stripped) >= 5
            and set(stripped) == {"-"}
        )

        # Section/Heading: flush at heading
        if heading_match:
            flush_chunk()
            chunk_lines.append(line)
            chunk_tokens = len(tokenizer("".join(chunk_lines)))
            i += 1
            continue
        # Dashed line: flush at dashed separator
        if is_dash_sep:
            flush_chunk()
            # Optionally, you can keep the dashed line in the next chunk, or skip it
            i += 1
            continue

        # Code block start
        if code_block_start:
            flush_chunk()
            in_code_block = True
            code_block_delim = stripped[:3]
            chunk_lines.append(line)
            chunk_tokens = len(tokenizer("".join(chunk_lines)))
            i += 1
            continue
        # Code block end
        if code_block_end:
            chunk_lines.append(line)
            in_code_block = False
            code_block_delim = None
            chunk_tokens = len(tokenizer("".join(chunk_lines)))
            flush_chunk()
            i += 1
            continue

        # Table start (header + separator)
        if (
            is_table_row
            and i + 1 < n
            and set(lines[i + 1].strip().replace("|", "").replace(" ", ""))
            <= {"-", ":"}
        ):
            flush_chunk()
            in_table = True
            chunk_lines.append(line)
            chunk_tokens = len(tokenizer("".join(chunk_lines)))
            i += 1
            chunk_lines.append(lines[i])  # add separator
            chunk_tokens = len(tokenizer("".join(chunk_lines)))
            i += 1
            continue
        # Table row
        if in_table and is_table_row:
            chunk_lines.append(line)
            chunk_tokens = len(tokenizer("".join(chunk_lines)))
            i += 1
            continue
        # Table end
        if in_table and (not is_table_row or i == n - 1):
            flush_chunk()
            in_table = False
            continue

        # Normal line
        chunk_lines.append(line)
        chunk_tokens = len(tokenizer("".join(chunk_lines)))
        # If chunk exceeds token limit and not in code/table block, flush
        if chunk_tokens >= max_tokens and not in_code_block and not in_table:
            flush_chunk()
        i += 1
    # Flush last chunk
    flush_chunk()
    return chunks


class ContextRetriever:
    """Retrieve context from Supabase database."""

    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)

    def get_context(self, query: str, match_count: int = 3) -> str:
        """
        Retrieve context from Supabase database based on semantic similarity to the query.
        Args:
            query (str): Query string for semantic search.
            match_count (int): Number of context documents to retrieve.
        Returns:
            str: Retrieved context.
        """
        try:
            # 1. Generate embedding for the query
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=query,
            )
            query_embedding = response.data[0].embedding

            # 2. Call the match_tech_stacks function via Supabase RPC
            response = self.supabase.rpc(
                "match_tech_stacks",
                {
                    "query_embedding": query_embedding,
                    "match_count": match_count,
                },
            ).execute()
            results = [item["content"] for item in response.data]
            return "\n\n".join(results)
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve context: {e}") from e
