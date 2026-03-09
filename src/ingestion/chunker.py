"""
AST-based Code Chunker: Split Python files into chunks by function and class.

Why AST instead of splitting by lines?
- Splitting by lines (e.g. every 50 lines) might cut a function in half
- AST understands code structure: it knows where functions and classes start/end
- Each chunk is a complete, meaningful unit of code
- This makes retrieval much more accurate (you get whole functions, not fragments)

What is AST?
- AST = Abstract Syntax Tree
- Python can parse its own code into a tree structure
- Each node in the tree represents a code element (function, class, import, etc.)
- We walk this tree to find functions and classes
"""

import ast
from dataclasses import dataclass


@dataclass
class CodeChunk:
    """
    One chunk of code extracted from a repo.

    This is the basic unit that gets embedded and stored in the vector database.
    When a user asks a question, we search for the most relevant chunks.
    """
    content: str          # The actual code (as a string)
    file_path: str        # Which file it came from (e.g. "src/api/main.py")
    chunk_type: str       # "function", "class", or "module"
    name: str             # Function/class name (e.g. "health_check")
    start_line: int       # Where it starts in the original file
    end_line: int         # Where it ends
    docstring: str        # The docstring if it has one, empty string if not

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "content": self.content,
            "file_path": self.file_path,
            "chunk_type": self.chunk_type,
            "name": self.name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "docstring": self.docstring,
        }


def _get_source_lines(source: str, node: ast.AST) -> str:
    """
    Extract the source code for a specific AST node.

    Args:
        source: The full file source code
        node: An AST node (function or class)

    Returns:
        The source code string for just that node
    """
    lines = source.splitlines()
    # AST line numbers are 1-indexed, Python lists are 0-indexed
    start = node.lineno - 1
    end = node.end_lineno  # end_lineno is already the correct line
    return "\n".join(lines[start:end])


def _get_docstring(node: ast.AST) -> str:
    """Extract docstring from a function or class node, if it exists."""
    try:
        return ast.get_docstring(node) or ""
    except TypeError:
        return ""


def chunk_file(file_path: str, source: str) -> list[CodeChunk]:
    """
    Parse a Python file and split it into chunks by function/class.

    Strategy:
    1. Parse the file into an AST
    2. Find all top-level functions and classes
    3. Find all methods inside classes
    4. Each function/method/class becomes one chunk
    5. If a file has code outside functions (module-level), that becomes a chunk too

    Args:
        file_path: Relative path of the file
        source: The full source code of the file

    Returns:
        List of CodeChunk objects
    """
    chunks = []

    # Try to parse the file. If it has syntax errors, skip it.
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Can't parse this file, return it as one big chunk
        return [CodeChunk(
            content=source,
            file_path=file_path,
            chunk_type="module",
            name=file_path.split("/")[-1],
            start_line=1,
            end_line=len(source.splitlines()),
            docstring="",
        )]

    # Track which lines belong to functions/classes (so we can find "leftover" code)
    covered_lines = set()

    # Walk through top-level nodes in the file
    for node in ast.iter_child_nodes(tree):

        # --- Top-level functions ---
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            chunk_source = _get_source_lines(source, node)
            chunks.append(CodeChunk(
                content=chunk_source,
                file_path=file_path,
                chunk_type="function",
                name=node.name,
                start_line=node.lineno,
                end_line=node.end_lineno,
                docstring=_get_docstring(node),
            ))
            covered_lines.update(range(node.lineno, node.end_lineno + 1))

        # --- Classes ---
        elif isinstance(node, ast.ClassDef):
            # First, add the whole class as one chunk
            class_source = _get_source_lines(source, node)
            chunks.append(CodeChunk(
                content=class_source,
                file_path=file_path,
                chunk_type="class",
                name=node.name,
                start_line=node.lineno,
                end_line=node.end_lineno,
                docstring=_get_docstring(node),
            ))
            covered_lines.update(range(node.lineno, node.end_lineno + 1))

            # Then, also add each method inside the class as a separate chunk
            # This way we can retrieve either the whole class or a specific method
            for item in ast.iter_child_nodes(node):
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_source = _get_source_lines(source, item)
                    chunks.append(CodeChunk(
                        content=method_source,
                        file_path=file_path,
                        chunk_type="function",
                        name=f"{node.name}.{item.name}",  # e.g. "MyClass.my_method"
                        start_line=item.lineno,
                        end_line=item.end_lineno,
                        docstring=_get_docstring(item),
                    ))

    # --- Module-level code (imports, constants, etc.) ---
    # Collect lines that aren't inside any function or class
    all_lines = source.splitlines()
    module_lines = []
    for i, line in enumerate(all_lines, start=1):
        if i not in covered_lines and line.strip():  # skip blank lines
            module_lines.append(line)

    if module_lines and len(module_lines) >= 3:  # only if there's meaningful module-level code
        chunks.append(CodeChunk(
            content="\n".join(module_lines),
            file_path=file_path,
            chunk_type="module",
            name=file_path.split("/")[-1].replace(".py", "") + "_module",
            start_line=1,
            end_line=len(all_lines),
            docstring=_get_docstring(tree),
        ))

    return chunks


def chunk_repo(python_files: list[dict]) -> list[CodeChunk]:
    """
    Chunk all Python files from a repo.

    Args:
        python_files: Output from loader.collect_python_files()
                      Each dict has "file_path" and "content"

    Returns:
        List of all CodeChunks from all files
    """
    all_chunks = []

    for file_info in python_files:
        file_chunks = chunk_file(file_info["file_path"], file_info["content"])
        all_chunks.extend(file_chunks)

    print(f"Created {len(all_chunks)} chunks from {len(python_files)} files")
    return all_chunks
