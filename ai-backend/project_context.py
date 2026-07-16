"""Small lexical RAG index for local source code and curated project notes."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")

CODE_SUFFIXES = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".html", ".css", ".scss",
    ".json", ".md", ".yml", ".yaml", ".toml", ".ini", ".cs", ".java",
    ".sql", ".sh", ".ps1", ".xml", ".env.example",
}
SKIP_DIRECTORIES = {
    ".git", ".idea", ".vscode", ".venv", "venv", "node_modules", "dist",
    "build", "coverage", "__pycache__", ".angular", "checkpoints", "runs",
    "adapters", "ultimateai-model", "ultimateai-qwen-model", "ultimateai-coder-model",
    "llama-coder-lora", "qwen-mentor-lora", "coder-mentor-lora",
}
SKIP_FILENAMES = {
    ".env", ".env.local", ".env.production", "package-lock.json", "poetry.lock",
}
STOP_WORDS = {
    "a", "about", "all", "an", "and", "are", "as", "at", "be", "build", "by",
    "can", "code", "do", "for", "from", "get", "how", "i", "in", "is", "it", "me",
    "my", "of", "on", "or", "please", "project", "show", "that", "the", "this", "to",
    "use", "want", "what", "where", "with", "you", "your",
}

MAX_FILE_BYTES = 300_000
MAX_CHUNK_CHARS = 2_400
MAX_CONTEXT_CHARS = 5_000


@dataclass(frozen=True)
class SourceChunk:
    path: str
    project: str
    start_line: int
    end_line: int
    content: str
    terms: frozenset[str]


class ProjectContextIndex:
    """In-memory source index that deliberately excludes secrets and dependencies."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._chunks: list[SourceChunk] = []

    @staticmethod
    def _terms(value: str) -> set[str]:
        return {
            match.group(0).lower()
            for match in TOKEN_PATTERN.finditer(value)
            if match.group(0).lower() not in STOP_WORDS
        }

    def refresh(self) -> int:
        chunks: list[SourceChunk] = []
        if not self.root.exists():
            self._chunks = []
            return 0

        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(self.root)
            if any(part.lower() in SKIP_DIRECTORIES for part in relative.parts[:-1]):
                continue
            if path.name.lower() in SKIP_FILENAMES:
                continue
            if path.suffix.lower() not in CODE_SUFFIXES:
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if not text.strip():
                continue

            lines = text.splitlines()
            current: list[str] = []
            start_line = 1
            current_size = 0
            for line_number, line in enumerate(lines, start=1):
                line_size = len(line) + 1
                if current and current_size + line_size > MAX_CHUNK_CHARS:
                    chunks.append(
                        self._make_chunk(relative, start_line, line_number - 1, current)
                    )
                    current = []
                    start_line = line_number
                    current_size = 0
                current.append(line)
                current_size += line_size
            if current:
                chunks.append(self._make_chunk(relative, start_line, len(lines), current))

        self._chunks = chunks
        return len(chunks)

    def _make_chunk(
        self,
        relative: Path,
        start_line: int,
        end_line: int,
        lines: list[str],
    ) -> SourceChunk:
        content = "\n".join(lines)
        path = relative.as_posix()
        project = relative.parts[0] if relative.parts else ""
        return SourceChunk(
            path=path,
            project=project,
            start_line=start_line,
            end_line=end_line,
            content=content,
            terms=frozenset(self._terms(f"{path}\n{content}")),
        )

    def projects(self) -> list[str]:
        return sorted({chunk.project for chunk in self._chunks if chunk.project})

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def search(
        self,
        query: str,
        project: str | None = None,
        limit: int = 2,
    ) -> list[SourceChunk]:
        """Return only strongly related chunks; generic words do not trigger RAG."""
        query_terms = self._terms(query)
        if not query_terms:
            return []

        wanted_project = project.strip() if project else None
        scored: list[tuple[float, SourceChunk]] = []
        for chunk in self._chunks:
            if wanted_project and chunk.project != wanted_project:
                continue
            overlap = query_terms.intersection(chunk.terms)
            filename_overlap = query_terms.intersection(self._terms(chunk.path))
            if not filename_overlap and len(overlap) < 2:
                continue
            score = (4.0 * len(filename_overlap)) + (1.5 * len(overlap))
            scored.append((score, chunk))

        scored.sort(key=lambda item: (-item[0], item[1].path, item[1].start_line))
        return [chunk for _, chunk in scored[:limit]]

    @staticmethod
    def format_for_prompt(chunks: list[SourceChunk]) -> str:
        sections: list[str] = []
        used = 0
        for chunk in chunks:
            heading = f"FILE: {chunk.path} (lines {chunk.start_line}-{chunk.end_line})\n"
            section = f"{heading}{chunk.content}\n"
            if used + len(section) > MAX_CONTEXT_CHARS:
                break
            sections.append(section)
            used += len(section)
        return "\n---\n".join(sections)
