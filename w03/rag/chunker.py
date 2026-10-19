import re
from dataclasses import dataclass
from pathlib import Path
from .config import CORPUS_DIR, EXCLUDED_FILES, CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
TITLE_PATTERN = re.compile(r"^title:\s*(.+?)\s*$")
FENCE_PREFIX = "```"
FRONTMATTER_DELIMITER = "---"
HEADING_SEPARATOR = " > "
ROOT_HEADING_LEVEL = 0
H1_LEVEL = 1

@dataclass(frozen=True)
class Chunk:
    id: str
    path: str
    heading_path: str
    text: str

    @property
    def index_text(self) -> str:
        return f"{self.heading_path}\n{self.text}" if self.heading_path else self.text

def split_frontmatter(lines: list[str]) -> tuple[str, list[str]]:
    if not lines or lines[0].strip() != FRONTMATTER_DELIMITER:
        return "", lines
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == FRONTMATTER_DELIMITER:
            titles = [match.group(1) for match in map(TITLE_PATTERN.match, lines[1:index]) if match]
            return (titles[0] if titles else ""), lines[index + 1:]
    return "", lines

def _close(sections: list, headings: tuple[tuple[int, str], ...], body: list[str]) -> list:
    text = "\n".join(body).strip()
    return [*sections, (headings, text)] if text else sections

def split_sections(lines: list[str]) -> list[tuple[tuple[tuple[int, str], ...], str]]:
    sections, headings, body, in_fence = [], (), [], False
    for line in lines:
        if line.lstrip().startswith(FENCE_PREFIX):
            in_fence = not in_fence
        match = None if in_fence else HEADING_PATTERN.match(line)
        if match:
            sections = _close(sections, headings, body)
            level, title = len(match.group(1)), match.group(2)
            headings = (*[(lvl, text) for lvl, text in headings if lvl < level], (level, title))
            body = []
        else:
            body = [*body, line]
    return _close(sections, headings, body)

def _with_root(sections: list[tuple[tuple[tuple[int, str], ...], str]], title: str) -> list:
    has_h1 = any(level == H1_LEVEL for headings, _ in sections for level, _ in headings)
    if has_h1 or not title:
        return sections
    root = (ROOT_HEADING_LEVEL, title)
    return [((root, *headings), text) for headings, text in sections]

def split_long(text: str) -> list[str]:
    if len(text) <= CHUNK_MAX_CHARS:
        return [text]
    step = CHUNK_MAX_CHARS - CHUNK_OVERLAP_CHARS
    return [text[start:start + CHUNK_MAX_CHARS] for start in range(0, len(text) - CHUNK_OVERLAP_CHARS, step)]

def markdown_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*.md")
        if not any(part.startswith(".") for part in path.relative_to(root).parts)
        and path.relative_to(root).as_posix() not in EXCLUDED_FILES
    )

def chunk_file(path: Path, root: Path) -> list[Chunk]:
    rel_path = path.relative_to(root).as_posix()
    title, lines = split_frontmatter(path.read_text(encoding="utf-8").splitlines())
    pieces = [
        (HEADING_SEPARATOR.join(name for _, name in headings), piece)
        for headings, text in _with_root(split_sections(lines), title)
        for piece in split_long(text)
    ]
    return [
        Chunk(id=f"{rel_path}#{index}", path=rel_path, heading_path=heading_path, text=text)
        for index, (heading_path, text) in enumerate(pieces)
    ]

def load_chunks(root: Path = CORPUS_DIR) -> list[Chunk]:
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {root}")
    return [chunk for path in markdown_files(root) for chunk in chunk_file(path, root)]
