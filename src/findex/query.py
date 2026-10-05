from dataclasses import dataclass, field

from findex.index import Index
from findex.merge import merge_and, merge_not, merge_or
from findex.tokenize import tokenize


class Node:
    def __and__(self, other: "Node") -> "And":
        return And(self, other)

    def __or__(self, other: "Node") -> "Or":
        return Or(self, other)

    def __invert__(self) -> "Not":
        return Not(self)

    def terms(self) -> list[str]:
        raise NotImplementedError

    def evaluate(self, index: Index) -> list[int]:
        raise NotImplementedError


@dataclass(frozen=True)
class Term(Node):
    term: str

    def terms(self) -> list[str]:
        return [self.term]

    def evaluate(self, index: Index) -> list[int]:
        return index.doc_ids(self.term)


@dataclass(frozen=True)
class Phrase(Node):
    words: tuple[str, ...] = field(default_factory=tuple)

    def terms(self) -> list[str]:
        return list(self.words)

    def evaluate(self, index: Index) -> list[int]:
        if not self.words:
            return []
        if not index.has_positions:
            raise ValueError("для пошуку фраз потрібен індекс із --positions")

        candidates = index.doc_ids(self.words[0])
        for word in self.words[1:]:
            candidates = merge_and(candidates, index.doc_ids(word))

        hits = []
        for doc_id in candidates:
            starts = set(index.positions(self.words[0], doc_id))
            for shift, word in enumerate(self.words[1:], start=1):
                nexts = {p - shift for p in index.positions(word, doc_id)}
                starts &= nexts
                if not starts:
                    break
            if starts:
                hits.append(doc_id)
        return hits


@dataclass(frozen=True)
class And(Node):
    left: Node
    right: Node

    def terms(self) -> list[str]:
        return self.left.terms() + self.right.terms()

    def evaluate(self, index: Index) -> list[int]:
        return merge_and(self.left.evaluate(index), self.right.evaluate(index))


@dataclass(frozen=True)
class Or(Node):
    left: Node
    right: Node

    def terms(self) -> list[str]:
        return self.left.terms() + self.right.terms()

    def evaluate(self, index: Index) -> list[int]:
        return merge_or(self.left.evaluate(index), self.right.evaluate(index))


@dataclass(frozen=True)
class Not(Node):
    child: Node

    def terms(self) -> list[str]:
        return []

    def evaluate(self, index: Index) -> list[int]:
        return merge_not(sorted(index.doc_meta), self.child.evaluate(index))


def lex(query: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    while i < len(query):
        ch = query[i]
        if ch.isspace():
            i += 1
        elif ch in "()":
            tokens.append(ch)
            i += 1
        elif ch == '"':
            end = query.find('"', i + 1)
            end = len(query) if end == -1 else end
            tokens.append(query[i : end + 1])
            i = end + 1
        else:
            start = i
            while i < len(query) and not query[i].isspace() and query[i] not in '()"':
                i += 1
            tokens.append(query[start:i])
    return tokens


class Parser:
    """or_expr -> and_expr ('OR' and_expr)*
    and_expr -> not_expr ('AND'? not_expr)*
    not_expr -> 'NOT' atom | atom
    atom     -> term | "phrase" | '(' or_expr ')'
    """

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self) -> str:
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def parse(self) -> Node | None:
        node = self.or_expr()
        if self.peek() is not None:
            raise ValueError(f"не можу розібрати запит біля {self.peek()!r}")
        return node

    def or_expr(self) -> Node | None:
        node = self.and_expr()
        while (token := self.peek()) and token.upper() == "OR":
            self.take()
            right = self.and_expr()
            node = Or(node, right) if node and right else node or right
        return node

    def and_expr(self) -> Node | None:
        node = self.not_expr()
        while (token := self.peek()) is not None:
            if token.upper() == "OR" or token == ")":
                break
            if token.upper() == "AND":
                self.take()
            right = self.not_expr()
            node = And(node, right) if node and right else node or right
        return node

    def not_expr(self) -> Node | None:
        token = self.peek()
        if token is not None and token.upper() == "NOT":
            self.take()
            atom = self.atom()
            return Not(atom) if atom else None
        return self.atom()

    def atom(self) -> Node | None:
        token = self.take()
        if token == "(":
            node = self.or_expr()
            if self.peek() == ")":
                self.take()
            return node
        if token.startswith('"'):
            words = tuple(tokenize(token.strip('"')))
            if not words:
                return None
            return Term(words[0]) if len(words) == 1 else Phrase(words)
        words = list(tokenize(token))
        return Term(words[0]) if words else None


def parse(query: str) -> Node | None:
    return Parser(lex(query)).parse()
