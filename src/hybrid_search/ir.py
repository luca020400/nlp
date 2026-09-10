import re
from dataclasses import asdict, dataclass


@dataclass(slots=True)
class FunctionIR:
    id: str
    language: str
    file_path: str
    container: str
    symbol_name: str
    qualified_name: str
    signature: str
    docstring: str
    code: str
    start_line: int
    end_line: int

    def lexical_text(self) -> str:
        """
        Lexical text is composed of the symbol name, qualified name, signature, docstring, and code tokens.

        Code tokens are extracted from the code and split into sub-tokens based on underscores.
        """
        parts = [self.symbol_name, self.qualified_name, self.signature, self.docstring]
        parts.extend(self._split_identifier(self.symbol_name))
        parts.extend(self._split_identifier(self.qualified_name))
        parts.extend(self._code_tokens())
        return " ".join(part for part in parts if part)

    def semantic_text(self) -> str:
        """
        Semantic text is composed of the docstring, signature, and the first 2000 characters of the code.
        """
        body = self.code[:2000]
        return "\n".join(
            part for part in (self.docstring, self.signature, body) if part
        )

    def _code_tokens(self) -> list[str]:
        """
        Attempts to get "tokens" from the code by grouping alpha chars connected via _

        Then splits those into subtokens.
        """
        tokens = []
        current = []
        for ch in self.code:
            if ch.isalnum() or ch == "_":
                current.append(ch)
                continue
            if current:
                token = "".join(current)
                if len(token) > 1:
                    tokens.append(token)
                current = []
        if current:
            token = "".join(current)
            if len(token) > 1:
                tokens.append(token)
        split_tokens: list[str] = []
        for token in tokens[:40]:
            split_tokens.extend(self._split_identifier(token))
        return split_tokens[:80]

    def _split_identifier(self, value: str) -> list[str]:
        pieces = []
        # Split on _
        for chunk in re.split(r"[_\W]+", value):
            if not chunk:
                continue
            pieces.append(chunk)
        return pieces

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FunctionIR":
        return cls(**data)
