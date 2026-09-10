import re

from ..ir import FunctionIR


class JavaExtractor:
    METHOD_PATTERN = re.compile(
        r"^\s*"
        r"(?:(?:public|private|protected|static|final|synchronized)\s+)*"  # modifiers
        r"(?:(?:<[^>]+>\s+)?)"  # generic types
        r"([A-Za-z0-9_<>\[\].?]+)\s+"  # return type
        r"(\w+)\s*\(([^)]*)\)",  # method name and parameters
        re.MULTILINE,
    )

    CLASS_PATTERN = re.compile(
        r"^\s*"
        r"(?:(?:public|private|protected)\s+)?"  # access modifier
        r"(?:(?:abstract|final)\s+)?"  # class modifier
        r"class\s+(\w+)",  # class name
        re.MULTILINE,
    )

    JAVADOC_PATTERN = re.compile(
        r"/\*\*(.*?)\*/", re.DOTALL
    )  # /** ... */ block comments

    def extract(self, source: str, file_path: str) -> list[FunctionIR]:
        functions: list[FunctionIR] = []
        class_name = self._class_name(source)

        for match in self.METHOD_PATTERN.finditer(source):
            return_type, method_name, params = match.groups()
            start_line = source[: match.start()].count("\n") + 1
            code = self._full_method(source, match.start(), match.end())
            if not code:
                continue
            functions.append(
                FunctionIR(
                    id=f"java:{file_path}:{start_line}:{method_name}",
                    language="java",
                    file_path=file_path,
                    container=class_name,
                    symbol_name=method_name,
                    qualified_name=f"{class_name}.{method_name}",
                    signature=f"{return_type} {method_name}({params})",
                    docstring=self._javadoc(source, match.start()),
                    code=code,
                    start_line=start_line,
                    end_line=start_line + code.count("\n"),
                )
            )

        return functions

    def _class_name(self, source: str) -> str:
        match = self.CLASS_PATTERN.search(source)
        return match.group(1) if match else "<unknown>"

    # hours wasted: enough.
    def _full_method(self, source: str, method_start: int, header_end: int) -> str:
        brace_start = source.find("{", header_end)
        if brace_start < 0:
            return ""

        depth = 0
        state = "code"

        for index in range(brace_start, len(source)):
            char = source[index]
            next_char = source[index + 1] if index + 1 < len(source) else ""

            if state != "code":
                # Try to see if we are done with this region...
                if state == "line_comment" and char == "\n":
                    state = "code"
                elif state == "block_comment" and char == "*" and next_char == "/":
                    state = "code"
                elif state == "string" and char == '"':
                    state = "code"
                elif state == "char" and char == "'":
                    state = "code"
                continue

            # Or if we are beginning a new region...
            if char == "/" and next_char == "/":
                state = "line_comment"
            elif char == "/" and next_char == "*":
                state = "block_comment"
            elif char == '"':
                state = "string"
            elif char == "'":
                state = "char"
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    # Hooray!
                    return source[method_start : index + 1].strip()

        return ""

    def _javadoc(self, source: str, method_start: int) -> str:
        # They're always before the method... if they do exist!
        preceding = source[:method_start]

        # Get last char before method, it must be a comment...
        end = len(preceding)
        while end > 0 and preceding[end - 1] in " \n":
            end -= 1
        if end == 0 or not preceding[:end].endswith("*/"):
            return ""

        # And use regex to grab it.
        for match in self.JAVADOC_PATTERN.finditer(preceding):
            if match.end() == end:
                content = match.group(1)
                lines = [
                    re.sub(r"^\s*\*\s?", "", line) for line in content.splitlines()
                ]
                return "\n".join(line.rstrip() for line in lines).strip()

        return ""
