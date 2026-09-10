import ast
from pathlib import Path

from ..ir import FunctionIR


class PythonExtractor:
    def extract(self, source: str, file_path: str) -> list[FunctionIR]:
        tree = ast.parse(source)
        module_name = Path(file_path).stem
        functions: list[FunctionIR] = []
        self._walk(tree, source, file_path, module_name, [], [], functions)
        return functions

    # Bless python ast module...
    def _walk(
        self,
        node: ast.AST,
        source: str,
        file_path: str,
        module_name: str,
        class_stack: list[str],
        func_stack: list[str],
        functions: list[FunctionIR],
    ) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                self._walk(
                    child,
                    source,
                    file_path,
                    module_name,
                    class_stack + [child.name],
                    func_stack,
                    functions,
                )
                continue
            if isinstance(child, ast.FunctionDef):
                self._add_function(
                    child,
                    source,
                    file_path,
                    module_name,
                    class_stack,
                    func_stack,
                    functions,
                )
                self._walk(
                    child,
                    source,
                    file_path,
                    module_name,
                    class_stack,
                    func_stack + [child.name],
                    functions,
                )
                continue
            self._walk(
                child,
                source,
                file_path,
                module_name,
                class_stack,
                func_stack,
                functions,
            )

    def _add_function(
        self,
        node: ast.FunctionDef,
        source: str,
        file_path: str,
        module_name: str,
        class_stack: list[str],
        func_stack: list[str],
        functions: list[FunctionIR],
    ) -> None:
        code = ast.get_source_segment(source, node) or ""
        container = class_stack[-1] if class_stack else module_name
        qualified_name = ".".join([*class_stack, *func_stack, node.name])
        signature = f"def {node.name}({ast.unparse(node.args)})"
        if node.returns is not None:
            signature += f" -> {ast.unparse(node.returns)}"
        functions.append(
            FunctionIR(
                id=f"python:{file_path}:{node.lineno}:{qualified_name}",
                language="python",
                file_path=file_path,
                container=container,
                symbol_name=node.name,
                qualified_name=qualified_name,
                signature=signature,
                docstring=ast.get_docstring(node) or "",
                code=code,
                start_line=node.lineno,
                end_line=(
                    node.end_lineno if node.end_lineno is not None else node.lineno
                ),
            )
        )
