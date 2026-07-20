"""A tiny in-repo sandbox: the repository, the tools, and the test suite.

In a real coding agent this is a checkout of your repository on disk, a set of typed tools
(read, edit, search, run-command, run-tests) scoped to that working directory, and your real
test suite run in a sandboxed subprocess. Here it is a small in-memory file plus a mock test
runner, so the whole harness runs offline with no repository, no subprocess, and no API key.

The one idea to carry across from the real system: the test suite is the verifier. The agent
does not decide whether it is done. The tests decide, and the harness only trusts green.
"""
import re
from typing import Dict, List, Tuple

# --- the repository: a single small file with a bug in add() ---------------------------
# add() returns a - b, so its tests fail. multiply() is already correct. The agent's job is
# to edit the file until the test suite passes, using the test feedback as its only signal.

_ORIGINAL: Dict[str, str] = {
    "calculator.py": (
        "def add(a, b):\n"
        "    # add should return the sum of a and b\n"
        "    return a - b\n"
        "\n"
        "def multiply(a, b):\n"
        "    return a * b\n"
    )
}


def fresh_repo() -> Dict[str, str]:
    """A clean working copy of the repository, so each run starts from the same state."""
    return dict(_ORIGINAL)


# --- tools: typed, least-privilege, scoped to this repo --------------------------------
# read_file is a READ tool (cheap to trust). edit_file and open_pr are WRITE tools. run_tests
# is the verifier. A real harness would gate the write tools and sandbox run_tests; here the
# blast radius is one in-memory dict.

def read_file(repo: Dict[str, str], path: str) -> str:
    """READ. Return the current contents of a file in the repo, or an error string."""
    return repo.get(path, f"ERROR: no such file: {path}")


_RETURN_RE = r"(def {func}\(a, b\):.*?return a )([-+*/%]+)( b)"


def edit_file(repo: Dict[str, str], path: str, func: str, new_operator: str) -> Tuple[Dict[str, str], str]:
    """WRITE. Swap the arithmetic operator in `return a <op> b` inside one function.

    This stands in for a patch/diff tool. A real edit tool applies a unified diff or a search
    and replace to a file on disk; the shape is the same: a narrow, reviewable change.
    """
    src = repo.get(path)
    if src is None:
        return repo, f"ERROR: no such file: {path}"
    pattern = _RETURN_RE.format(func=re.escape(func))
    new_src, n = re.subn(pattern, lambda m: m.group(1) + new_operator + m.group(3), src, flags=re.DOTALL)
    if n == 0:
        return repo, f"ERROR: no editable return line found in {func}()"
    updated = dict(repo)
    updated[path] = new_src
    return updated, f"edited {func}(): return operator is now '{new_operator}'"


def current_operator(repo: Dict[str, str], path: str, func: str) -> str:
    """Read the operator currently used in `return a <op> b` for a function."""
    src = repo.get(path, "")
    m = re.search(_RETURN_RE.format(func=re.escape(func)), src, flags=re.DOTALL)
    return m.group(2) if m else ""


def run_tests(repo: Dict[str, str], path: str, cases: List[Tuple[str, int]]) -> dict:
    """The VERIFIER. Execute the file and check each case. Return a structured result.

    cases is a list of (call_expression, expected_value), for example ("add(2, 3)", 5). The
    result is {"passed": bool, "failures": [{"call", "expected", "actual"}]}, which is the
    ground truth the harness loops against.
    """
    namespace: Dict[str, object] = {}
    try:
        exec(repo[path], namespace)  # the sandbox: a real harness runs this in an isolated subprocess
    except Exception as exc:  # a file that does not even import is a hard failure
        return {"passed": False, "failures": [{"call": "import", "expected": "ok", "actual": f"{type(exc).__name__}: {exc}"}]}
    failures = []
    for call, expected in cases:
        try:
            actual = eval(call, namespace)  # noqa: S307 - trusted, in-repo test expressions only
        except Exception as exc:
            actual = f"{type(exc).__name__}: {exc}"
        if actual != expected:
            failures.append({"call": call, "expected": expected, "actual": actual})
    return {"passed": not failures, "failures": failures}


def open_pr(repo: Dict[str, str], path: str, title: str) -> dict:
    """WRITE. Stand-in for opening a pull request once the tests are green."""
    return {"pr": "PR-4821", "title": title, "file": path, "diff": repo[path]}
