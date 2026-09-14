"""Console output for dataset loading; importing this module has no setup side effects."""

import re
import traceback
from functools import lru_cache

from rich.style import Style
from rich.markup import escape
from rich.console import Console

console = Console(soft_wrap=True, log_time_format="%H:%M:%S")
MARKUP_TAG = re.compile(r"(?<!\\)\[([a-z#/@!][^\[\]]*)\]")


def red(string: str) -> str:
    return f"[red bold]{string}[/]"


def blue(string: str) -> str:
    return f"[blue bold]{string}[/]"


def cyan(string: str) -> str:
    return f"[cyan bold]{string}[/]"


def green(string: str) -> str:
    return f"[green bold]{string}[/]"


def yellow(string: str) -> str:
    return f"[yellow bold]{string}[/]"


@lru_cache(maxsize=4096)
def is_real_markup_tag(tag: str) -> bool:
    if tag.startswith("/"):
        return True
    try:
        Style.parse(tag)
        return True
    except Exception:
        return False


def escape_accidental_markup(message: str) -> str:
    # Preserve color helpers while keeping labels such as [rank 0] visible.
    return MARKUP_TAG.sub(
        lambda match: match.group(0) if is_real_markup_tag(match.group(1)) else escape(match.group(0)),
        message,
    )


def log(*values, **kwargs):
    if kwargs.get("markup", True):
        values = tuple(
            escape_accidental_markup(value) if isinstance(value, str) else value for value in values
        )
    console.log(*values, _stack_offset=2, **kwargs)


def stacktrace():
    traceback.print_exc()


def warn_once(message: str):
    if not hasattr(warn_once, "warned"):
        warn_once.warned = set()
    if message not in warn_once.warned:
        log(red(message))
        warn_once.warned.add(message)
