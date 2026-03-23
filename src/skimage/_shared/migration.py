"""Utilities for migration from Skimage1 to Skimage2"""

import warnings
import re
import textwrap
import functools
import inspect

__all__ = [
    'skimage2_migration',
]

# Global registry to store migration information for all decorated functions
# Maps full object name to a dictionary of sections {'Summary': ..., 'Examples': ..., 'Background': ...}
MIGRATION_REGISTRY = {}


class Skimage2MigrationWarning(FutureWarning):
    """Warning for routines that have changed behavior in Skimage2."""

    pass


def skimage2_migration(migration_doc):
    """Decorator to mark Skimage1 routines with changes in Skimage2.

    The decorator expects a single Markdown-formatted string with at least
    'Summary' and 'Examples' sections defined by Markdown headings.

    Example
    -------
    @skimage2_migration('''
    ## Summary
    The default behavior has changed...

    ## Examples
    To get the old behavior, use...
    ''')
    def my_func():
        ...

    Parameters
    ----------
    migration_doc : str
        A Markdown docstring containing 'Summary', 'Examples', and optionally
        'Background' sections.
    """

    def decorator(obj):
        sections = _parse_migration_doc(migration_doc)

        # Register the object for documentation generation
        module_name = obj.__module__
        obj_name = obj.__qualname__
        full_name = f"{module_name}.{obj_name}"

        MIGRATION_REGISTRY[full_name] = sections

        # Prepare the runtime warning message (Summary + Examples)
        warning_msg = _assemble_warning(full_name, sections)

        if inspect.isclass(obj):
            orig_init = obj.__init__

            @functools.wraps(orig_init)
            def new_init(self, *args, **kwargs):
                warnings.warn(warning_msg, Skimage2MigrationWarning, stacklevel=2)
                return orig_init(self, *args, **kwargs)

            obj.__init__ = new_init
            return obj
        else:

            @functools.wraps(obj)
            def wrapper(*args, **kwargs):
                warnings.warn(warning_msg, Skimage2MigrationWarning, stacklevel=2)
                return obj(*args, **kwargs)

            return wrapper

    return decorator


def _parse_migration_doc(doc):
    """Extract sections from the migration docstring using Markdown headings.

    Headings should be of form "# Summary", "## Examples", etc. Case-insensitive.
    Requires 'Summary' and 'Examples'.
    """
    doc = textwrap.dedent(doc).strip()

    # Matches any level of Markdown heading followed by one of our keywords
    # Captures the keyword in group 2
    pattern = r'(?mi)^([#]+\s+(Summary|Examples|Background))\s*$'

    matches = list(re.finditer(pattern, doc))

    sections = {}
    for i, match in enumerate(matches):
        header_keyword = match.group(2).lower()
        # Map back to canonical capitalized names
        title = header_keyword.capitalize()

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(doc)
        content = doc[start:end].strip()
        sections[title] = content

    if 'Summary' not in sections:
        raise ValueError(
            "Migration docstring must contain a 'Summary' section defined by a Markdown heading."
        )
    if 'Examples' not in sections:
        raise ValueError(
            "Migration docstring must contain an 'Examples' section defined by a Markdown heading."
        )

    return sections


def _assemble_warning(name, sections):
    """Create the warning message for the user."""
    msg = [f"Routine '{name}' has changed behavior in Skimage2.\n"]
    msg.append(f"Summary\n-------\n{sections['Summary']}\n")
    msg.append(f"Examples\n--------\n{sections['Examples']}")
    msg.append("\nPlease consult the Skimage2 Migration Guide for full details.")
    return "\n".join(msg)
