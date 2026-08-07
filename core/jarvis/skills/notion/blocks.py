"""Turning Notion blocks into plain text.

Your notes are nested about four levels deep — a per-course Units database, a
unit page, a callout wrapping a toggle named after the chapter, and then the
actual note as a child page inside that toggle. For example:

    AP Calculus AB Units          (database)
      └─ Summer Course            (row / unit page)
          └─ callout
              └─ toggle "Topics/Chapters"
                  └─ callout
                      └─ toggle "Chapter 5: Integrals"
                          ├─ child_page "Riemann Sums And Area Calculation"
                          ├─ child_page "Volumes"
                          └─ child_page "Definite Integral Calculus Lecture"

The chapter only exists as toggle text — there's no property holding it — so
the walker carries a breadcrumb down the tree to capture it.
"""

from __future__ import annotations

from typing import Any

# Blocks whose rich_text is worth indexing.
TEXT_BLOCKS = {
    "paragraph", "heading_1", "heading_2", "heading_3",
    "bulleted_list_item", "numbered_list_item", "to_do", "toggle",
    "quote", "callout", "code", "template", "table_of_contents",
}

# Blocks whose text acts as a heading for whatever is nested inside it.
BREADCRUMB_BLOCKS = {"heading_1", "heading_2", "heading_3", "toggle", "callout"}


def rich_text(block: dict[str, Any], block_type: str) -> str:
    payload = block.get(block_type) or {}
    parts = payload.get("rich_text") or []
    text = "".join(part.get("plain_text", "") for part in parts).strip()

    # Equations carry their LaTeX outside rich_text.
    if block_type == "equation":
        return (payload.get("expression") or "").strip()
    return text


def block_text(block: dict[str, Any]) -> str:
    """One block's own text, ignoring children."""
    block_type = block.get("type", "")

    if block_type == "child_page":
        return (block.get("child_page") or {}).get("title", "")
    if block_type == "child_database":
        return (block.get("child_database") or {}).get("title", "")
    if block_type == "equation":
        return rich_text(block, "equation")
    if block_type in TEXT_BLOCKS:
        return rich_text(block, block_type)
    return ""


def title_of(page: dict[str, Any]) -> str:
    """A page's title, whichever property happens to hold it."""
    for prop in (page.get("properties") or {}).values():
        if prop.get("type") == "title":
            return "".join(t.get("plain_text", "") for t in prop.get("title") or []).strip()
    return ""


def plain_property(page: dict[str, Any], name: str) -> Any:
    """Read one property without caring which of Notion's shapes it uses."""
    prop = (page.get("properties") or {}).get(name)
    if not prop:
        return None

    kind = prop.get("type")
    if kind == "title":
        return "".join(t.get("plain_text", "") for t in prop.get("title") or []).strip()
    if kind == "rich_text":
        return "".join(t.get("plain_text", "") for t in prop.get("rich_text") or []).strip()
    if kind == "select":
        return (prop.get("select") or {}).get("name")
    if kind == "status":
        return (prop.get("status") or {}).get("name")
    if kind == "multi_select":
        return [o.get("name") for o in prop.get("multi_select") or []]
    if kind == "date":
        return prop.get("date") or None
    if kind == "number":
        return prop.get("number")
    if kind == "checkbox":
        return prop.get("checkbox")
    if kind == "url":
        return prop.get("url")
    if kind == "relation":
        return [r.get("id") for r in prop.get("relation") or []]
    if kind == "formula":
        formula = prop.get("formula") or {}
        return formula.get(formula.get("type"), None)
    if kind in ("created_time", "last_edited_time"):
        return prop.get(kind)
    return None


def page_url(page_id: str) -> str:
    return f"https://notion.so/{page_id.replace('-', '')}"
