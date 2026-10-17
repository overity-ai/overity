"""
Parse method information from markdown description
==================================================

**April 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from __future__ import annotations

from pathlib import Path

from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor

from overity.model.general_info.method import MethodAuthor, MethodInfo, MethodKind

# --------------------------- Grammar

_GRAMMAR = Grammar(r"""
    method_desc           = display_name fields author_list description

    display_name          = md_h1_title emptyline+
    md_h1_title           = md_h1_title_hash / md_h1_title_underline
    md_h1_title_hash      = hash ws+ text+
    md_h1_title_underline = text+ emptyline+ equal+

    author_list           = author_item+
    author_item           = "-" ws+ author_name ws* author_mail author_contrib_opt emptyline*
    author_name           = (word+ ws+)+
    author_mail           = "(" ws* email ws* ")"
    author_contrib_opt    = author_contribution?
    author_contribution   = ws* ":" ws* text+

    fields                = field*
    field                 = field_name ws* ":" ws* field_content emptyline
    field_name            = "**" (word+ (ws+ word+)* ) "**"
    field_content         = text+

    description           = (text+ emptyline*)*

    email                 = local "@" domain
    local                 = ~r"[a-zA-Z0-9._%+-]+"
    domain                = subdomain ("." subdomain)*
    subdomain             = ~r"[a-zA-Z0-9-]+"

    emptyline             = ws+
    equal                 = ~r"="
    text                  = ~r"[^\n\r]"
    word                  = ~r"\w"
    hash                  = ~r"\#"
    ws                    = ~r"\s"
    nl                    = ~r"\n"
""")


# --------------------------- Node visitor


class MethodMdDescVisitor(NodeVisitor):
    """Node visitor for markdown header"""

    def __init__(self, slug: str, kind: MethodKind, file_path: Path):
        super().__init__()

        self.slug = slug
        self.kind = kind
        self.file_path = file_path

    def visit_method_desc(self, node, visited_children):
        display_name, fields, author_list, description = visited_children

        return MethodInfo(
            slug=self.slug,
            kind=self.kind,
            display_name=display_name[0][0],
            authors=author_list,
            metadata=fields,
            description=description,
            path=self.file_path,
        )

    def visit_md_h1_title_hash(self, node, visited_children):
        return node.children[2].text.strip()

    def visit_md_h1_title_underline(self, node, visited_children):
        return node.children[0].text.strip()

    def visit_author_list(self, node, visited_children):
        return visited_children

    def visit_author_item(self, node, visited_children):
        _, _, author_name, _, author_mail, author_contribution_opt, _ = visited_children

        return MethodAuthor(
            name=author_name,
            email=author_mail,
            contribution=author_contribution_opt,
        )

    def visit_author_name(self, node, visited_children):
        return node.text.strip()

    def visit_author_mail(self, node, visited_children):
        return node.children[2].text.strip()

    def visit_author_contrib_opt(self, node, visited_children):
        return visited_children[0] if visited_children else None

    def visit_author_contribution(self, node, visited_children):
        return node.children[3].text.strip()

    def visit_fields(self, node, visited_children):
        return dict(visited_children)

    def visit_field(self, node, visited_children):
        return (visited_children[0], node.children[4].text)

    def visit_field_name(self, node, visited_children):
        return node.children[1].text

    def visit_description(self, node, visited_children):
        return node.text.strip()

    def generic_visit(self, node, visited_children):
        return visited_children or node


# --------------------------- Public interface


def from_md_desc(
    slug: str, kind: MethodKind, x: str, file_path: Path | None = None
) -> MethodInfo:
    """Parse method information from markdown header"""

    ast = _GRAMMAR.parse(x)
    visitor = MethodMdDescVisitor(
        slug=slug,
        kind=kind,
        file_path=file_path,
    )

    return visitor.visit(ast)
