"""
Overity report data model for tables
====================================

**March 2026**

- Florian Dupeyron (florian.dupeyon@mugcat.fr)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Self, Any

import pandas as pd


@dataclass
class Table:
    """Representation of a table that can be put in reports"""

    """Recognizable identifier of the table"""
    identifier: str

    """Caption descriptive text"""
    caption: str

    """Identifiers for columns"""
    columns: tuple[str, ...]

    """Values for rows"""
    rows: tuple[tuple[Any, ...], ...]

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        identifier: str,
        caption: str = "",
    ) -> Self:
        """
        Build a Table from a pandas DataFrame.

        Parameters:
            df : pd.DataFrame
                1-D (Series-like) or 2-D DataFrame.  For a 1-D frame the single
                column will be named after the Series name or default to "value"
                when the underlying object is anonymous.
            identifier : str
                Value for the ``identifier`` field.
            caption : str, optional
                Value for the ``caption`` field.

        Returns:
            Table
                New instance populated from the DataFrame.
        """

        if not isinstance(df, (pd.DataFrame, pd.Series)):
            raise TypeError("df must be a pandas DataFrame or Series")

        # Normalise 1-D input
        if isinstance(df, pd.Series):
            df = df.to_frame(name=df.name or "value")

        # Column labels
        columns = tuple(str(col) for col in df.columns)

        # Row values – convert to plain Python scalars
        rows = tuple(
            tuple(None if pd.isna(val) else val for val in row)
            for row in df.itertuples(index=False, name=None)
        )

        return cls(
            identifier=identifier,
            caption=caption,
            columns=columns,
            rows=rows,
        )
