"""
Utility to display table
========================

**April 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""


def table_format(headers, rows):
    headers = tuple(headers)
    rows = tuple(rows)

    # Get number of columns and total rows
    ncols = len(headers)
    nrows = len(rows)

    # Format all elements as string
    formatted_rows = [None] * nrows

    nlong_str = [len(h) for h in headers]
    for i, row in enumerate(rows):
        if len(row) != ncols:
            raise ValueError(f"Invalid number of columns for row: {row}")

        formatted_row = [None] * len(row)

        for j, col in enumerate(row):
            s_col = str(col)

            nlong_str[j] = max(nlong_str[j], len(s_col))

            formatted_row[j] = s_col
        formatted_rows[i] = tuple(formatted_row)

    # Compute separator
    l_sep = "+"
    for i in range(ncols):
        l_sep += "-" * (nlong_str[i] + 2) + "+"

    # Display header
    output = l_sep + "\n" + "|"

    for i, h in enumerate(headers):
        output += " " + str(h) + " " * (nlong_str[i] - len(h)) + " |"

    output += "\n" + l_sep

    # Display rows
    for row in formatted_rows:
        output += "\n|"

        for i, col in enumerate(row):
            output += " " + col + " " * (nlong_str[i] - len(col)) + " |"

    output += "\n" + l_sep

    return output


if __name__ == "__main__":
    # Test case 1: Simple table
    headers = ["Name", "Age"]
    rows = [("Alice", 30), ("Bob", 25), ("Charlie", 35)]
    print("Test case 1:")
    print(table_format(headers, rows))

    # Test case 2: Table with varying column widths
    headers = ["Item", "Description", "Price"]
    rows = [
        ("Apple", "A red fruit", 1.99),
        ("Banana", "A yellow fruit", 0.99),
        ("Orange", "A citrus fruit", 1.50),
    ]
    print("\nTest case 2:")
    print(table_format(headers, rows))

    # Test case 3: Empty rows
    headers = ["Column1", "Column2"]
    rows = []
    print("\nTest case 3 (empty rows):")
    print(table_format(headers, rows))

    # Test case 4: Row with more columns than headers
    headers = ["A", "B"]
    rows = [("X", "Y", "Z")]
    try:
        print("\nTest case 4 (row with extra columns):")
        print(table_format(headers, rows))
    except ValueError as e:
        print(f"Error: {e}")
