import argparse
import pandas as pd

def df_to_latex_subtables(df, index_col, cols_per_table=5, caption="Results Table", label="tab:results"):
    """
    Convert a DataFrame into a LaTeX table with subtables, splitting columns.
    Each subtable includes the index column (first column).
    The maximum value in each column is bolded.
    """
    # Make a copy so we don’t overwrite original
    df = df.copy()

    # Bold max per column
    def make_bold_max(col):
        if pd.api.types.is_numeric_dtype(col):
            max_val = col.max()
            return lambda x: f"\\textbf{{{x:.4f}}}" if x == max_val else f"{x:.4f}"
        else:
            return None  # escape=True will handle text
    
    formatters = {col: make_bold_max(df[col]) for col in df.columns}

    # Start building LaTeX code
    latex_code = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
    ]

    # Split columns into chunks
    if index_col is None:
        index_col = df.index.name or "Index"
    all_columns = df.columns.tolist()
    all_columns.remove(index_col)
    col_chunks = [[index_col] + all_columns[i:i+cols_per_table] for i in range(1, len(all_columns), cols_per_table)]
    num_subtables = len(col_chunks)
    subtable_width = (0.95 / num_subtables)  # leave a little margin

    # Add each subtable
    for i, chunk in enumerate(col_chunks, 1):
        sub_df = df[chunk]
        # sub_df = 
        subtable = sub_df.to_latex(
            escape=True,
            index=False,
            longtable=False,
            multicolumn=True,
            multicolumn_format="c",
            formatters = formatters
        )

        latex_code.append(f"\\begin{{subtable}}")
        latex_code.append("\\centering")
        latex_code.append(subtable)
        # latex_code.append(f"\\caption{{Part {i}}}")
        latex_code.append("\\end{subtable}")
        if i < len(col_chunks):
            latex_code.append("\\hfill")  # spacing between subtables

    latex_code.append("\\end{table}")

    return "\n".join(latex_code)

def main():
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Convert wide CSV into LaTeX table with subtables")
    parser.add_argument("input_csv", help="Input CSV file")
    parser.add_argument("output_tex", help="Output LaTeX file")
    parser.add_argument("--columns", nargs="+", help="Columns to include (default all)")
    parser.add_argument("--index_col", default="run_id", help="Column to repeat as index")
    parser.add_argument("--cols_per_subtable", type=int, default=5, help="Max non-index columns per subtable")
    parser.add_argument("--caption", default="Results Table", help="Caption for the table")
    parser.add_argument("--label", default="tab:results", help="Label for the table")
    args = parser.parse_args()

    # Read CSV
    df = pd.read_csv(args.input_csv)

    # Filter columns if requested
    if args.columns:
        df = df[[c for c in args.columns if c in df.columns]]

    # Reset index if needed to make index_col explicit
    # if args.index_col not in df.columns:
    #     df = df.reset_index().rename(columns={"index": args.index_col})

    # Generate LaTeX code with subtables
    latex_code = df_to_latex_subtables(
        df,
        args.index_col,
        cols_per_table=args.cols_per_subtable,
        caption=args.caption,
        label=args.label
    )

    # Write to output .tex
    with open(args.output_tex, "w") as f:
        f.write(latex_code)

    print(f"LaTeX table with subtables written to {args.output_tex}")

if __name__ == "__main__":
    main()