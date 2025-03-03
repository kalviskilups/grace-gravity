#!/usr/bin/env python3
import os
import re

# Set the folder path (here, the current directory)
folder = "time_series/ddk3_2004_weekly"
pattern = re.compile(
    r"^\s*Reference:\s*\r?\n"  # Match 'Reference:' with optional whitespace and newline
    r"(?:.*\r?\n)+?"  # Match one or more lines, non-greedily
    r"(?=^\s*(?:time_period_of_data|begin_of_head))",  # Lookahead for the next header
    re.MULTILINE,
)

for filename in os.listdir(folder):
    if filename.endswith(".gfc"):
        filepath = os.path.join(folder, filename)
        enc = "utf-8"
        try:
            with open(filepath, "r", encoding=enc) as f:
                content = f.read()
        except UnicodeDecodeError:
            enc = "latin-1"
            with open(filepath, "r", encoding=enc) as f:
                content = f.read()
        new_content, num_subs = re.subn(pattern, "", content)
        print(f"{filename}: {num_subs} substitution(s) made using encoding {enc}.")
        with open(filepath, "w", encoding=enc) as f:
            f.write(new_content)
