import json
import re

with open('kode_praktikum.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Membagi file berdasarkan header CELL
blocks = re.split(r'(# ╔═+╗\n(?:# ║.*?\n)+# ╚═+╝\n)', content)

cells = []

# Bagian pertama (Header Praktikum dll) -> Markdown
if blocks[0].strip():
    md_source = []
    lines = blocks[0].split('\n')
    for j, line in enumerate(lines):
        # Hapus '# ' di awal baris
        clean_line = re.sub(r'^#\s?', '', line)
        md_source.append(clean_line + ('\n' if j < len(lines)-1 else ''))
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": md_source
    })

# Memproses setiap CELL header dan kode di bawahnya
for i in range(1, len(blocks), 2):
    cell_header = blocks[i]
    cell_code = blocks[i+1] if i+1 < len(blocks) else ""
    
    # CELL header -> Markdown
    md_source = []
    lines = cell_header.split('\n')
    for line in lines:
        if line.strip():
            clean_line = re.sub(r'^#\s?', '', line).strip()
            if '╔' in clean_line or '╚' in clean_line:
                continue
            clean_line = clean_line.replace('║', '').strip()
            if clean_line:
                md_source.append('## ' + clean_line + '\n')
            
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": md_source
    })
    
    # Kode di bawah CELL -> Code Cell
    if cell_code.strip():
        code_source = []
        lines = cell_code.split('\n')
        # Hilangkan newline kosong di awal/akhir blok jika ada
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop(-1)
            
        for j, line in enumerate(lines):
            code_source.append(line + ('\n' if j < len(lines)-1 else ''))
            
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code_source
        })

notebook = {
 "cells": cells,
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open('kode_praktikum.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Konversi berhasil! File kode_praktikum.ipynb telah dibuat.")
