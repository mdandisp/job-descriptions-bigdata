import re

with open('kode_praktikum.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace savefig
content = re.sub(r"savefig\('([^\/]+\.png)'", r"savefig('output/\1'", content)

# Inject os.makedirs
content = re.sub(r'(import os, re, time\n)', r'\1\nos.makedirs("output", exist_ok=True)\n', content)

with open('kode_praktikum.py', 'w', encoding='utf-8') as f:
    f.write(content)
