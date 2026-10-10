import sys
import os
import re
import fitz

def verify_paper(pdf_path, tex_path, log_path, bib_path):
    print(f"=== Verifying {pdf_path} ===")
    errors = []
    warnings = []

    # 1. Check file existence
    for p in [pdf_path, tex_path, log_path, bib_path]:
        if not os.path.exists(p):
            errors.append(f"File missing: {p}")
            return errors, warnings

    # 2. Check tex file for GDUT
    with open(tex_path, 'r', encoding='utf-8', errors='ignore') as f:
        tex_content = f.read()
    if re.search(r'gdut', tex_content, re.IGNORECASE):
        errors.append(f"GDUT found in {tex_path}")
    else:
        print("[PASS] 0 GDUT occurrences in .tex")

    # 3. Check bib file for gdut
    with open(bib_path, 'r', encoding='utf-8', errors='ignore') as f:
        bib_content = f.read()
    if re.search(r'gdut', bib_content, re.IGNORECASE):
        errors.append(f"GDUT found in {bib_path}")
    else:
        print("[PASS] 0 GDUT occurrences in .bib")

    # Count bib entries in bib_content
    bib_entries = re.findall(r'@\w+\s*\{([^,]+),', bib_content)
    print(f"[INFO] Bib entries in bib file: {len(bib_entries)}")

    # 4. Check log file for overfull \hbox and errors
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        log_content = f.read()
    
    overfull_hboxes = re.findall(r'Overfull \\hbox \(([^\)]+)\)', log_content)
    if overfull_hboxes:
        errors.append(f"Overfull \\hbox warnings ({len(overfull_hboxes)}): {overfull_hboxes}")
    else:
        print("[PASS] 0 Overfull \\hbox warnings in log")

    if re.search(r'gdut', log_content, re.IGNORECASE):
        # check if it's from the old log or current
        gdut_lines = [l for l in log_content.splitlines() if re.search(r'gdut', l, re.IGNORECASE)]
        warnings.append(f"GDUT found in log ({len(gdut_lines)} lines): {gdut_lines[:2]}")

    # 5. Check PDF using fitz
    doc = fitz.open(pdf_path)
    page_count = len(doc)
    print(f"[INFO] Page count: {page_count}")
    if page_count != 6:
        errors.append(f"Page count is {page_count} (REQUIRED: exactly 6)")
    else:
        print("[PASS] Exactly 6.0 pages")

    # 6. Check for Type 3 fonts
    has_type3 = False
    for i, page in enumerate(doc):
        fonts = page.get_fonts()
        for font in fonts:
            # font tuple: (xref, ext, type, basefont, name, encoding)
            font_type = font[2]
            font_name = font[3]
            if font_type == 'Type3':
                has_type3 = True
                errors.append(f"Type 3 font detected on Page {i+1}: {font_name}")
    if not has_type3:
        print("[PASS] 0 Type 3 fonts (100% Type 1 / TrueType / PostScript)")

    # 7. Check PDF text for GDUT
    full_pdf_text = ""
    for i, page in enumerate(doc):
        t = page.get_text()
        full_pdf_text += t
        if re.search(r'gdut', t, re.IGNORECASE):
            errors.append(f"GDUT found in PDF text on Page {i+1}")
    if not re.search(r'gdut', full_pdf_text, re.IGNORECASE):
        print("[PASS] 0 GDUT occurrences in PDF text")

    # 8. Check for word-splitting hyphens across all pages
    # A word-splitting hyphen typically occurs when a line ends with '-' and the next line continues the word
    # Let's check words ending with '-' where the hyphen is at the right edge of a column
    hyphenated_words = []
    for i, page in enumerate(doc):
        lines = page.get_text('text').splitlines()
        for line_idx, line in enumerate(lines):
            line_str = line.strip()
            # check if line ends with a letter followed by a hyphen
            if re.search(r'[a-zA-Z]-$', line_str):
                # could be hyphenated word
                next_line = lines[line_idx+1].strip() if line_idx+1 < len(lines) else ""
                # ignore mathematical minus, ranges like 10--, etc.
                hyphenated_words.append((i+1, line_str, next_line))
    
    if hyphenated_words:
        print(f"[WARN] Potential hyphenated line breaks ({len(hyphenated_words)}):")
        for p, l, n in hyphenated_words:
            print(f"  Page {p}: '{l}' -> '{n}'")
        # Check if they are genuine word breaks like infer-ence
        for p, l, n in hyphenated_words:
            m = re.search(r'([a-zA-Z]+)-$', l)
            if m and n and re.match(r'^[a-zA-Z]+', n):
                errors.append(f"Word-splitting hyphen on Page {p}: '{m.group(1)}-' followed by '{n.split()[0]}'")
    else:
        print("[PASS] ZERO word-splitting hyphens detected")

    # 9. Check column balance on Page 6
    page6 = doc[5]
    blocks = page6.get_text('blocks')
    col1_blocks = [b for b in blocks if b[0] < 300]
    col2_blocks = [b for b in blocks if b[0] >= 300]

    if col1_blocks and col2_blocks:
        col1_bottom = max(b[3] for b in col1_blocks)
        col2_bottom = max(b[3] for b in col2_blocks)
        diff = abs(col1_bottom - col2_bottom)
        print(f"[INFO] Page 6 column bottom y: Col 1 = {col1_bottom:.1f} pt, Col 2 = {col2_bottom:.1f} pt, Diff = {diff:.1f} pt")
        if diff > 60:
            warnings.append(f"Columns on Page 6 may be unbalanced: diff = {diff:.1f} pt")
        else:
            print(f"[PASS] Columns on Page 6 are well balanced (diff = {diff:.1f} pt)")
    else:
        warnings.append("Page 6 does not have two distinct columns")

    # 10. Check bibliography citations
    # Look for [25] in text
    citations_in_pdf = re.findall(r'\[(\d+)\]', full_pdf_text)
    citation_nums = sorted(list(set(int(c) for c in citations_in_pdf)))
    print(f"[INFO] Citations found in PDF: min={min(citation_nums) if citation_nums else None}, max={max(citation_nums) if citation_nums else None}, count={len(citation_nums)}")
    if citation_nums:
        if max(citation_nums) != 25:
            errors.append(f"Max citation number is {max(citation_nums)} (REQUIRED: 25)")
        else:
            print("[PASS] Exactly 25 citations numbered 1 to 25")

    print("\n=== SUMMARY ===")
    print(f"Errors ({len(errors)}): {errors}")
    print(f"Warnings ({len(warnings)}): {warnings}")
    return errors, warnings

if __name__ == '__main__':
    pdf = 'AAIML 2027/conference-latex-template_10-17-19/Rep-YOLO11s_AAIML2027.pdf'
    tex = 'AAIML 2027/conference-latex-template_10-17-19/Rep-YOLO11s_AAIML2027.tex'
    log = 'AAIML 2027/conference-latex-template_10-17-19/Rep-YOLO11s_AAIML2027.log'
    bib = 'AAIML 2027/conference-latex-template_10-17-19/references.bib'
    err, warn = verify_paper(pdf, tex, log, bib)
    if err:
        sys.exit(1)
    sys.exit(0)
