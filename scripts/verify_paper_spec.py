import sys
import os
import re
import hashlib
import fitz

def verify_paper(pdf_path, tex_path, log_path, bib_path):
    print(f"\n=== Verifying {pdf_path} ===")
    errors = []
    warnings = []

    # 1. Check file existence
    for p in [pdf_path, tex_path, log_path, bib_path]:
        if not os.path.exists(p):
            errors.append(f"File missing: {p}")
            return errors, warnings

    # 2. Check tex file for GDUT and authors
    with open(tex_path, 'r', encoding='utf-8', errors='ignore') as f:
        tex_content = f.read()
    if re.search(r'gdut', tex_content, re.IGNORECASE):
        errors.append(f"GDUT found in {tex_path}")
    else:
        print("[PASS] 0 GDUT occurrences in .tex")

    # Author verification
    for author in ["Nguyen Han Nhu", "Nguyen Van Thanh", "Tran Pham Tuan Dung", "Ha Anh Vu"]:
        if author not in tex_content:
            errors.append(f"Author {author} missing in {tex_path}")
    if "FPT University, Ho Chi Minh City, Vietnam" not in tex_content:
        errors.append(f"Affiliation missing in {tex_path}")
    for email in ["Nhunhse183644@fpt.edu.vn", "thanhnvSE180387@fpt.edu.vn", "dungtptse180382@fpt.edu.vn", "AnhVH54@fe.edu.vn"]:
        if email not in tex_content:
            errors.append(f"Email {email} missing in {tex_path}")
    if "Corresponding author: Nguyen Han Nhu (email: Nhunhse183644@fpt.edu.vn)" not in tex_content:
        errors.append(f"Corresponding author footnote missing in {tex_path}")
    print("[PASS] Author roster, emails, affiliation, and corresponding author verified")

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
    if len(bib_entries) != 25:
        errors.append(f"Bib entries count is {len(bib_entries)} (REQUIRED: 25)")
    else:
        print("[PASS] Exactly 25 bibliography entries")

    # 4. Check log file for overfull \hbox and errors
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        log_content = f.read()
    
    overfull_hboxes = re.findall(r'Overfull \\hbox \(([^\)]+)\)', log_content)
    if overfull_hboxes:
        errors.append(f"Overfull \\hbox warnings ({len(overfull_hboxes)}): {overfull_hboxes}")
    else:
        print("[PASS] 0 Overfull \\hbox warnings in log")

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
    hyphenated_words = []
    for i, page in enumerate(doc):
        lines = page.get_text('text').splitlines()
        for line_idx, line in enumerate(lines):
            line_str = line.strip()
            if re.search(r'[a-zA-Z]-$', line_str):
                next_line = lines[line_idx+1].strip() if line_idx+1 < len(lines) else ""
                hyphenated_words.append((i+1, line_str, next_line))
    
    if hyphenated_words:
        for p, l, n in hyphenated_words:
            m = re.search(r'([a-zA-Z]+)-$', l)
            if m and n and re.match(r'^[a-zA-Z]+', n):
                errors.append(f"Word-splitting hyphen on Page {p}: '{m.group(1)}-' followed by '{n.split()[0]}'")
    if not errors or not any("Word-splitting hyphen" in e for e in errors):
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
    citations_in_pdf = re.findall(r'\[(\d+)\]', full_pdf_text)
    citation_nums = sorted(list(set(int(c) for c in citations_in_pdf)))
    print(f"[INFO] Citations found in PDF: min={min(citation_nums) if citation_nums else None}, max={max(citation_nums) if citation_nums else None}, count={len(citation_nums)}")
    if citation_nums:
        if max(citation_nums) != 25:
            errors.append(f"Max citation number is {max(citation_nums)} (REQUIRED: 25)")
        else:
            print("[PASS] Exactly 25 citations numbered 1 to 25")

    return errors, warnings

def verify_all_pdf_deliverables():
    print("\n=== Verifying All 6 PDF Deliverables Synchronization ===")
    pdf_paths = [
        'AAIML 2027/conference-latex-template_10-17-19/Rep-YOLO11s_AAIML2027.pdf',
        'Rep-YOLO11s_AAIML2027_Submission_Final.pdf',
        'AAIML2027_Submission.pdf',
        'AAIML 2027/Rep-YOLO11s_AAIML2027_Submission.pdf',
        'AAIML 2027/Rep-YOLO11s_AAIML2027.pdf',
        'paper_overleaf/main.pdf'
    ]
    hashes = {}
    errors = []
    for p in pdf_paths:
        if not os.path.exists(p):
            errors.append(f"Deliverable missing: {p}")
            continue
        h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
        hashes[p] = h
        sz = os.path.getsize(p)
        print(f"[INFO] {p}: {sz} bytes (SHA256: {h[:16]}...)")

    unique_hashes = set(hashes.values())
    if len(unique_hashes) > 1:
        errors.append(f"PDF deliverables are NOT synchronized! Distinct hashes: {unique_hashes}")
    elif len(unique_hashes) == 1:
        print("[PASS] All 6 PDF deliverables are 100% bit-for-bit identical")

    return errors

if __name__ == '__main__':
    all_errors = []
    all_warnings = []

    # 1. Verify AAIML 2027 template
    err1, warn1 = verify_paper(
        'AAIML 2027/conference-latex-template_10-17-19/Rep-YOLO11s_AAIML2027.pdf',
        'AAIML 2027/conference-latex-template_10-17-19/Rep-YOLO11s_AAIML2027.tex',
        'AAIML 2027/conference-latex-template_10-17-19/Rep-YOLO11s_AAIML2027.log',
        'AAIML 2027/conference-latex-template_10-17-19/references.bib'
    )
    all_errors.extend(err1)
    all_warnings.extend(warn1)

    # 2. Verify Overleaf directory
    err2, warn2 = verify_paper(
        'paper_overleaf/main.pdf',
        'paper_overleaf/main.tex',
        'paper_overleaf/main.log',
        'paper_overleaf/references.bib'
    )
    all_errors.extend(err2)
    all_warnings.extend(warn2)

    # 3. Verify all 6 PDFs synchronization
    err3 = verify_all_pdf_deliverables()
    all_errors.extend(err3)

    print("\n=== FINAL SPEC VERIFICATION SUMMARY ===")
    print(f"Total Errors: {len(all_errors)}")
    for e in all_errors:
        print(f"  [ERROR] {e}")
    print(f"Total Warnings: {len(all_warnings)}")
    for w in all_warnings:
        print(f"  [WARN] {w}")

    if all_errors:
        sys.exit(1)
    print("\n[SUCCESS] ALL SPECIFICATIONS RIGOROUSLY VERIFIED!")
    sys.exit(0)
