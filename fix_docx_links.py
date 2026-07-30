#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_docx_links.py - 将 Pandoc 生成的 DOCX 中的超链接转换为 Word 标准交叉引用

问题：Pandoc 使用 w:hyperlink 元素生成文献引用链接，在 Word 中点击这些链接时
      会打开一个新文件窗口再跳转，而不是在文档内直接跳转。

修复：将 w:hyperlink（内部锚点链接）转换为 Word 标准的 REF 字段交叉引用，
      等效于 Word 中"插入 → 交叉引用"操作。

用法：
  python3 fix_docx_links.py thesis_output.docx
  python3 fix_docx_links.py input.docx -o output.docx
"""

import sys
import os
import copy
import zipfile
import tempfile
import xml.etree.ElementTree as ET

NS_W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS_R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
NS_XML = 'http://www.w3.org/XML/1998/namespace'

# Register namespaces to preserve them in output
ET.register_namespace('w', NS_W)
ET.register_namespace('r', NS_R)
ET.register_namespace('xml', NS_XML)
# Common OOXML namespaces
ET.register_namespace('wpc', 'http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas')
ET.register_namespace('mc', 'http://schemas.openxmlformats.org/markup-compatibility/2006')
ET.register_namespace('o', 'urn:schemas-microsoft-com:office:office')
ET.register_namespace('m', 'http://schemas.openxmlformats.org/officeDocument/2006/math')
ET.register_namespace('v', 'urn:schemas-microsoft-com:vml')
ET.register_namespace('wp', 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing')
ET.register_namespace('w10', 'urn:schemas-microsoft-com:office:word')
ET.register_namespace('w14', 'http://schemas.microsoft.com/office/word/2010/wordml')
ET.register_namespace('w15', 'http://schemas.microsoft.com/office/word/2012/wordml')
ET.register_namespace('wpg', 'http://schemas.microsoft.com/office/word/2010/wordprocessingGroup')
ET.register_namespace('wpi', 'http://schemas.microsoft.com/office/word/2010/wordprocessingInk')
ET.register_namespace('wne', 'http://schemas.microsoft.com/office/word/2006/wordml')
ET.register_namespace('wps', 'http://schemas.microsoft.com/office/word/2010/wordprocessingShape')
ET.register_namespace('a', 'http://schemas.openxmlformats.org/drawingml/2006/main')
ET.register_namespace('rel', 'http://schemas.openxmlformats.org/package/2006/relationships')


def _make_element(tag, attrib=None, text=None):
    """Create an OOXML element with optional attributes and text."""
    elem = ET.Element(tag, attrib or {})
    if text is not None:
        elem.text = text
    return elem


def _clone_rpr(source_run):
    """Clone w:rPr from a source w:r element, removing hyperlink-specific styles."""
    rPr = source_run.find(f'{{{NS_W}}}rPr')
    if rPr is None:
        return None

    new_rPr = copy.deepcopy(rPr)

    # Remove Hyperlink character style reference
    rStyle = new_rPr.find(f'{{{NS_W}}}rStyle')
    if rStyle is not None and rStyle.get(f'{{{NS_W}}}val') == 'Hyperlink':
        new_rPr.remove(rStyle)

    return new_rPr


def convert_hyperlink_to_ref_field(parent, hyperlink):
    """Convert a w:hyperlink element to a REF field code sequence.

    Replaces:
      <w:hyperlink w:anchor="ref-XXX">
        <w:r><w:rPr>...</w:rPr><w:t>[1]</w:t></w:r>
      </w:hyperlink>

    With:
      <w:r><w:rPr>...</w:rPr><w:fldChar w:fldCharType="begin"/></w:r>
      <w:r><w:rPr>...</w:rPr><w:instrText> REF ref-XXX \\h </w:instrText></w:r>
      <w:r><w:rPr>...</w:rPr><w:fldChar w:fldCharType="separate"/></w:r>
      <w:r><w:rPr>...</w:rPr><w:t>[1]</w:t></w:r>
      <w:r><w:rPr>...</w:rPr><w:fldChar w:fldCharType="end"/></w:r>
    """
    anchor = hyperlink.get(f'{{{NS_W}}}anchor')
    if not anchor:
        return False

    # Collect all runs and their text from the hyperlink
    runs = hyperlink.findall(f'{{{NS_W}}}r')
    if not runs:
        return False

    # Get formatting from first run
    rPr_template = _clone_rpr(runs[0])

    # Collect display text from all runs
    display_parts = []
    for run in runs:
        for t in run.findall(f'{{{NS_W}}}t'):
            display_parts.append(t.text or '')
    display_text = ''.join(display_parts)

    # Find position of hyperlink in parent
    idx = list(parent).index(hyperlink)

    # Remove the hyperlink element
    parent.remove(hyperlink)

    def make_run_with_rpr():
        r = ET.Element(f'{{{NS_W}}}r')
        if rPr_template is not None:
            r.append(copy.deepcopy(rPr_template))
        return r

    # 1. fldChar begin
    run_begin = make_run_with_rpr()
    fld_begin = ET.SubElement(run_begin, f'{{{NS_W}}}fldChar')
    fld_begin.set(f'{{{NS_W}}}fldCharType', 'begin')
    parent.insert(idx, run_begin)
    idx += 1

    # 2. instrText
    run_instr = make_run_with_rpr()
    instr = ET.SubElement(run_instr, f'{{{NS_W}}}instrText')
    instr.set(f'{{{NS_XML}}}space', 'preserve')
    instr.text = f' REF {anchor} \\h '
    parent.insert(idx, run_instr)
    idx += 1

    # 3. fldChar separate
    run_sep = make_run_with_rpr()
    fld_sep = ET.SubElement(run_sep, f'{{{NS_W}}}fldChar')
    fld_sep.set(f'{{{NS_W}}}fldCharType', 'separate')
    parent.insert(idx, run_sep)
    idx += 1

    # 4. Display text (preserve original runs for formatting)
    for run in runs:
        parent.insert(idx, copy.deepcopy(run))
        idx += 1

    # 5. fldChar end
    run_end = make_run_with_rpr()
    fld_end = ET.SubElement(run_end, f'{{{NS_W}}}fldChar')
    fld_end.set(f'{{{NS_W}}}fldCharType', 'end')
    parent.insert(idx, run_end)

    return True


def fix_document_xml(xml_content):
    """Process document.xml: convert internal hyperlinks to REF fields."""
    root = ET.fromstring(xml_content)

    converted = 0
    skipped = 0

    # Find all paragraphs that contain hyperlinks
    # We need parent elements to manipulate children
    for parent in root.iter():
        hyperlinks = [
            child for child in list(parent)
            if child.tag == f'{{{NS_W}}}hyperlink'
            and child.get(f'{{{NS_W}}}anchor') is not None
        ]

        for hyperlink in hyperlinks:
            if convert_hyperlink_to_ref_field(parent, hyperlink):
                converted += 1
            else:
                skipped += 1

    return ET.tostring(root, encoding='unicode', xml_declaration=True), converted, skipped


def fix_docx(input_path, output_path=None):
    """Fix hyperlinks in a DOCX file by converting them to REF field codes."""
    if output_path is None:
        output_path = input_path

    # Create temp directory for processing
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Extract DOCX
        with zipfile.ZipFile(input_path, 'r') as zin:
            zin.extractall(tmp_dir)

        # Process document.xml
        doc_xml_path = os.path.join(tmp_dir, 'word', 'document.xml')
        if not os.path.exists(doc_xml_path):
            print(f"错误：{input_path} 中未找到 word/document.xml")
            return False

        with open(doc_xml_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()

        fixed_xml, converted, skipped = fix_document_xml(xml_content)

        if converted == 0:
            print(f"提示：未找到需要转换的内部超链接")
            return True

        # Write fixed document.xml
        with open(doc_xml_path, 'w', encoding='utf-8') as f:
            f.write(fixed_xml)

        # Repackage DOCX
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zout:
            for root_dir, dirs, files in os.walk(tmp_dir):
                for file in files:
                    file_path = os.path.join(root_dir, file)
                    arcname = os.path.relpath(file_path, tmp_dir)
                    zout.write(file_path, arcname)

        print(f"✓ 已转换 {converted} 个引用链接为 Word 标准交叉引用")
        if skipped:
            print(f"  跳过 {skipped} 个无法处理的链接")

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='将 Pandoc 生成的 DOCX 中的超链接转换为 Word 标准交叉引用')
    parser.add_argument('input', help='输入 DOCX 文件路径')
    parser.add_argument('-o', '--output', default=None,
                        help='输出 DOCX 文件路径（默认：覆盖输入文件）')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"错误：文件不存在：{args.input}")
        sys.exit(1)

    output = args.output or args.input
    success = fix_docx(args.input, output)
    if not success:
        sys.exit(1)

    print(f"输出文件：{output}")


if __name__ == '__main__':
    main()
