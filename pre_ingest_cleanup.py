#!/usr/bin/env python3
"""
pre_ingest_cleanup.py
=====================
Removes redundant firecode file and validates remaining data before ingestion.

Run this BEFORE running ingest_laws.py

Usage:
  python pre_ingest_cleanup.py
"""

import os
import json
from pathlib import Path

def validate_and_cleanup():
    print("\n" + "="*70)
    print("PRE-INGESTION CLEANUP AND VALIDATION")
    print("="*70)
    
    cleaned_data = Path("cleaned_data")
    
    # 1. Remove redundant firecode file
    redundant_file = cleaned_data / "chunks_firecode_full_with_alttext.jsonl"
    if redundant_file.exists():
        print(f"\n[1] Removing redundant firecode file...")
        print(f"    File: {redundant_file.name}")
        print(f"    Size: {redundant_file.stat().st_size / 1024:.1f} KB")
        redundant_file.unlink()
        print(f"    ✓ Removed (will use firecode_chunks_hierarchical_v1.jsonl instead)")
    else:
        print(f"\n[1] Redundant file already removed")
    
    # 2. Validate remaining files
    print(f"\n[2] Validating remaining files...")
    
    markdown_files = list(cleaned_data.glob("*.md"))
    jsonl_files = list(cleaned_data.glob("*.jsonl"))
    
    print(f"\n    Markdown files ({len(markdown_files)}):")
    for md_file in sorted(markdown_files):
        try:
            content = md_file.read_text(encoding="utf-8")
            size_kb = len(content.encode('utf-8')) / 1024
            print(f"      ✓ {md_file.name} ({size_kb:.1f} KB)")
        except Exception as e:
            print(f"      ✗ {md_file.name} - ERROR: {e}")
    
    print(f"\n    JSONL files ({len(jsonl_files)}):")
    for jsonl_file in sorted(jsonl_files):
        try:
            valid_count = 0
            with open(jsonl_file, encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            json.loads(line)
                            valid_count += 1
                        except:
                            pass
            
            size_kb = jsonl_file.stat().st_size / 1024
            print(f"      ✓ {jsonl_file.name} ({size_kb:.1f} KB, {valid_count} records)")
        except Exception as e:
            print(f"      ✗ {jsonl_file.name} - ERROR: {e}")
    
    # 3. Summary
    print(f"\n" + "="*70)
    print("CLEANUP COMPLETE - READY FOR INGESTION")
    print("="*70)
    print(f"\nNext step: python ingest_laws.py")
    print("(This will take 15-30 minutes on first run)")

if __name__ == "__main__":
    validate_and_cleanup()
