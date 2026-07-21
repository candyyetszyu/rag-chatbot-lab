# knowledge_base/

This is the student-authored corpus layer — there's no bundled sample, so add at least one document here before building the index. Each `<doc_id>/meta.yaml` you add becomes part of the active corpus.

## Layout

```
knowledge_base/
├── my_doc/
│   ├── meta.yaml         ← required
│   └── ocr_output.json   ← your text, keyed by page (default filename)
└── another_doc/
    ├── meta.yaml
    └── ...
```

## `meta.yaml`

```yaml
id: my_doc                       # unique slug
title: "Document title"
author: "Author Name"
year: "2024"
genre: "category"                # free-form
source: "course_pack"            # free-form
folder: "my_doc"                 # on-disk folder name, must match this directory
ocr_file: "ocr_output.json"      # default if omitted
```

## `ocr_output.json`

A flat JSON object mapping page keys to page text:

```json
{
  "page_1": "First page of your document text goes here...",
  "page_2": "Second page continues here..."
}
```

One page is fine to start — split longer documents into as many `page_N` keys as you like. To use a different filename, set `ocr_file` in `meta.yaml`; the content still needs to be a page-keyed JSON object.

## After adding or changing documents

Rebuild the index:

```bash
python tests/quick_test.py --rebuild
```

See [`STUDENT_GUIDE.md`](../STUDENT_GUIDE.md) step 3 for a full walkthrough.
