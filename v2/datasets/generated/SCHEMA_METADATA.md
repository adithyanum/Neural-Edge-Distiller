# Generated Dataset Schema Metadata

This document describes the schema of each generated JSONL file in this directory. The schema is inferred from the currently generated records and reflects the fields present in the output examples.

## Common schema across all generated files

Each generated record typically contains:

- `id`: string — unique identifier for the generated example
- `category`: string — dataset category name
- `source`: string — usually `generated`
- `difficulty`: string — difficulty level such as `easy` or `medium`
- `parent_id`: string — ID of the source seed example that this record was derived from
- `variation_role`: string — how the example was varied, such as `difficulty` or `context`
- `prompt`: string — the generated task prompt
- `response`: string — the model-produced answer or completion

### Optional field

- `context`: string — extra contextual detail used by some generated variants

## Per-dataset schema

### classification.jsonl

- `id`: string
- `category`: string, typically `classification`
- `source`: string
- `difficulty`: string
- `parent_id`: string
- `variation_role`: string
- `context`: string, optional
- `prompt`: string
- `response`: string

### code_debugging.jsonl

- `id`: string
- `category`: string, typically `code_debugging`
- `source`: string
- `difficulty`: string
- `parent_id`: string
- `variation_role`: string
- `prompt`: string
- `response`: string

### code_generation.jsonl

- `id`: string
- `category`: string, typically `code_generation`
- `source`: string
- `difficulty`: string
- `parent_id`: string
- `variation_role`: string
- `prompt`: string
- `response`: string

### data_transformation.jsonl

- `id`: string
- `category`: string, typically `data_transformation`
- `source`: string
- `difficulty`: string
- `parent_id`: string
- `variation_role`: string
- `prompt`: string
- `response`: string

### information_extraction.jsonl

- `id`: string
- `category`: string, typically `information_extraction`
- `source`: string
- `difficulty`: string
- `parent_id`: string
- `variation_role`: string
- `prompt`: string
- `response`: string

### instruction_following.jsonl

- `id`: string
- `category`: string, typically `instruction_following`
- `source`: string
- `difficulty`: string
- `parent_id`: string
- `variation_role`: string
- `prompt`: string
- `response`: string

### logical_reasoning.jsonl

- `id`: string
- `category`: string, typically `logical_reasoning`
- `source`: string
- `difficulty`: string
- `parent_id`: string
- `variation_role`: string
- `prompt`: string
- `response`: string

### math_reasoning.jsonl

- `id`: string
- `category`: string, typically `math_reasoning`
- `source`: string
- `difficulty`: string
- `parent_id`: string
- `variation_role`: string
- `prompt`: string
- `response`: string

### planning.jsonl

- `id`: string
- `category`: string, typically `planning`
- `source`: string
- `difficulty`: string
- `parent_id`: string
- `variation_role`: string
- `prompt`: string
- `response`: string

### question_answering.jsonl

- `id`: string
- `category`: string, typically `question_answering`
- `source`: string
- `difficulty`: string
- `parent_id`: string
- `variation_role`: string
- `prompt`: string
- `response`: string

## Notes

- These files are newline-delimited JSON (JSONL).
- The generated records are derived from seed examples via augmentation or variation.
- `parent_id` is useful for tracing each generated example back to its seed source.
- `response` is freeform text rather than a strict structured field.
