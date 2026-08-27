# Judged Dataset Schema Metadata

This document describes the schema of each JSONL file in this directory. The schema is inferred from the current judged records and reflects the fields present in the judge output examples.

## Common schema across all judged files

Each judged record typically contains:

- `id`: string — unique identifier for the example
- `category`: string — dataset category name
- `source`: string — usually `generated`
- `difficulty`: string — task difficulty, typically `easy`, `medium`, or `hard`
- `parent_id`: string — ID of the source seed example that this record was derived from
- `variation_role`: string — how the example was varied, such as `difficulty` or `context`
- `context`: string, optional — extra context used in some variants
- `prompt`: string — the task prompt
- `response`: string — the generated answer or completion
- `judge_verdict`: object — judge metadata for the record

### `judge_verdict` object

- `verdict`: string — usually `pass` or `fail`
- `reason`: string — short explanation for the judge decision

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
- `judge_verdict`: object

### code_debugging.jsonl

- `id`: string
- `category`: string, typically `code_debugging`
- `source`: string
- `difficulty`: string
- `parent_id`: string
- `variation_role`: string
- `prompt`: string
- `response`: string
- `judge_verdict`: object

### code_generation.jsonl

- `id`: string
- `category`: string, typically `code_generation`
- `source`: string
- `difficulty`: string
- `parent_id`: string
- `variation_role`: string
- `prompt`: string
- `response`: string
- `judge_verdict`: object

### data_transformation.jsonl

- `id`: string
- `category`: string, typically `data_transformation`
- `source`: string
- `difficulty`: string
- `parent_id`: string
- `variation_role`: string
- `prompt`: string
- `response`: string
- `judge_verdict`: object

### information_extraction.jsonl

- `id`: string
- `category`: string, typically `information_extraction`
- `source`: string
- `difficulty`: string
- `parent_id`: string
- `variation_role`: string
- `prompt`: string
- `response`: string
- `judge_verdict`: object

### instruction_following.jsonl

- `id`: string
- `category`: string, typically `instruction_following`
- `source`: string
- `difficulty`: string
- `parent_id`: string
- `variation_role`: string
- `prompt`: string
- `response`: string
- `judge_verdict`: object

### logical_reasoning.jsonl

- `id`: string
- `category`: string, typically `logical_reasoning`
- `source`: string
- `difficulty`: string
- `parent_id`: string
- `variation_role`: string
- `prompt`: string
- `response`: string
- `judge_verdict`: object

### math_reasoning.jsonl

- `id`: string
- `category`: string, typically `math_reasoning`
- `source`: string
- `difficulty`: string
- `parent_id`: string
- `variation_role`: string
- `prompt`: string
- `response`: string
- `judge_verdict`: object

### planning.jsonl

- `id`: string
- `category`: string, typically `planning`
- `source`: string
- `difficulty`: string
- `parent_id`: string
- `variation_role`: string
- `prompt`: string
- `response`: string
- `judge_verdict`: object

### question_answering.jsonl

- `id`: string
- `category`: string, typically `question_answering`
- `source`: string
- `difficulty`: string
- `parent_id`: string
- `variation_role`: string
- `prompt`: string
- `response`: string
- `judge_verdict`: object

## Notes

- These files are newline-delimited JSON (JSONL).
- Judged records are generated candidates that have been evaluated by the judge.
- `judge_verdict.verdict` is used to filter records when assembling the final dataset.
- Records with `verdict: pass` are considered accepted by the judge.