# Seed Dataset Schema Metadata

This document describes the schema of each JSONL seed file in this directory. The schema is inferred from the current seed records and reflects the fields present in each example.

## Common schema across all seed files

Every seed record contains the following fields:

- `id`: string — unique identifier for the example
- `category`: string — dataset category name
- `source`: string — usually `seed`
- `difficulty`: string — task difficulty, typically `easy`, `medium`, or `hard`
- `prompt`: string — the main instruction or question
- `response`: string — the reference answer, explanation, or expected output for the example

## Per-dataset schema

### classification.jsonl

- `id`: string
- `category`: string, typically `classification`
- `source`: string
- `difficulty`: string
- `domain`: string — topic or label domain such as `sentiment` or `spam_detection`
- `prompt`: string
- `response`: string

### code_debugging.jsonl

- `id`: string
- `category`: string, typically `code_debugging`
- `source`: string
- `language`: string — programming language, e.g. `python`
- `difficulty`: string
- `bug_type`: string — type of bug, e.g. `wrong_operator` or `off_by_one`
- `prompt`: string
- `response`: string

### code_generation.jsonl

- `id`: string
- `category`: string, typically `code_generation`
- `source`: string
- `language`: string
- `difficulty`: string
- `reasoning_type`: string — reasoning pattern such as `iteration` or `string_processing`
- `prompt`: string
- `response`: string

### data_transformation.jsonl

- `id`: string
- `category`: string, typically `data_transformation`
- `source`: string
- `difficulty`: string
- `domain`: string — transformation type such as `json_conversion` or `table_conversion`
- `prompt`: string
- `response`: string

### information_extraction.jsonl

- `id`: string
- `category`: string, typically `information_extraction`
- `source`: string
- `difficulty`: string
- `domain`: string — extraction topic such as `contact_info` or `dates`
- `prompt`: string
- `response`: string

### instruction_following.jsonl

- `id`: string
- `category`: string, typically `instruction_following`
- `source`: string
- `difficulty`: string
- `domain`: string — style or constraint domain such as `exact_output` or `format_constraint`
- `prompt`: string
- `response`: string

### logical_reasoning.jsonl

- `id`: string
- `category`: string, typically `logical_reasoning`
- `source`: string
- `difficulty`: string
- `domain`: string — reasoning type such as `syllogism` or `conditional_reasoning`
- `prompt`: string
- `response`: string

### math_reasoning.jsonl

- `id`: string
- `category`: string, typically `math_reasoning`
- `source`: string
- `difficulty`: string
- `domain`: string — math topic such as `rate` or `percentage`
- `prompt`: string
- `response`: string

### planning.jsonl

- `id`: string
- `category`: string, typically `planning`
- `source`: string
- `difficulty`: string
- `domain`: string — task domain such as `daily_task` or `cooking`
- `prompt`: string
- `response`: string

### question_answering.jsonl

- `id`: string
- `category`: string, typically `question_answering`
- `source`: string
- `difficulty`: string
- `domain`: string — topic such as `geography` or `science`
- `prompt`: string
- `response`: string

## Notes

- All files are newline-delimited JSON (JSONL).
- The `prompt` field is the core content for each sample.
- Category-specific fields such as `domain`, `language`, `bug_type`, and `reasoning_type` describe the task subtype.
