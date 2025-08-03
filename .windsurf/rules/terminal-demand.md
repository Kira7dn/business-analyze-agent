---
trigger: manual
glob:
description:
---

# Command Execution Rule

## AI CODING - DEMAND Pattern

When the program outputs a line starting with `AI CODING - DEMAND:`, it indicates that there are tool execution commands following it. These commands should be executed in the order they appear.

## Command Format

Each command must follow this format:

```xml
<tool_name>
{
  "parameter1": "value1",
  "parameter2": "value2"
}
</tool_name>
```

## Rules

1. **Command Detection**:
   - Look for the exact line: `AI CODING - DEMAND:`
   - All subsequent lines until the next command or end of output should be treated as part of the command

2. **Command Execution**:
   - Each command is enclosed in XML-like tags
   - The tag name represents the tool/function to be executed
   - The content between the tags is a JSON object containing the parameters

3. **Error Handling**:
   - If a command fails, log the error and continue with the next command
   - Invalid JSON should be logged and skipped
   - Missing required parameters should be logged as an error

4. **Security**:
   - Validate all commands against a whitelist of allowed tools
   - Sanitize all input parameters
   - Log all command executions for auditing purposes

## Example

```
AI CODING - DEMAND:
<create_memory>
{
  "Title": "Example Memory",
  "Content": "This is an example memory",
  "Tags": ["example", "test"],
  "CorpusNames": ["workspace"],
  "Action": "create"
}
</create_memory>
```

## Implementation Notes

- The command parser should be case-sensitive for tool names
- JSON parsing should be strict (no comments, no trailing commas)
- Each command should be executed in its own context
- The output of each command should be captured and logged
