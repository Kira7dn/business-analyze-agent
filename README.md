# Business Analyze Agent - MCP Server

AI Agent hỗ trợ phân tích yêu cầu dự án web thông qua MCP (Model Context Protocol).

## Quick Start

### Connect with Cascade
Add this configuration to your Cascade MCP settings:
```json
{
  "mcpServers": {
    "business-analyze-agent": {
      "command": "c:\\Workspace\\business-analyze-agent\\venv\\Scripts\\python.exe",
      "args": ["c:\\Workspace\\business-analyze-agent\\main.py"]
    }
  }
}
```

## Available Tools
- `analyze_requirements` - Analyze project requirements using MoSCoW method
- `generate_questions` - Generate context-aware clarification questions
- `health_check` - Check server status

## Project Structure

```
business-analyze-agent/
├── main.py               # Main entry point
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── .env                 # Environment variables
├── config/              # Configuration files
│   └── server_config.json
├── src/                 # Source code
│   ├── __init__.py
│   ├── server.py        # MCP server implementation
│   ├── models/          # Data models
│   │   ├── __init__.py
│   │   └── schemas.py
│   ├── tools/           # Business analysis tools
│   │   ├── __init__.py
│   │   ├── analyzer.py
│   │   └── question_generator.py
│   └── utils/           # Utility functions
│       ├── __init__.py
│       ├── config.py
│       └── logger.py
├── tests/               # Test files
│   ├── test_mcp.py
│   ├── test_analyzer.py
│   └── verify_mcp.py
└── tools/               # Legacy tools (to be migrated)
    ├── cascade_api.py
    ├── memory_manager.py
    └── workflow_manager.py
```


# Business Analysis Workflow

### 1. Initialize Analysis
- Load the input idea/requirements file
- Verify the content is valid and not empty

### 2. Initial Evaluation
- Call `evaluate_requirements` tool with the initial requirements
- The evaluator will:
  - Score the requirements against evaluation criteria (0-5)
  - Identify missing or unclear information
  - Generate specific questions for clarification

### 3. Handle Evaluation Results
- Display the evaluation scores and overall assessment
- If questions exist:
  - Present questions to the user one by one
  - Collect user responses
  - Update the requirements with the new information
  - Return to step 2 with updated requirements
- If no questions:
  - Proceed to detailed requirements analysis


### 4. Detailed Requirements Analysis
- Call `analyze_requirements` to perform MoSCoW prioritization
- Generate a structured requirements document with:
  - Must Have requirements
  - Should Have requirements
  - Could Have requirements
  - Won't Have requirements (this time)

### 5. Generate Clarification Questions
- For any remaining ambiguities, use `generate_questions`
- Focus on specific areas needing more detail
- Collect and incorporate user responses

### 6. Finalize Documentation
- Compile all gathered information into a comprehensive document
- Include:
  - Project overview
  - Detailed requirements (prioritized)
  - Evaluation scores and rationale
  - Assumptions and constraints
  - Open questions (if any)

## Output
- Structured requirements document in Markdown format
- Evaluation report with scores and feedback
- List of any remaining questions or uncertainties

## Error Handling
- If evaluation fails, provide clear error messages
- Allow manual override if automatic evaluation is not possible
- Log all operations for debugging and auditing

## Best Practices
- Keep requirements concise and testable
- Use clear, unambiguous language
- Document all assumptions
- Validate requirements with stakeholders
- Update documentation as requirements evolve