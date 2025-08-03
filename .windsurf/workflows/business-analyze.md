---
description: Analyze Idea then return Document with detail Features and Requirement
---

1. Load the input idea/requirements file
2. Call `evaluate_requirements` tool with the initial requirements
3. If questions exist Present questions to the user one by one and update idea/requirements file
4. Call `evaluate_requirements` tool again with new requirements (if have update for questions)
5. Update the original file with clear format as markdown document
6. Call `suggest_features` tool with updated requirements
7. Update return infomation from `suggest_features` to original file
8. Generate product requirement document PRD.md from updated idea/requirements in the same folder