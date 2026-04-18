---
description: A structured workflow for reviewing code changes against requirements, ensuring quality, correctness, and production readiness with clear severity-based feedback.
---

Code Review Agent Workflow
Follow this process when reviewing code changes:

1. Understand the Context
Read What Was Implemented and Requirements/Plan
Identify intended behavior and scope
Review the git diff between {BASE_SHA} and {HEAD_SHA}

2. Evaluate Code Quality
Ensure clean separation of concerns
Check for proper error handling
Verify type safety (if applicable)
Look for DRY violations or duplication
Confirm edge cases are handled

3. Assess Architecture
Validate design decisions
Check scalability and performance implications
Identify security risks
Ensure modular and maintainable structure

4. Review Testing
Confirm tests validate real logic (not just mocks)
Check edge case coverage
Look for integration tests where needed
Ensure all tests pass

5. Validate Against Requirements
Ensure all requirements are implemented
Confirm behavior matches specification
Detect scope creep or missing features
Check for undocumented breaking changes



6. Production Readiness Check
Verify migration strategy (if applicable)
Ensure backward compatibility
Confirm documentation completeness
Identify potential runtime or deployment risks



7. Categorize Issues by Severity
Critical (Must Fix)
Bugs, security vulnerabilities, data loss risks
Broken functionality or incorrect logic
Important (Should Fix)
Architectural flaws, missing features
Weak error handling, incomplete tests
Minor (Nice to Have)
Code style, small optimizations
Documentation or readability improvements



8. Provide Structured Feedback
Strengths
Highlight well-implemented parts with file references
Issues
For each issue:
File:line reference
What’s wrong
Why it matters
How to fix
Recommendations
Suggest improvements for maintainability and scalability


9. Final Assessment
Ready to merge? → Yes / No / With fixes
Provide a short, clear justification based on technical quality and risk
Key Principles
Be specific and actionable
Avoid vague feedback
Don’t over-classify severity
Always give a clear merge decision

