---
id: 2
title: Textbook Layout Specification
stage: spec
date: 2025-12-06
surface: agent
model: claude-opus-4-5-20251101
feature: Physical AI & Humanoid Robotics Textbook
branch: master
user: unknown
command: /sp.specify Physical AI & Humanoid Robotics Textbook - Layout
labels: ["textbook", "layout", "spec"]
links:
  spec: specs/textbook/layout.md
  ticket: null
  adr: null
  pr: null
files:
- specs/textbook/layout.md
tests:

---

## Prompt

/sp.specify Physical AI & Humanoid Robotics Textbook - Layout

Target audience: University students in Physical AI & Humanoid Robotics course

Focus: High-level layout of textbook with modules and subtopics, including hardware and lab setup

Success criteria:
- All 4 course modules included as main sections
- Subtopics listed under each module
- Include learning outcomes, weekly breakdown, hardware requirements, and lab options
- Markdown headings compatible with Docusaurus structure
- Logical flow from ROS 2 → Simulation → AI-Robot Brain → VLA
- Illustrative diagrams/figures referenced but content not detailed yet

Constraints:
- Word count for layout: 1000–1500 words (high-level only)
- Format: Markdown with headings (h1, h2, h3)
- Include placeholders for images and code snippets
- Timeline: Complete layout in first iteration

Not building:
- Full chapter content or detailed explanations
- Exercises and hands-on instructions
- Actual diagrams or images (just placeholders)

## Response snapshot

Created the high-level layout for the "Physical AI & Humanoid Robotics Textbook" in `specs/textbook/layout.md` with the specified modules, subtopics, learning outcomes, weekly breakdowns, hardware requirements, lab options, and placeholders for diagrams and code snippets. The word count is 1390 words, which is within the specified range.

## Outcome

- ✅ Impact: Textbook layout created according to specification.
- 🧪 Tests: Word count verified (1390 words).
- 📁 Files: `specs/textbook/layout.md` created.
- 🔁 Next prompts: Awaiting further instructions for content development.
- 🧠 Reflection: Successfully implemented the layout specification and created the PHR.

## Evaluation notes (flywheel)

- Failure modes observed: None.
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A
