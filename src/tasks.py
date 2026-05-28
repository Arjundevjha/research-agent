"""
Task definitions — the 4-step research pipeline.

Plan → Research → Write → Edit
Each task feeds its output into the next via context chaining.
"""

from crewai import Task

from src.agents import create_planner, create_researcher, create_writer, create_editor
from src.prompts import REPORT_STRUCTURE


def create_tasks() -> tuple:
    """
    Create the 4 sequential tasks and their agents.

    Returns:
        (agents_list, tasks_list) — both needed to build the Crew.
    """
    planner = create_planner()
    researcher = create_researcher()
    writer = create_writer()
    editor = create_editor()

    # Task 1: Plan the research
    plan_task = Task(
        description=(
            "The user wants to research: {topic}\n\n"
            "Break this topic into 3-5 focused sub-questions that together "
            "give a complete picture. Think about:\n"
            "- What are the key facts someone needs to know?\n"
            "- What recent developments matter?\n"
            "- Are there controversies or open debates?\n"
            "- What practical implications exist?\n\n"
            "Output a numbered list of sub-questions, each with a brief note "
            "on why it matters."
        ),
        expected_output=(
            "A numbered list of 3-5 research sub-questions, each with a "
            "one-line explanation of its importance."
        ),
        agent=planner,
    )

    # Task 2: Research each sub-question
    research_task = Task(
        description=(
            "Take the sub-questions from the planner and research each one.\n\n"
            "For each sub-question:\n"
            "1. Run 2-3 different web searches with varied queries\n"
            "2. Scrape the most relevant pages to get details\n"
            "3. Extract concrete facts, numbers, and findings\n"
            "4. Record full source metadata for each finding:\n"
            "   Author | Date | Title | Site Name | URL\n\n"
            "Aim for 2-4 quality findings per sub-question from different "
            "sources. Total: at least 8 sources across all sub-questions.\n\n"
            "Do NOT make up information. If you cannot find something, say so."
        ),
        expected_output=(
            "Structured findings organized by sub-question. Each finding "
            "includes the fact/data and complete source metadata "
            "(author, date, title, site, URL)."
        ),
        agent=researcher,
        context=[plan_task],
    )

    # Task 3: Write the report
    write_task = Task(
        description=(
            "Using the research findings, write a comprehensive report on: "
            "{topic}\n\n"
            "Requirements:\n"
            "- Follow the report structure provided in your instructions\n"
            "- Use APA 7th Edition in-text citations for every factual claim\n"
            "- Build citations from the source metadata the researcher provided\n"
            "- Write in clear, direct language — no jargon or filler\n"
            "- Include a complete References section at the end\n"
            "- Aim for 1,500-3,000 words\n\n"
            f"{REPORT_STRUCTURE}"
        ),
        expected_output=(
            "A complete research report in Markdown format with:\n"
            "- Title and date\n"
            "- Introduction\n"
            "- 3-5 body sections (one per sub-question)\n"
            "- Conclusion\n"
            "- Full APA References section"
        ),
        agent=writer,
        context=[plan_task, research_task],
    )

    # Task 4: Edit and polish
    edit_task = Task(
        description=(
            "Review and polish the draft report. Check for:\n\n"
            "1. JARGON: Remove any filler phrases, buzzwords, or vague "
            "language. Replace with clear, direct statements.\n"
            "2. CITATIONS: Every factual claim must have an (Author, Year) "
            "citation. Every citation must appear in References.\n"
            "3. APA FORMAT: Verify in-text citations and References follow "
            "APA 7th Edition exactly.\n"
            "4. FLOW: Ensure logical progression between sections.\n"
            "5. SUBSTANCE: Every paragraph must contain at least one specific "
            "fact or finding.\n\n"
            "Output the COMPLETE final report — this is what the user receives."
        ),
        expected_output=(
            "The final, polished research report in Markdown format. "
            "Clean, jargon-free, properly cited in APA 7th Edition, "
            "ready to deliver."
        ),
        agent=editor,
        context=[write_task],
    )

    agents = [planner, researcher, writer, editor]
    tasks = [plan_task, research_task, write_task, edit_task]

    return agents, tasks
