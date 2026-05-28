"""
Research Agent — CLI entry point.

Takes a topic from the user, runs the 4-agent research crew,
and saves the final report to outputs/.
"""

import os
import re
import sys
from datetime import datetime

from crewai import Crew, Process
from dotenv import load_dotenv

from src.tasks import create_tasks


def slugify(text: str) -> str:
    """Turn a topic string into a safe filename slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "_", text)
    text = re.sub(r"-+", "-", text)
    return text[:60]


def save_report(content: str, topic: str) -> str:
    """Save the report to outputs/ and return the file path."""
    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = slugify(topic)
    filename = f"outputs/{slug}_{timestamp}.md"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(str(content))

    return filename


def main():
    load_dotenv()

    # Get topic from CLI args or interactive input
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
    else:
        print("\n" + "=" * 60)
        print("  RESEARCH AGENT — Multi-Agent Deep Research")
        print("=" * 60)
        topic = input("\nWhat topic do you want to research?\n> ").strip()

    if not topic:
        print("No topic provided. Exiting.")
        sys.exit(1)

    print(f"\n📋 Topic: {topic}")
    print("🚀 Starting research crew...\n")

    # Build the crew
    agents, tasks = create_tasks()

    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
        max_rpm=10,  # Throttle to avoid free-tier rate limits
    )

    # Run it
    result = crew.kickoff(inputs={"topic": topic})

    # Save the report
    filepath = save_report(result.raw, topic)

    print("\n" + "=" * 60)
    print(f"✅ Report saved to: {filepath}")
    print("=" * 60)


if __name__ == "__main__":
    main()
