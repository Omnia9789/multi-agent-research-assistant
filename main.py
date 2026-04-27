"""
Multi-Agent Research Assistant
Entry point for the LangGraph + LlamaIndex orchestrated system.
"""

import asyncio
import argparse
from core.graph import build_graph
from core.state import ResearchState
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


def print_banner():
    console.print(Panel.fit(
        "[bold cyan]🔬 Multi-Agent Research Assistant[/bold cyan]\n"
        "[dim]LangGraph · LlamaIndex · RAG · Fact-Check[/dim]",
        border_style="cyan"
    ))


async def run_query(query: str, docs_dir: str = "data/sample_docs", verbose: bool = False):
    print_banner()
    console.print(f"\n[bold yellow]Query:[/bold yellow] {query}\n")

    graph = build_graph(docs_dir=docs_dir)

    initial_state: ResearchState = {
        "query": query,
        "retrieved_chunks": [],
        "reasoning_output": "",
        "fact_check_result": {},
        "final_answer": "",
        "confidence_score": 0.0,
        "flagged": False,
        "agent_logs": [],
        "error": None,
    }

    with console.status("[bold green]Running multi-agent pipeline...[/bold green]"):
        result = await graph.ainvoke(initial_state)

    console.print("\n" + "─" * 60)
    console.print("[bold cyan]📋 Research Report[/bold cyan]\n")
    console.print(Markdown(result["final_answer"]))

    console.print("\n" + "─" * 60)
    fc = result.get("fact_check_result", {})
    confidence = result.get("confidence_score", 0.0)
    flagged = result.get("flagged", False)

    status_color = "red" if flagged else "green"
    status_icon = "⚠️" if flagged else "✅"
    console.print(
        f"[bold]{status_icon} Fact-Check Status:[/bold] "
        f"[{status_color}]{'FLAGGED' if flagged else 'PASSED'}[/{status_color}]  |  "
        f"[bold]Confidence:[/bold] [cyan]{confidence:.0%}[/cyan]"
    )

    if fc.get("issues"):
        console.print("[yellow]Issues detected:[/yellow]")
        for issue in fc["issues"]:
            console.print(f"  • {issue}")

    if verbose:
        console.print("\n[dim]Agent Logs:[/dim]")
        for log in result.get("agent_logs", []):
            console.print(f"  [dim]{log}[/dim]")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Research Assistant (LangGraph + LlamaIndex)"
    )
    parser.add_argument("query", nargs="?", help="Research query")
    parser.add_argument("--docs", default="data/sample_docs", help="Path to documents directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show agent logs")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    if args.interactive or not args.query:
        print_banner()
        console.print("[dim]Type 'exit' to quit.[/dim]\n")
        while True:
            try:
                query = console.input("[bold yellow]Query:[/bold yellow] ").strip()
                if query.lower() in ("exit", "quit", "q"):
                    break
                if not query:
                    continue
                asyncio.run(run_query(query, args.docs, args.verbose))
                console.print()
            except KeyboardInterrupt:
                break
    else:
        asyncio.run(run_query(args.query, args.docs, args.verbose))


if __name__ == "__main__":
    main()
