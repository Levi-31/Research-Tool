import re
import asyncio
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from blog_generator.config import llm
from blog_generator.prompts import DECIDE_IMAGES_SYSTEM
from blog_generator.models import State, GlobalImagePlan
from blog_generator.services.factory import ServiceFactory

def merge_content(state: State) -> dict:
    """
    Combines written sections into a single markdown body.
    """
    plan = state["plan"]
    if plan is None:
        raise ValueError("merge_content called without plan.")
    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}

def decide_images(state: State) -> dict:
    """
    Decides where to place images and drafts their generation prompts.
    """
    planner = llm.with_structured_output(GlobalImagePlan)
    merged_md = state["merged_md"]
    plan = state["plan"]
    assert plan is not None

    image_plan = planner.invoke(
        [
            SystemMessage(content=DECIDE_IMAGES_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Topic: {state['topic']}\n\n"
                    "Insert placeholders + propose image prompts.\n\n"
                    f"{merged_md}"
                )
            ),
        ]
    )

    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images],
    }

def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"

def generate_and_place_images(state: State) -> dict:
    """
    Generates images via the image service and places them in the final markdown document.
    Processes all image generation tasks concurrently using asyncio.run.
    """
    image_service = ServiceFactory.get_image_generator()
    
    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []

    # If no images requested, just write merged markdown
    if not image_specs:
        filename = f"{_safe_slug(plan.blog_title)}.md"
        Path(filename).write_text(md, encoding="utf-8")
        return {"final": md}

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    async def run_async_gen() -> list[tuple[str, str]]:
        # Helper function to generate a single image spec asynchronously
        async def process_spec(spec: dict) -> tuple[str, str]:
            placeholder = spec["placeholder"]
            filename = spec["filename"]
            out_path = images_dir / filename

            if not out_path.exists():
                try:
                    # API call is async
                    img_bytes = await image_service.generate_image_bytes(spec["prompt"])
                    # Non-blocking file writing
                    await asyncio.to_thread(out_path.write_bytes, img_bytes)
                except Exception as e:
                    # Graceful fallback: keep doc usable
                    prompt_block = (
                        f"> **[IMAGE GENERATION FAILED]** {spec.get('caption','')}\n>\n"
                        f"> **Alt:** {spec.get('alt','')}\n>\n"
                        f"> **Prompt:** {spec.get('prompt','')}\n>\n"
                        f"> **Error:** {e}\n"
                    )
                    return placeholder, prompt_block

            img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
            return placeholder, img_md

        # Run all image generations concurrently
        return await asyncio.gather(*(process_spec(spec) for spec in image_specs))

    results = asyncio.run(run_async_gen())

    # Apply replacements
    for placeholder, replacement in results:
        md = md.replace(placeholder, replacement)

    # Write final file
    filename = f"{_safe_slug(plan.blog_title)}.md"
    Path(filename).write_text(md, encoding="utf-8")
    
    return {"final": md}

# Build reducer subgraph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()
