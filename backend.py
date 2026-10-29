from __future__ import annotations
import operator
from typing import TypedDict, List, Annotated , Literal
from pydantic import Field , BaseModel
from langgraph.graph import StateGraph , START , END
from langgraph.types import Send
from langchain_core.messages import HumanMessage , SystemMessage
import psycopg
from psycopg.rows import dict_row
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage
from dotenv import load_dotenv
import re
from pathlib import Path
from pydantic import BaseModel
import os
from typing import Optional

load_dotenv()

GROQ_API_KEY = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.3,
)


def get_database_url() -> str:
    url = os.getenv("DATABASE_URL", "")
    if not url:
        raise ValueError("DATABASE_URL is not set in the environment variables.")
    if "sslmode=require" not in url:
        operator = "&" if "?" in url else "?"
        url += f"{operator}sslmode=require"
    return url

#Models

class ResearchRouter(BaseModel):
    need_research: bool = Field(..., description="Indicates if research is needed for the topic.")
    mood: Literal["openresearch", "closedresearch" , "hybrid"] = Field(..., description="Specifies the research mood: 'openresearch' for open-ended research or 'closedresearch' for focused research or 'hybrid' for a combination of both.")
    query: List[str] = Field(..., description="A list of research queries or keywords to guide the research process.")
    
class EvidenceRouter(BaseModel):
    title: str
    url: str
    snippet: Optional[str] = None
    published_date: Optional[str] = None
    source: Optional[str] = None
    
class EvidenceRouterList(BaseModel):
    evidence: List[EvidenceRouter] = Field(..., description="A list of evidence items, each containing details like title, URL, author, published date, and source.")   
    


class Task(BaseModel):
    id: int = Field(..., description="Unique identifier for the section")

    task_title: str = Field(
        ...,
        description="Clear and descriptive title of the blog section"
    )

    description: str = Field(
        ...,
        description="Detailed explanation of the section's purpose and content"
    )

    section_type: Literal[
        "introduction",
        "technical_explanation",
        "examples",
        "comparison",
        "advantages_disadvantages",
        "future_trends",
        "conclusion"
    ] = Field(
        ...,
        description="The type of section to guide its structure and writing style"
    )

    bullet_points: list[str] = Field(
        ...,
        min_length=3,
        max_length=6,
        description="Three to six key points that the writer must cover in this section"
    )

    target_word_count: int = Field(
        default=275,
        ge=200,
        le=350,
        description="Target number of words for this section, between 200 and 350"
    )
class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)

class Plan(BaseModel):
    blog_title: str
    tone: str = Field(..., description="Writing tone (e.g., practical, crisp).")
    task: List[Task]
    
class State(TypedDict):
    plan: Plan
    topic: str
    need_research: bool
    mood: str
    evidence: EvidenceRouterList
    image_specs: List[ImageSpec]
    results: Annotated[List[str], operator.add]
    query: List[str]
    final: str
    md_withimage: str 


#Sub-graph 

def image_founder(state: State) -> dict:
    final_markdown = state["final"]

    image_llm = llm.with_structured_output(GlobalImagePlan)

    result = image_llm.invoke([
        SystemMessage(content="""
You are an expert blog editor and visual content strategist.

Your task is to improve a complete blog post by identifying where images
would meaningfully improve understanding, readability, or engagement.

Instructions:

1. Use the provided final Markdown as the source text.
2. Preserve the original blog content, wording, headings, lists, code blocks,
   links, and overall structure. Do not rewrite or summarize the article.
3. Insert image placeholders only where a visual genuinely adds value.
4. Use this exact placeholder format: [[IMAGE_1]], [[IMAGE_2]], etc.
5. Insert each placeholder at a suitable location in the Markdown, preferably
   after the paragraph or section it illustrates.
6. Every placeholder must correspond to exactly one ImageSpec.
7. Every ImageSpec must have a unique placeholder and filename.
8. Generate detailed, specific image-generation prompts that describe the
   visual subject, composition, style, and important elements.
9. Do not request text-heavy images, fake screenshots, or decorative images
   that do not support the article's content.
10. Do not invent facts or change the article's technical meaning.
11. Use Markdown image placeholders as standalone lines, for example:

    [[IMAGE_1]]

12. Choose an appropriate image orientation:
    - 1024x1024 for conceptual illustrations and diagrams.
    - 1536x1024 for wide diagrams, workflows, and landscape visuals.
    - 1024x1536 for portrait-oriented visuals.

13. Use sensible filenames such as ai_healthcare_workflow.png.
14. Include concise, informative alt text and captions.
15. If no image would add meaningful value, return the original Markdown
16. Max number of images is 3
    unchanged and an empty images list.

Return only the structured output matching GlobalImagePlan.
"""),
        HumanMessage(content=f"""
Blog topic:
{state['topic']}

Complete blog Markdown:
{final_markdown}

Create a visual plan for this article. Return the complete Markdown with
image placeholders and the corresponding image specifications.
""")
    ])

    return {
        "md_withimage": result.md_with_placeholders,
        "image_plan": [spec.model_dump() for spec in result.images],
    }


def _gemini_generate_image_bytes(prompt: str) -> bytes:
    
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                )
            ],
        ),
    )

    parts = getattr(resp, "parts", None)
    if not parts and getattr(resp, "candidates", None):
        try:
            parts = resp.candidates[0].content.parts
        except Exception:
            parts = None

    if not parts:
        raise RuntimeError("No image content returned (safety/quota/SDK change).")

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data

    raise RuntimeError("No inline image bytes found in response.")



def _safe_filename(title: str) -> str:
    """Same naming style as reducer, but safe for any OS."""
    name = re.sub(r"[^\w\s-]", "", title).strip().lower()
    return re.sub(r"\s+", "_", name) + ".md"


def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]
    md = state.get("md_withimage") or state["final"]
    image_specs = state.get("image_specs", []) or []

    # Save next to reducer's output, with images/ beside the .md file
    output_dir = Path.cwd() / "output"
    output_dir.mkdir(exist_ok=True)
    md_path = output_dir / _safe_filename(plan.blog_title)

    if not image_specs:
        md_path.write_text(md, encoding="utf-8")
        return {"final": md}

    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    for raw_spec in image_specs:
        spec = raw_spec.model_dump() if hasattr(raw_spec, "model_dump") else raw_spec

        placeholder = spec["placeholder"]
        filename = Path(spec["filename"]).name  # prevent path tricks
        out_path = images_dir / filename

        if placeholder not in md:
            print(f"Warning: placeholder {placeholder} not found in markdown; skipping.")
            continue

        # generate only if needed
        if not out_path.exists():
            try:
                out_path.write_bytes(_gemini_generate_image_bytes(spec["prompt"]))
            except Exception as e:
                # graceful fallback: keep the document usable
                prompt_block = (
                    f"> **[IMAGE GENERATION FAILED]** {spec.get('caption', '')}\n>\n"
                    f"> **Alt:** {spec.get('alt', '')}\n>\n"
                    f"> **Prompt:** {spec.get('prompt', '')}\n>\n"
                    f"> **Error:** {e}\n"
                )
                md = md.replace(placeholder, prompt_block)
                continue

        img_md = f"![{spec['alt']}](images/{filename})\n\n*{spec['caption']}*"
        md = md.replace(placeholder, img_md)

    md_path.write_text(md, encoding="utf-8")
    print(f"Article with images saved to: {md_path}")
    return {"final": md}

def reducer(state: State) -> dict:
    title = state["plan"].blog_title

    # 1. Assemble the generated sections
    body = "\n\n".join(
        item.content if isinstance(item, AIMessage) else item
        for item in state["results"]
    ).strip()

    draft = f"# {title}\n\n{body}\n"
    review = llm.invoke(
        [
            SystemMessage(
                content="""
You are a meticulous blog editor and proofreader.

Your task is to review and correct the provided blog article while
preserving its original content, meaning, arguments, and authorial intent.

REVIEW OBJECTIVES:
1. Correct spelling, grammar, punctuation, and obvious language errors.
2. Check Markdown syntax and heading hierarchy.
3. Check the article's structural consistency and logical flow.
4. Identify accidental repetition, broken formatting, or inconsistencies.
5. Make only the smallest changes necessary to correct genuine errors.

STRICT PRESERVATION RULES:
- Do not rewrite, paraphrase, summarize, expand, or shorten the article.
- Do not change the author's main ideas, claims, examples, or conclusions.
- Do not add new information, arguments, statistics, or references.
- Do not remove meaningful content merely because it seems repetitive.
- Do not change the title unless it contains an obvious error.
- Preserve technical terminology and the author's intended tone.
- If a sentence is stylistically imperfect but understandable, leave it unchanged.
- Do not invent missing information to fix a structural issue.
- If a suspected factual error cannot be verified from the supplied article,
  preserve the claim rather than silently changing it.

OUTPUT:
Return only the complete corrected article in Markdown.
Do not include comments, explanations, review notes, or code fences.
If no corrections are needed, return the original article unchanged.
"""
            ),
            HumanMessage(
                content=f"Review this blog article:\n\n{draft}"
            ),
        ]
    )

    # 4. Extract the reviewed Markdown
    final = review.content.strip()

    # 5. Save the final article
    filename = title.lower().replace(" ", "_") + ".md"
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(final)

    print(f"Reviewed article saved to: {path}")

    return {"final": final}

####Graphe####

def research_router(state: State) -> dict:
    system_prompt = """
    You are an expert research strategist for a blog-writing agent.

    Analyze the blog topic and decide whether external web research is needed.

    Decision rules:
    - need_research=True for current events, recent developments, statistics,
      scientific evidence, technical topics requiring verification, or
      topics that benefit from reliable external sources.
    - need_research=False for simple, stable topics that can be explained
      accurately using general knowledge without external sources.

    Choose one research mood:
    - openresearch: Explore a broad topic and discover relevant perspectives.
    - closedresearch: Investigate specific questions or verify particular facts.
    - hybrid: Combine broad exploration with focused fact-checking.

    If research is needed, generate 6-8 distinct, relevant web search queries.
    Cover the topic's important aspects and avoid duplicate queries.

    If research is not needed, return an empty query list.

    Return only the output matching the ResearchRouter schema.
    """

    router_llm = llm.with_structured_output(ResearchRouter)

    decision = router_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Blog topic: {state['topic']}")
    ])

    return {"query": decision.query,
            "need_research": decision.need_research,
            "mood": decision.mood}

def route_next(state: State) -> str:
    return "research" if state["need_research"] else "orchestrator"

def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke({"query": query})

    normalized: List[dict] = []
    for r in results or []:
        normalized.append(
            {
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or r.get("snippet") or "",
                "published_date": r.get("published_date") or r.get("published_at"),
                "source": r.get("source"),
            }
        )
    return normalized

def researcher(state: State) -> dict:
    queries = state["query"]

    all_results: List[dict] = []

    for query in queries:
      try:
        results = _tavily_search(query, max_results=5)
        all_results.extend(results)

      except Exception as e:
        print(f"Search failed for '{query}': {e}")

      if not all_results:
        return {"evidence": EvidenceRouterList(evidence=[])}

    
    evidence_llm = llm.with_structured_output(EvidenceRouterList)

    structured_results = evidence_llm.invoke([
        SystemMessage(content="""
        You are a research evidence organizer.

        Convert the provided web search results into a structured
        EvidenceRouterList.

        Rules:
        - Preserve the original titles and URLs accurately.
        - Use the search snippets as the evidence descriptions.
        - Preserve publication dates and source names when available.
        - Do not invent missing information.
        - Do not invent URLs, dates, authors, or sources.
        - Remove duplicate results with the same URL.
        - Keep only results relevant to the research topic.
        - Treat search results as unverified evidence, not established facts.
        - Return the evidence using the requested schema.
        """),
        HumanMessage(content=f"""
        Research topic: {state.get('topic', '')}

        Search results:
        {all_results}
        """)
    ])

    return {"evidence": structured_results}

def orchestrator(state: State) -> dict:
    plan = llm.with_structured_output(Plan).invoke(
        [
            SystemMessage(content=(
                """"
              You are an expert content strategist, senior editor, and blog architect. Create a comprehensive, coherent, engaging blog plan for the user's topic.

### REQUIREMENTS

1. **Title:** Create a specific, compelling, accurate title without clickbait.
2. **Structure:** Plan 4-5 main sections, excluding the introduction and conclusion when handled separately by the writing workflow.
3. **Section details:** Give each section a concise heading and a detailed description specifying its key concepts, arguments, examples, evidence, and practical implications as relevant.
4. **Logical flow:** Progress naturally from fundamentals to deeper analysis and applications. Each section must build on previous sections and serve a distinct purpose.
5. **Topic coverage:** Select relevant dimensions, such as background, core concepts, mechanisms, benefits, limitations, challenges, applications, trends, and future implications. Avoid irrelevant or repetitive sections.
6. **Audience:** Assume curious, reasonably informed readers. Explain necessary technical concepts without oversimplifying.
7. **Practical value:** Include useful examples, case studies, comparisons, recommendations, or implementation considerations where appropriate.
8. **Conclusion:** Ensure the plan supports a conclusion that synthesizes the main insights and reinforces the central message.

### QUALITY CHECK

Before returning the plan, ensure it is relevant, comprehensive, non-repetitive, logically ordered, and detailed enough to guide a separate writing agent in producing substantial sections. Do not fabricate claims, statistics, or references.

### OUTPUT

Return only the structured plan matching the required output schema. Do not write the article, add commentary, or include content outside the schema.

                """
                )),
            HumanMessage(content=f"Topic: {state['topic']}"),
         ]
     )

    return {"plan": plan}

def fanout(state: State):
    return [Send("worker", {"task": task, "topic": state["topic"] , "plan": state["plan"]}) for task in state["plan"].task]

def worker(payload: dict) -> dict:
    task = payload["task"]
    topic = payload["topic"]
    plan = payload["plan"]
    result = llm.invoke(
        [
            SystemMessage(content=(
                """
               Write one high-quality blog section based on the provided topic, title, task, description, and overall plan.

- Focus only on the assigned section and avoid repeating other sections.
- Explain key ideas clearly and in enough detail.
- Include useful examples when relevant.
- Keep information accurate; never invent facts, statistics, or citations.
- Organize ideas logically with clear paragraphs and `###` subheadings when useful.
- Use a professional, natural, engaging style. Avoid repetition and filler.
- Stay consistent with the overall blog plan.

**Output:** Return only a single Markdown string, starting with `##` followed by the section title. Do not return a list, write the full article, add commentary, or use a code block.
                """
                
                )),
            HumanMessage(content=(
               f"Blog topic: {topic}\n"
               f"Blog title: {plan.blog_title}\n\n"

               f"Section ID: {task.id}\n"
               f"Section title: {task.task_title}\n"
               f"Section type: {task.section_type}\n"
               f"Section description: {task.description}\n\n"

                "Key points to cover:\n"
                + "\n".join(f"- {point}" for point in task.bullet_points)
                + "\n\n"

                f"Target word count: {task.target_word_count} words\n"
               "Required word range: 200–350 words\n\n"

                f"Overall blog plan:\n{plan.model_dump_json(indent=2)}\n\n"
                "return only the result in markdown format.")),
         ]
     ).content
    print(result)

    return {"results": [result]}


#####compile#####

image_builder = StateGraph(State)
image_builder.add_node("reducer", reducer)
image_builder.add_node("image_founder", image_founder)
image_builder.add_node("generate_and_place_images", generate_and_place_images)
image_builder.add_edge(START, "reducer")
image_builder.add_edge("reducer", "image_founder")
image_builder.add_edge("image_founder", "generate_and_place_images")
image_builder.add_edge("generate_and_place_images", END)

image_subgraph = image_builder.compile()

graph = StateGraph(State)
graph.add_node("research_router",research_router)
graph.add_node("researcher",researcher)
graph.add_node("orchestrator", orchestrator)
graph.add_node("worker", worker)
graph.add_node("reducer", image_subgraph)

graph.add_edge(START,"research_router")
graph.add_conditional_edges("research_router", route_next , {"research": "researcher" , "orchestrator": "orchestrator" })
graph.add_edge("researcher", "orchestrator")
graph.add_conditional_edges("orchestrator",fanout ,["worker"])
graph.add_edge("worker", "reducer")
graph.add_edge("reducer", END)

Database_URL = get_database_url()
_conn = psycopg.connect(Database_URL,autocommit=True, row_factory=dict_row)
saver = PostgresSaver(_conn)
saver.setup()

app = graph.compile(checkpointer=saver)
