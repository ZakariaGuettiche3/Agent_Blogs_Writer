# AI Blog Writer Agent

A LangGraph pipeline that takes a topic and produces a researched, proofread Markdown blog post with AI-generated images.

## Features

- **Research routing**: decides whether web research is needed and generates search queries (Tavily)
- **Planning**: builds a structured outline (title, tone, sections)
- **Parallel writing**: one worker per section, run in parallel with `Send`
- **Editing**: merges the sections and proofreads them in a single pass
- **Image generation**: places up to 3 images (Gemini) and saves the final `.md` file
- **Persistence**: graph state is checkpointed in PostgreSQL

## Workflow

```
START → research_router ─┬─(need research)─→ researcher ─┐
                         └─(no research)─────────────────┴→ orchestrator
orchestrator → worker (×N, parallel) → [ image_subgraph ] → END

image_subgraph:  reducer → image_founder → generate_and_place_images
```

| Node                        | Purpose                                                                                                    |
| --------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `research_router`           | Decides if research is needed, picks a mode (`openresearch` / `closedresearch` / `hybrid`) and 6–8 queries |
| `researcher`                | Runs Tavily searches and structures the results as `EvidenceRouterList`                                    |
| `orchestrator`              | Creates the `Plan` (4–5 sections)                                                                          |
| `fanout` / `worker`         | Writes each section (200–350 words)                                                                        |
| `reducer`                   | Joins the sections, proofreads them, saves `output/<title>.md`                                             |
| `image_founder`             | Inserts `[[IMAGE_n]]` placeholders and creates image specs                                                 |
| `generate_and_place_images` | Generates images and replaces the placeholders with Markdown image tags                                    |

## Setup

**Requirements:** Python 3.10+, a PostgreSQL database, and Google AI and Tavily API keys.

```bash
pip install langgraph langgraph-checkpoint-postgres langchain-google-genai \
            langchain-community google-genai "psycopg[binary]" \
            python-dotenv pydantic tavily-python
```

Create a `.env` file:

```env
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
DATABASE_URL=postgresql://user:password@host:5432/dbname
```

`sslmode=require` is appended to `DATABASE_URL` automatically.

## Usage

```python
config = {"configurable": {"thread_id": "my-blog-1"}}

out = app.invoke(
    {"topic": "The Future of Artificial Intelligence in Everyday Life", "results": []},
    config=config,
)
print(out["final"])
```

## Output

```
output/
├── <blog_title>.md      # final article
└── images/              # generated images referenced by the article
```

## Models

| Task                                       | Model                    |
| ------------------------------------------ | ------------------------ |
| Text (routing, planning, writing, editing) | `gemini-3.5-flash`       |
| Image generation                           | `gemini-2.5-flash-image` |

## Known Issues

- `image_founder` returns the key `image_plan`, but the state and `generate_and_place_images` expect `image_specs`. Rename the returned key.
- In `researcher`, the `if not all_results: return ...` check is indented inside the `for` loop. Move it out of the loop.
- `evidence` from the researcher is not yet passed to the section workers.
- `ImageSpec.size` and `quality` are defined but not used when generating images.
