from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import requests
import json
import os
import uuid
import re
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory stores
report_store = {}   # industry_key -> list of reports
chat_store = {}     # session_id -> {industry, report, messages}

# ── Models ──────────────────────────────────────────────────
class ResearchRequest(BaseModel):
    industry: str
    depth: Optional[str] = "standard"

class ChatMessage(BaseModel):
    session_id: str
    message: str

class SearchRequest(BaseModel):
    session_id: str
    query: str

# ── Tool: Web Search ─────────────────────────────────────────
def search_web(query: str, num_results: int = 5) -> List[dict]:
    current_year = datetime.now().year
    # Append year range to ensure recency
    enhanced_query = f"{query} {current_year} OR {current_year - 1} OR {current_year - 2}"

    response = requests.post(
        "https://google.serper.dev/search",
        headers={
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json"
        },
        json={
            "q": enhanced_query,
            "num": num_results,
            "gl": "us",
            "hl": "en",
            "tbs": "qdr:y2"  # Last 2 years
        },
        timeout=15
    )

    if response.status_code != 200:
        return []

    data = response.json()
    results = []
    for item in data.get("organic", []):
        results.append({
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "link": item.get("link", ""),
            "date": item.get("date", "")
        })
    return results

# ── Tool: Fetch Article ──────────────────────────────────────
def fetch_article(url: str) -> str:
    try:
        response = requests.get(
            url, timeout=10,
            headers={"User-Agent": "Mozilla/5.0 (compatible; MarketPulseBot/1.0)"}
        )
        if response.status_code != 200:
            return ""

        text = response.text
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:2000]
    except Exception:
        return ""

# ── Agent: Plan Queries ──────────────────────────────────────
def plan_search_queries(industry: str, depth: str) -> List[str]:
    current_year = datetime.now().year
    num_queries = 4 if depth == "standard" else 6 if depth == "deep" else 2

    prompt = f"""You are a senior business intelligence research planner.
Today is {datetime.now().strftime("%B %Y")}.

Generate exactly {num_queries} search queries to research the "{industry}" industry.

Requirements:
- Cover recent developments from {current_year - 2} to {current_year}
- Each query targets a DIFFERENT angle: news, market size, key players, trends, challenges, outlook
- Be specific — include company names, regions, metrics where relevant
- Do NOT repeat angles

Return ONLY a valid JSON array of strings. No explanation, no markdown.
Example: ["query one", "query two", "query three", "query four"]"""

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 400,
            "temperature": 0.3
        },
        timeout=30
    )

    content = response.json()["choices"][0]["message"]["content"].strip()
    match = re.search(r'\[.*?\]', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    return [
        f"{industry} market news {current_year}",
        f"{industry} industry trends {current_year}",
        f"{industry} key companies analysis {current_year}",
        f"{industry} challenges opportunities outlook {current_year}"
    ]

# ── Agent: Synthesise Report ─────────────────────────────────
def synthesise_report(industry: str, research_data: List[dict]) -> str:
    current_year = datetime.now().year

    research_text = ""
    for i, item in enumerate(research_data, 1):
        research_text += f"\n[Source {i}] {item['title']}\n"
        research_text += f"URL: {item['link']}\n"
        if item.get('date'):
            research_text += f"Date: {item['date']}\n"
        research_text += f"Summary: {item['snippet']}\n"
        if item.get('full_content'):
            research_text += f"Content: {item['full_content'][:600]}\n"
        research_text += "---\n"

    prompt = f"""You are a senior business intelligence analyst. Today is {datetime.now().strftime("%B %Y")}.

Write a comprehensive, up-to-date intelligence report on the **{industry}** industry.
Prioritise information from {current_year} and {current_year - 1}. Include older context only where it adds value.

Research data:
{research_text}

Write the report with these EXACT sections using markdown headers:

## Executive Summary
3-4 sentences. Most critical finding, current market state, and key number.

## Market Overview
Size, growth rate, geography. Use specific figures from the research.

## Key Trends ({current_year})
5 major trends with bullet points. Each trend must be specific and evidence-based.

## Key Players & Competitive Landscape
Named companies, their positioning, recent moves, market share where available.

## Opportunities
3 specific, actionable opportunities with reasoning.

## Risks & Challenges
3 significant risks with context and evidence.

## Outlook
Short-term (next 12 months) and medium-term (2-3 years) with specific predictions.

Rules:
- Use specific numbers, company names, dates
- Never fabricate data not in the research
- Write for a senior business executive audience
- Be direct and analytical, not vague"""

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2500,
            "temperature": 0.3
        },
        timeout=60
    )

    return response.json()["choices"][0]["message"]["content"]

# ── Main Research Endpoint ───────────────────────────────────
@app.post("/research")
def run_research_agent(request: ResearchRequest):
    industry = request.industry.strip()
    if not industry:
        raise HTTPException(status_code=400, detail="Industry cannot be empty.")

    print(f"[Agent] Planning research for: {industry}")
    queries = plan_search_queries(industry, request.depth)
    print(f"[Agent] Queries: {queries}")

    all_results = []
    seen_urls = set()

    for query in queries:
        print(f"[Agent] Searching: {query}")
        results = search_web(query, num_results=4)
        for r in results:
            if r["link"] not in seen_urls:
                seen_urls.add(r["link"])
                all_results.append(r)

    print(f"[Agent] Fetching content from top articles")
    for i in range(min(5, len(all_results))):
        all_results[i]["full_content"] = fetch_article(all_results[i]["link"])

    print(f"[Agent] Synthesising report from {len(all_results)} sources")
    report = synthesise_report(industry, all_results)

    report_id = str(uuid.uuid4())[:8]
    session_id = str(uuid.uuid4())
    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    sources = [r["link"] for r in all_results[:10]]

    report_entry = {
        "report_id": report_id,
        "session_id": session_id,
        "industry": industry,
        "report": report,
        "sources": sources,
        "generated_at": generated_at,
        "search_queries_used": queries
    }

    # Save to history
    industry_key = industry.lower().strip()
    if industry_key not in report_store:
        report_store[industry_key] = []
    report_store[industry_key].insert(0, report_entry)
    report_store[industry_key] = report_store[industry_key][:10]

    # Create chat session with report as context
    chat_store[session_id] = {
        "industry": industry,
        "report": report,
        "sources": sources,
        "messages": []
    }

    print(f"[Agent] Done: {report_id}")
    return report_entry

# ── Chat Endpoint ────────────────────────────────────────────
@app.post("/chat")
def chat_with_report(request: ChatMessage):
    """
    Allows the user to ask follow-up questions after a report is generated.
    The agent intelligently decides whether to:
    - Answer from the existing report
    - Search the web for more specific information
    - Compare with other industries
    """
    if request.session_id not in chat_store:
        raise HTTPException(status_code=404, detail="Session not found. Please generate a report first.")

    session = chat_store[request.session_id]
    industry = session["industry"]
    report = session["report"]
    history = session["messages"]

    # First, let the agent decide if it needs to search for more info
    decision_prompt = f"""You are an AI research assistant. A user is asking a follow-up question about a research report on "{industry}".

User question: "{request.message}"

Decide if you need to search the web for more information, or if you can answer from the existing report.

Reply with ONLY one of these exact responses:
- "ANSWER_FROM_REPORT" if the question can be answered from the report
- "SEARCH: <search query>" if you need to search for more specific information

Examples:
- "What are the key players?" -> ANSWER_FROM_REPORT
- "What is Moniepoint's latest valuation?" -> SEARCH: Moniepoint valuation 2025
- "Compare this to the Kenyan market" -> SEARCH: Kenyan {industry} market 2025"""

    decision_response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": decision_prompt}],
            "max_tokens": 60,
            "temperature": 0.1
        },
        timeout=15
    )

    decision = decision_response.json()["choices"][0]["message"]["content"].strip()
    print(f"[Agent] Decision: {decision}")

    additional_context = ""
    searched = False

    # If agent decides to search, do it
    if decision.startswith("SEARCH:"):
        search_query = decision.replace("SEARCH:", "").strip()
        print(f"[Agent] Searching for: {search_query}")
        results = search_web(search_query, num_results=4)
        if results:
            additional_context = "\n\nAdditional research found:\n"
            for r in results[:3]:
                additional_context += f"- {r['title']}: {r['snippet']}\n"
            searched = True

    # Build conversation history for context
    messages = [
        {
            "role": "system",
            "content": f"""You are MarketPulse AI, an expert business intelligence assistant.
You have just generated a research report on the "{industry}" industry.

Here is the full report for context:
{report}
{additional_context}

Guidelines:
- Answer questions directly and professionally
- Use specific data from the report when available
- If you searched for additional information, incorporate it naturally
- If asked to compare industries, be analytical
- If asked for opinions or recommendations, give them confidently based on the data
- Keep responses focused and structured
- Use bullet points for lists, be concise but thorough
- Today is {datetime.now().strftime("%B %Y")}"""
        }
    ]

    # Add conversation history
    for msg in history[-10:]:  # Keep last 10 exchanges for context
        messages.append(msg)

    # Add current message
    messages.append({"role": "user", "content": request.message})

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.4
        },
        timeout=30
    )

    reply = response.json()["choices"][0]["message"]["content"]

    # Save to history
    session["messages"].append({"role": "user", "content": request.message})
    session["messages"].append({"role": "assistant", "content": reply})

    return {
        "reply": reply,
        "searched_web": searched,
        "search_query": decision.replace("SEARCH:", "").strip() if searched else None
    }

# ── History Endpoints ────────────────────────────────────────
@app.get("/history/{industry}")
def get_industry_history(industry: str):
    industry_key = industry.lower().strip()
    reports = report_store.get(industry_key, [])
    return {"industry": industry, "report_count": len(reports), "reports": reports}

@app.get("/history")
def get_all_history():
    summary = []
    for industry_key, reports in report_store.items():
        if reports:
            summary.append({
                "industry": reports[0]["industry"],
                "report_count": len(reports),
                "last_researched": reports[0]["generated_at"]
            })
    return {"industries": summary}

@app.get("/")
def root():
    return {"status": "MarketPulse AI is running", "version": "2.0"}

@app.get("/ping")
def ping():
    return {"status": "alive"}