import dspy
import json
from typing import List, Dict, Any
import os
from dataclasses import dataclass

# ============================================================================
# SETUP: Configure DSPy with your LLM
# ============================================================================

# Option 1: OpenAI
# dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# Option 2: Anthropic (if you prefer)
# dspy.settings.configure(lm=dspy.LM("anthropic/claude-3-5-sonnet-20241022"))

# Option 3: Local/Ollama
# dspy.settings.configure(lm=dspy.LM("ollama/llama3.1", api_base="http://localhost:11434"))

print("✓ DSPy configured")
#
# ============================================================================
# MOCK DATA
# ============================================================================

JIRA_TICKETS = {
    "SHOP-2847": {
        "title": "Safari checkout crashes on iOS 17",
        "status": "In Review",
        "assignee": "Alice Chen",
        "priority": "P0",
        "description": "Users on Safari 17/iOS report checkout crashes at payment step. Hotfix deployed yesterday, monitoring for recovery.",
        "created": "2025-01-15",
        "updated": "2025-01-18"
    },
    "SHOP-2901": {
        "title": "Payment gateway timeout",
        "status": "Open",
        "assignee": "Bob Smith",
        "priority": "P1",
        "description": "Stripe webhook timeouts causing order confirmation delays.",
        "created": "2025-01-16",
        "updated": "2025-01-17"
    },
    "SHOP-3001": {
        "title": "Mobile web performance degradation",
        "status": "In Progress",
        "assignee": "Carol Wang",
        "priority": "P1",
        "description": "Mobile page load times increased 20% after new analytics integration.",
        "created": "2025-01-14",
        "updated": "2025-01-18"
    },
    "SHOP-2955": {
        "title": "Address validation API errors",
        "status": "Open",
        "assignee": "David Lee",
        "priority": "P2",
        "description": "Third-party address validation service returning 500 errors for Canadian addresses.",
        "created": "2025-01-17",
        "updated": "2025-01-17"
    }
}

CONFLUENCE_DOCS = {
    "checkout-rewrite": {
        "title": "Checkout Rewrite Q2 2025",
        "content": "Project is 75% complete and on track for Q2 delivery. Main focus areas: Safari compatibility, payment flow optimization, mobile UX improvements.",
        "updated": "2025-01-15"
    },
    "mobile-strategy": {
        "title": "Mobile Optimization Strategy 2025",
        "content": "Mobile conversion funnel analysis shows Safari-specific issues affecting iOS users. Target: improve mobile conversion rate by 15% through performance and UX enhancements.",
        "updated": "2025-01-10"
    },
    "payment-architecture": {
        "title": "Payment Flow Architecture",
        "content": "Current payment architecture uses Stripe webhooks for order confirmation. Known issues: webhook timeouts during peak traffic, retry logic needs improvement.",
        "updated": "2025-01-12"
    }
}

ANALYTICS_DATA = {
    "mobile_conversions": {
        "current": 3.2,
        "previous": 3.5,
        "trend": "down",
        "change_pct": -8.6,
        "period": "week-over-week"
    },
    "checkout_completion": {
        "current": 78.5,
        "previous": 82.1,
        "trend": "down",
        "change_pct": -4.4,
        "period": "week-over-week"
    },
    "safari_users": {
        "current": 24.3,
        "previous": 25.1,
        "trend": "down",
        "change_pct": -3.2,
        "period": "week-over-week"
    },
    "payment_success_rate": {
        "current": 96.2,
        "previous": 97.8,
        "trend": "down",
        "change_pct": -1.6,
        "period": "week-over-week"
    }
}

# ============================================================================
# TOOLS: Organized by Intent
# ============================================================================

# SEARCH TOOLS (find information)
def search_jira(query: str) -> str:
    """Search Jira tickets by keyword."""
    query_lower = query.lower()
    results = []
    
    for ticket_id, ticket in JIRA_TICKETS.items():
        if (query_lower in ticket["title"].lower() or 
            query_lower in ticket["description"].lower() or
            query_lower in ticket["priority"].lower() or
            query_lower in ticket["status"].lower() or
            query_lower in ticket["assignee"].lower()):
            results.append(
                f"{ticket_id}: {ticket['title']} "
                f"(Status: {ticket['status']}, Priority: {ticket['priority']}, "
                f"Assignee: {ticket['assignee']})"
            )
    
    if not results:
        return f"No Jira tickets found matching '{query}'"
    
    return f"Found {len(results)} ticket(s):\n" + "\n".join(results)

def search_confluence(query: str) -> str:
    """Search Confluence documentation."""
    query_lower = query.lower()
    results = []
    
    for doc_id, doc in CONFLUENCE_DOCS.items():
        if (query_lower in doc["title"].lower() or 
            query_lower in doc["content"].lower()):
            results.append(f"• {doc['title']} (Key: {doc_id}, Updated: {doc['updated']})")
    
    if not results:
        return f"No Confluence docs found matching '{query}'"
    
    return f"Found {len(results)} document(s):\n" + "\n".join(results)

# RETRIEVE TOOLS (get specific items)
def get_ticket_details(ticket_id: str) -> str:
    """Get full details for a specific Jira ticket."""
    ticket = JIRA_TICKETS.get(ticket_id.upper())
    
    if not ticket:
        return f"Ticket {ticket_id} not found"
    
    return f"""Ticket {ticket_id}: {ticket['title']}
Status: {ticket['status']}
Assignee: {ticket['assignee']}
Priority: {ticket['priority']}
Created: {ticket['created']}
Updated: {ticket['updated']}

Description:
{ticket['description']}"""

def get_confluence_doc(doc_key: str) -> str:
    """Get full content of a Confluence document."""
    doc = CONFLUENCE_DOCS.get(doc_key)
    
    if not doc:
        return f"Document '{doc_key}' not found. Available keys: {', '.join(CONFLUENCE_DOCS.keys())}"
    
    return f"""{doc['title']}
Last updated: {doc['updated']}

Content:
{doc['content']}"""

# ANALYZE TOOLS (metrics and trends)
def get_metric(metric_name: str) -> str:
    """Get current value and trend for a specific metric."""
    metric = ANALYTICS_DATA.get(metric_name)
    
    if not metric:
        available = ', '.join(ANALYTICS_DATA.keys())
        return f"Metric '{metric_name}' not found. Available: {available}"
    
    return f"""{metric_name}:
Current: {metric['current']}
Previous: {metric['previous']}
Trend: {metric['trend']} ({metric['change_pct']:+.1f}%)
Period: {metric['period']}"""

def compare_metrics(metric_a: str, metric_b: str) -> str:
    """Compare two metrics side by side."""
    data_a = ANALYTICS_DATA.get(metric_a)
    data_b = ANALYTICS_DATA.get(metric_b)
    
    if not data_a or not data_b:
        return f"One or both metrics not found: {metric_a}, {metric_b}"
    
    return f"""Comparison:
{metric_a}: {data_a['current']} ({data_a['trend']} {data_a['change_pct']:+.1f}%)
{metric_b}: {data_b['current']} ({data_b['trend']} {data_b['change_pct']:+.1f}%)"""

def list_available_metrics() -> str:
    """List all available metrics."""
    metrics = []
    for name, data in ANALYTICS_DATA.items():
        metrics.append(f"• {name}: {data['current']} ({data['trend']} {data['change_pct']:+.1f}%)")
    return "Available metrics:\n" + "\n".join(metrics)

print("✓ Tools defined\n")

# ============================================================================
# PLAN STEP SCHEMA
# ============================================================================

@dataclass
class PlanStep:
    """A single step in an execution plan."""
    subquery: str
    intent: str  # search, retrieve, or analyze
    
    def __repr__(self):
        return f"[{self.intent}] {self.subquery}"

# ============================================================================
# SPECIALIZED AGENTS BY INTENT
# ============================================================================

class SearchQuery(dspy.Signature):
    """Search for relevant information in Jira tickets and Confluence docs."""
    question: str = dspy.InputField()
    context: str = dspy.InputField(desc="Results from previous steps (if any)")
    answer: str = dspy.OutputField(desc="Summary of search results")

class SearchAgent(dspy.Module):
    """Specialized agent for searching (finding information)."""
    def __init__(self):
        super().__init__()
        self.react = dspy.ReAct(
            signature=SearchQuery,
            tools=[search_jira, search_confluence],
            max_iters=4
        )
    
    def forward(self, question: str, context: str = ""):
        result = self.react(question=question, context=context or "No previous context")
        return dspy.Prediction(answer=result.answer)

class RetrieveQuery(dspy.Signature):
    """Retrieve detailed information for specific tickets or documents."""
    question: str = dspy.InputField()
    context: str = dspy.InputField(desc="Results from previous steps (if any)")
    answer: str = dspy.OutputField(desc="Detailed information from retrieved items")

class RetrieveAgent(dspy.Module):
    """Specialized agent for retrieving (getting specific items by ID)."""
    def __init__(self):
        super().__init__()
        self.react = dspy.ReAct(
            signature=RetrieveQuery,
            tools=[get_ticket_details, get_confluence_doc],
            max_iters=3
        )
    
    def forward(self, question: str, context: str = ""):
        result = self.react(question=question, context=context or "No previous context")
        return dspy.Prediction(answer=result.answer)

class AnalyzeQuery(dspy.Signature):
    """Analyze metrics, trends, and data patterns."""
    question: str = dspy.InputField()
    context: str = dspy.InputField(desc="Results from previous steps (if any)")
    answer: str = dspy.OutputField(desc="Analysis with trends and insights")

class AnalyzeAgent(dspy.Module):
    """Specialized agent for analyzing (metrics and trends)."""
    def __init__(self):
        super().__init__()
        self.react = dspy.ReAct(
            signature=AnalyzeQuery,
            tools=[get_metric, compare_metrics, list_available_metrics],
            max_iters=4
        )
    
    def forward(self, question: str, context: str = ""):
        result = self.react(question=question, context=context or "No previous context")
        return dspy.Prediction(answer=result.answer)

print("✓ Specialized agents initialized\n")

# ============================================================================
# ORCHESTRATOR WITH INTENT-BASED ROUTING
# ============================================================================

class QueryPlanning(dspy.Signature):
    """Decompose a user question into an execution plan.
    
    Intent types:
    - search: Find relevant tickets or docs using keywords
    - retrieve: Get specific items by ID (ticket numbers, doc keys)
    - analyze: Analyze metrics, trends, or compare data
    """
    question: str = dspy.InputField()
    available_intents: str = dspy.InputField()
    plan: List[dict] = dspy.OutputField(
        desc="List of dicts with keys: subquery (str), intent (str: search/retrieve/analyze)"
    )

class ScoutOrchestrator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.planner = dspy.ChainOfThought(QueryPlanning)
        
        # Intent-based agent registry
        self.agents = {
            "search": SearchAgent(),
            "retrieve": RetrieveAgent(),
            "analyze": AnalyzeAgent(),
        }
        
        # Intent descriptions for planner
        self.intent_descriptions = """
Available intents:
- search: Use when you need to FIND tickets or docs using keywords (e.g., "find Safari issues", "search for checkout docs")
- retrieve: Use when you have specific IDs and need DETAILS (e.g., "get ticket SHOP-2847", "get doc checkout-rewrite")
- analyze: Use when you need to examine METRICS or TRENDS (e.g., "how are conversions trending?", "compare mobile vs desktop")

Examples:
- "What tickets are about Safari?" → search
- "Get details for SHOP-2847" → retrieve
- "Are mobile conversions down?" → analyze
- "Find checkout issues and check if conversions dropped" → search (find issues), analyze (check metrics)
        """.strip()
    
    def forward(self, question: str):
        # Generate plan
        plan_result = self.planner(
            question=question,
            available_intents=self.intent_descriptions
        )
        
        # Convert to PlanStep objects
        steps = []
        for step_dict in plan_result.plan:
            intent = step_dict.get("intent", "search").lower()
            # Validate intent
            if intent not in self.agents:
                print(f"⚠️  Unknown intent '{intent}', defaulting to 'search'")
                intent = "search"
            
            steps.append(PlanStep(
                subquery=step_dict.get("subquery", question),
                intent=intent
            ))
        
        # Fallback if empty plan
        if not steps:
            print("⚠️  Empty plan returned, using fallback")
            steps = [PlanStep(subquery=question, intent="search")]
        
        # Execute each step by routing to intent-specific agent
        step_results = []
        accumulated_context = ""
        
        for i, step in enumerate(steps):
            agent = self.agents[step.intent]
            result = agent(question=step.subquery, context=accumulated_context)
            step_results.append({
                "step_id": i,
                "step": step,
                "answer": result.answer
            })
            
            # Build context for next step
            accumulated_context += f"\nStep {i} ({step.intent}): {result.answer}\n"
        
        # Synthesize final answer
        if len(step_results) == 1:
            final_answer = step_results[0]["answer"]
        else:
            answers = []
            for sr in step_results:
                intent_label = sr['step'].intent.upper()
                answers.append(f"**{intent_label}**: {sr['answer']}")
            final_answer = "\n\n".join(answers)
        
        return dspy.Prediction(
            answer=final_answer,
            plan=steps,
            step_results=step_results
        )

print("✓ Orchestrator initialized\n")



# ============================================================================
# TEST QUERIES
# ============================================================================

scout = ScoutOrchestrator()

test_queries = [
    # Simple search
    "What tickets mention Safari?",
    
    # Simple retrieve
    "Get details for ticket SHOP-2847",
    
    # Simple analyze
    "How are mobile conversions trending?",
    
    # Multi-step: search then retrieve
    "Find P0 tickets and get details for the most critical one",
    
    # Multi-step: search + analyze
    "Are there checkout issues and are conversion rates down?",
    
    # Multi-step: retrieve + analyze
    "Get details for SHOP-3001 and check if mobile metrics are affected",
    
    # Complex: all three intents
    "Find Safari-related tickets, get details for SHOP-2847, and check Safari user metrics"
]

print("=" * 80)
print("SCOUT TEST QUERIES - Intent-Based Routing")
print("=" * 80)

for i, query in enumerate(test_queries, 1):
    print(f"\n{'='*80}")
    print(f"Query {i}: {query}")
    print("=" * 80)
    
    result = scout(question=query)
    
    print("\n📋 EXECUTION PLAN:")
    for j, step in enumerate(result.plan):
        print(f"   Step {j}: {step}")
    
    print(f"\n💬 ANSWER:")
    print(result.answer)
    print()

print("\n" + "=" * 80)
print("✓ All test queries complete!")
print("=" * 80)


"""
✓ DSPy configured
✓ Tools defined

✓ Specialized agents initialized

✓ Orchestrator initialized

================================================================================
SCOUT TEST QUERIES - Intent-Based Routing
================================================================================

================================================================================
Query 1: What tickets mention Safari?
================================================================================

📋 EXECUTION PLAN:
   Step 0: [search] Find tickets that mention Safari

💬 ANSWER:
The following ticket mentions Safari:
- **SHOP-2847**: Safari checkout crashes on iOS 17 (Status: In Review, Priority: P0, Assignee: Alice Chen)


================================================================================
Query 2: Get details for ticket SHOP-2847
================================================================================

📋 EXECUTION PLAN:
   Step 0: [retrieve] Get details for ticket SHOP-2847

💬 ANSWER:
Ticket SHOP-2847: Safari checkout crashes on iOS 17
- **Status**: In Review
- **Assignee**: Alice Chen
- **Priority**: P0
- **Created**: 2025-01-15
- **Updated**: 2025-01-18

**Description**:
Users on Safari 17/iOS report checkout crashes at the payment step. A hotfix was deployed yesterday, and the team is currently monitoring for recovery.


================================================================================
Query 3: How are mobile conversions trending?
================================================================================

📋 EXECUTION PLAN:
   Step 0: [analyze] Analyze the trend of mobile conversions over time.

💬 ANSWER:
The trend of mobile conversions over time shows a decline, with a week-over-week decrease of 8.6%. The current value is 3.2, down from 3.5 in the previous period. This suggests a negative trend in mobile conversions, which may require further investigation to identify potential causes and address them.


================================================================================
Query 4: Find P0 tickets and get details for the most critical one
================================================================================

📋 EXECUTION PLAN:
   Step 0: [search] Find all P0 tickets
   Step 1: [retrieve] Get details for the most critical P0 ticket

💬 ANSWER:
**SEARCH**: Found 1 P0 ticket:
- **SHOP-2847**: Safari checkout crashes on iOS 17 (Status: In Review, Priority: P0, Assignee: Alice Chen)

**RETRIEVE**: The most critical P0 ticket is **SHOP-2847**. Here are the details:

- **Title**: Safari checkout crashes on iOS 17
- **Status**: In Review
- **Assignee**: Alice Chen
- **Priority**: P0
- **Created**: 2025-01-15
- **Updated**: 2025-01-18
- **Description**: Users on Safari 17/iOS report checkout crashes at the payment step. A hotfix was deployed yesterday, and the team is currently monitoring for recovery.

This ticket is being actively worked on, with a hotfix already deployed and under observation.


================================================================================
Query 5: Are there checkout issues and are conversion rates down?
================================================================================

📋 EXECUTION PLAN:
   Step 0: [search] Find tickets or documents related to checkout issues
   Step 1: [analyze] Analyze conversion rate metrics to determine if they are down

💬 ANSWER:
**SEARCH**: No Jira tickets or Confluence documents related to "checkout issues" were found.

**ANALYZE**: The conversion rate metrics are down. Specifically, "mobile_conversions" decreased by 8.6% (from 3.5 to 3.2), and "checkout_completion" decreased by 4.4% (from 82.1 to 78.5) week-over-week. These declines indicate a negative trend in conversion performance, which may warrant further investigation to identify potential causes and address the issue.


================================================================================
Query 6: Get details for SHOP-3001 and check if mobile metrics are affected
================================================================================

📋 EXECUTION PLAN:
   Step 0: [retrieve] Get details for SHOP-3001
   Step 1: [analyze] Check if mobile metrics are affected

💬 ANSWER:
**RETRIEVE**: Ticket ID: SHOP-3001
Title: Mobile web performance degradation
Status: In Progress
Assignee: Carol Wang
Priority: P1
Created: 2025-01-14
Updated: 2025-01-18

Description:
Mobile page load times increased 20% after new analytics integration.

**ANALYZE**: Mobile metrics have been negatively affected, as evidenced by the following trends:
1. "mobile_conversions" decreased by 8.6% week-over-week, indicating fewer successful transactions on mobile.
2. "checkout_completion" dropped by 4.4%, showing a decline in the percentage of users completing the checkout process.
3. "safari_users" fell by 3.2%, suggesting reduced engagement or retention among Safari users.

These declines align with the reported 20% increase in mobile page load times, suggesting that the new analytics integration has adversely impacted mobile performance and user behavior.


================================================================================
Query 7: Find Safari-related tickets, get details for SHOP-2847, and check Safari user metrics
================================================================================

📋 EXECUTION PLAN:
   Step 0: [search] Find Safari-related tickets
   Step 1: [retrieve] Get details for ticket SHOP-2847
   Step 2: [analyze] Check Safari user metrics

💬 ANSWER:
**SEARCH**: The Safari-related ticket found is:
- **SHOP-2847**: Safari checkout crashes on iOS 17 (Status: In Review, Priority: P0, Assignee: Alice Chen)

**RETRIEVE**: Ticket SHOP-2847: Safari checkout crashes on iOS 17
- **Status**: In Review
- **Assignee**: Alice Chen
- **Priority**: P0
- **Created**: 2025-01-15
- **Updated**: 2025-01-18

**Description**:
Users on Safari 17/iOS report checkout crashes at the payment step. A hotfix was deployed yesterday, and the team is currently monitoring for recovery.

**ANALYZE**: The Safari user metrics indicate a decline in activity and performance, likely linked to the reported checkout crash issue on iOS 17 (SHOP-2847). Specifically:
- Safari user activity is down by 3.2% week-over-week (current: 24.3, previous: 25.1).
- Checkout completion rates have dropped by 4.4% (current: 78.5, previous: 82.1).
- Payment success rates have decreased by 1.6% (current: 96.2, previous: 97.8).

These trends suggest that the Safari checkout issue is impacting user engagement and conversion rates. Monitoring these metrics in the coming days will help determine if the recent hotfix is effective in reversing these declines.


================================================================================
✓ All test queries complete!
================================================================================
"""
`
