# Using metrics and rewards to optimize multi step agents in DSPy
## Why functional workflows still fail users, and how to make behavior learnable using differentiable programming.

import re
FRIENDLY_WORDS = {"hi", "hey", "please", "thanks", "thank you", "sure", "happy"}

def friendly_eta_metric(ex, pred, trace=None):
    # 0. Tag must match
    if ex.tag != pred.tag:
        return 0.0

    # For non-ETA tickets, any correct tag scores 1.0
    if ex.tag != "eta":
        return 1.0

    body = pred.body.lower()

    rules = [
        "eta" in body,                           # mentions ETA
        any(w in body for w in FRIENDLY_WORDS),  # at least one polite word
        5 <= len(body.split()) <= 40,            # not too short / long
        re.search(r"\b\d{1,2}\s?min", body) is not None  # gives a minutes estimate
    ]

    return sum(rules) / len(rules)              # 0.0-1.0

import dspy
import re
import os

# Securely load OpenAI API key from environment variable
openai_key = os.environ.get("OPENAI_API_KEY")
if not openai_key:
    raise RuntimeError("OPENAI_API_KEY environment variable not set.")
lm = dspy.LM('openai/gpt-4o-mini', api_key=openai_key)
dspy.configure(lm=lm)
print(f"\u2713 DSPy configured with OpenAI GPT-4o-mini (env key)")

# ── Signatures
class ServiceReply(dspy.Signature):
    tag:  str = dspy.OutputField()
    body: str = dspy.OutputField()

class RouterSignature(dspy.Signature):
    ticket: str = dspy.InputField()
    route:  str = dspy.OutputField(desc='eta | missing | driver | fallback')

class LatePathSignature(dspy.Signature):
    ticket: str = dspy.InputField()
    body:   str = dspy.OutputField(desc='Friendly ETA sentence that includes the word "eta".')

# ── Modules
class Router(dspy.Module):
    def __init__(self):
        self.step = dspy.Predict(RouterSignature)
    def forward(self, ticket: str):
        return self.step(ticket=ticket).route.lower().strip()

class LatePath(dspy.Module):
    def __init__(self):
        self.step = dspy.Predict(LatePathSignature)
    def forward(self, ticket: str):
        body = self.step(ticket=ticket).body.strip()
        return dspy.Prediction(tag="eta", body=body, _sig=ServiceReply)

class MissingPath(dspy.Module):
    def forward(self, ticket: str):
        return dspy.Prediction(tag="missing",
                               body="Item verified missing via photo. Refund has been issued.",
                               _sig=ServiceReply)

class DriverPath(dspy.Module):
    def forward(self, ticket: str):
        return dspy.Prediction(tag="driver",
                               body="Driver located, contact info sent.",
                               _sig=ServiceReply)

class FallbackPath(dspy.Module):
    def forward(self, ticket: str):
        return dspy.Prediction(tag="fallback",
                               body="Please see our FAQ or reach live support.",
                               _sig=ServiceReply)

# ── Top-level agent
class SupportAgent(dspy.Module):
    def __init__(self):
        self.router   = Router()
        self.eta      = LatePath()
        self.missing  = MissingPath()
        self.driver   = DriverPath()
        self.fallback = FallbackPath()
    def forward(self, ticket: str):
        route = self.router(ticket=ticket)
        if route == "eta":     return self.eta(ticket=ticket)
        if route == "missing": return self.missing(ticket=ticket)
        if route == "driver":  return self.driver(ticket=ticket)
        return self.fallback(ticket=ticket)


THREADS = 1
dev = [
    dspy.Example(ticket="Order #1001 is 10 minutes late.", tag="eta").with_inputs("ticket"),
    dspy.Example(ticket="Missing fries in order #1002.", tag="missing").with_inputs("ticket"),
    dspy.Example(ticket="Driver can't be reached for order #1003.", tag="driver").with_inputs("ticket"),
    dspy.Example(ticket="Order #1004 delivered but nothing here.", tag="missing").with_inputs("ticket"),
    dspy.Example(ticket="Order #1005 is 25 minutes late.", tag="eta").with_inputs("ticket"),
    dspy.Example(ticket="How do I cancel order #1006?", tag="fallback").with_inputs("ticket"),
    dspy.Example(ticket="Order #1007 missing drink.", tag="missing").with_inputs("ticket"),
    dspy.Example(ticket="Driver for order #1008 is lost.", tag="driver").with_inputs("ticket"),
    dspy.Example(ticket="Order #1009 is late, any update?", tag="eta").with_inputs("ticket"),
    dspy.Example(ticket="App says delivered but no food for #1010.", tag="missing").with_inputs("ticket"),
    dspy.Example(ticket="Order #1011: driver not moving.", tag="driver").with_inputs("ticket"),
    dspy.Example(ticket="Order #1012 is 30 minutes late.", tag="eta").with_inputs("ticket"),
    dspy.Example(ticket="No utensils in order #1013.", tag="missing").with_inputs("ticket"),
    dspy.Example(ticket="Order #1014: can I get ETA?", tag="eta").with_inputs("ticket"),
    dspy.Example(ticket="Order #1015: missing side salad.", tag="missing").with_inputs("ticket"),
    dspy.Example(ticket="Driver phone off for order #1016.", tag="driver").with_inputs("ticket"),
    dspy.Example(ticket="Order #1017: how to contact support?", tag="fallback").with_inputs("ticket"),
    dspy.Example(ticket="Order #1018: late and missing item.", tag="missing").with_inputs("ticket"),
    dspy.Example(ticket="Order #1019: ETA please?", tag="eta").with_inputs("ticket"),
    dspy.Example(ticket="Order #1020: driver not assigned.", tag="driver").with_inputs("ticket"),
]
print("Before evaluate")
evaluate = dspy.Evaluate(
    devset=dev,
    metric=friendly_eta_metric,
    num_threads=THREADS,
    display_progress=True,
    display_table=5,
)

support_bot = SupportAgent()
evaluate(support_bot)


teacher_lm = dspy.LM('openai/gpt-4o-mini', api_key=openai_key)
print(f"✓ DSPy configured for teacher_lm with OpenAI GPT-4o")

train = [
    dspy.Example(ticket="Order #2001 is 15 minutes late.", tag="eta").with_inputs("ticket"),
    dspy.Example(ticket="Order #2002 missing fries.", tag="missing").with_inputs("ticket"),
    dspy.Example(ticket="Driver for order #2003 not answering.", tag="driver").with_inputs("ticket"),
    dspy.Example(ticket="Order #2004: how do I cancel?", tag="fallback").with_inputs("ticket"),
    dspy.Example(ticket="Order #2005 is 20 minutes late.", tag="eta").with_inputs("ticket"),
    dspy.Example(ticket="Order #2006: missing drink.", tag="missing").with_inputs("ticket"),
    dspy.Example(ticket="Driver for order #2007 lost.", tag="driver").with_inputs("ticket"),
    dspy.Example(ticket="Order #2008: app says delivered but nothing here.", tag="missing").with_inputs("ticket"),
    dspy.Example(ticket="Order #2009: ETA please?", tag="eta").with_inputs("ticket"),
    dspy.Example(ticket="Order #2010: driver not assigned.", tag="driver").with_inputs("ticket"),
    dspy.Example(ticket="Order #2011: missing utensils.", tag="missing").with_inputs("ticket"),
    dspy.Example(ticket="Order #2012: can I get ETA?", tag="eta").with_inputs("ticket"),
    dspy.Example(ticket="Order #2013: missing side salad.", tag="missing").with_inputs("ticket"),
    dspy.Example(ticket="Driver phone off for order #2014.", tag="driver").with_inputs("ticket"),
    dspy.Example(ticket="Order #2015: how to contact support?", tag="fallback").with_inputs("ticket"),
    dspy.Example(ticket="Order #2016: late and missing item.", tag="missing").with_inputs("ticket"),
    dspy.Example(ticket="Order #2017: ETA for my order?", tag="eta").with_inputs("ticket"),
    dspy.Example(ticket="Order #2018: driver not moving.", tag="driver").with_inputs("ticket"),
    dspy.Example(ticket="Order #2019: missing dessert.", tag="missing").with_inputs("ticket"),
    dspy.Example(ticket="Order #2020: how do I get a refund?", tag="fallback").with_inputs("ticket"),
]
THREADS=1
optimizer = dspy.MIPROv2(
    metric=friendly_eta_metric,
    auto="light",                 # minimal search space
    num_threads=THREADS,
    teacher_settings=dict(lm=teacher_lm),
    prompt_model=dspy.settings.lm # reuse default mini LM for prompts
)

optimized_support_bot = optimizer.compile(
    SupportAgent(),               # program to optimize
    trainset=train[:100],
    requires_permission_to_run=False,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
)
print("After optimization")
print("Change done in Docker")
evaluate(optimized_support_bot)


