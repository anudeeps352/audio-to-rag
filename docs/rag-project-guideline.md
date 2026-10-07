# RAG Project Guideline

## Goal

Build a realistic RAG system starting from a simple customer-support use case, then improve it step by step into a stronger retrieval system.

The important part of the project is **not** "I built a chatbot."

The important part is:

> Start with a normal RAG baseline, identify concrete retrieval and production problems, introduce improvements one by one, and measure whether those improvements actually help.

The final project should demonstrate:

- semantic retrieval
- lexical retrieval
- hybrid search
- Reciprocal Rank Fusion (RRF)
- reranking
- retrieval evaluation
- answer evaluation
- multi-tenant isolation
- document/version correctness
- cache correctness
- realistic support workflows

---

# 1. Problem Statement

Customer-support teams usually have useful knowledge scattered across multiple sources:

1. uploaded internal documentation
2. scraped help-center / website pages
3. previously resolved support conversations

A naive semantic-search RAG system can work reasonably well in demos, but production introduces problems:

- exact identifiers may not retrieve well semantically
- similar documents from different customers can collide
- resolved tickets can contain useful answers not present in docs
- outdated documents can continue influencing answers
- semantically similar but irrelevant chunks can rank too highly
- caches can return stale answers
- tenant filtering mistakes can become data leaks
- improvements can sound better while actually reducing retrieval quality

The project should evolve from a naive system into a retrieval pipeline that addresses these problems.

---

# 2. Initial Product

Assume we are building a customer-support knowledge platform for multiple SaaS companies.

Each customer has three knowledge sources.

```text
Customer / Tenant
│
├── Uploaded documents
│   ├── product manuals
│   ├── policy documents
│   └── internal guides
│
├── Scraped website/help center
│   ├── FAQ pages
│   ├── troubleshooting guides
│   └── product documentation
│
└── Resolved support conversations
    ├── problem description
    ├── investigation
    └── final resolution
```

Users should be able to retrieve relevant knowledge for a support question.

Example:

```text
"My SSO login keeps redirecting back to the login page.
What should I check?"
```

The system should retrieve useful information from any of the three sources.

---

# 3. Phase 1 — Baseline RAG

Start intentionally simple.

```text
documents
   ↓
parse
   ↓
chunk
   ↓
embed
   ↓
pgvector
   ↓
vector similarity search
   ↓
top-k chunks
   ↓
LLM
   ↓
answer
```

Recommended first version:

- PostgreSQL
- pgvector
- one embedding model
- one LLM
- simple chunking
- top-k vector retrieval
- source citations
- basic API + minimal UI

Do **not** start with LangGraph, agents, multiple databases, or advanced retrieval.

The baseline needs to be simple enough that later improvements are measurable.

---

# 4. Create an Evaluation Dataset Early

Before improving retrieval, create a small benchmark.

Example:

```json
{
  "question": "How long is the enterprise refund window?",
  "expected_source": "refund-policy.md",
  "expected_answer": "30 days"
}
```

Start with roughly:

```text
50-100 questions
```

Include different question types:

- direct factual questions
- paraphrased questions
- exact identifiers
- error codes
- acronym-heavy queries
- questions requiring a resolved support conversation
- questions where multiple documents look similar
- intentionally unanswerable questions

This dataset becomes the basis for every later comparison.

---

# 5. Measure the Baseline

Measure retrieval independently from answer generation.

## Retrieval metrics

Track:

```text
Recall@1
Recall@5
MRR
NDCG
```

Example:

```text
Query:
"What is the timeout after 3 failed payment attempts?"

Expected source:
payments-policy#chunk-12

Retrieved:
1. account-security#chunk-4
2. payments-policy#chunk-12
3. retry-guide#chunk-8
```

The answer may still look correct, but the retrieval rank matters.

---

# 6. Phase 2 — Add Lexical Search

Vector search is good at semantic similarity.

It is often weaker for:

- product codes
- error IDs
- ticket numbers
- acronyms
- exact feature names
- version numbers
- account identifiers

Add lexical retrieval.

Possible implementation:

```text
PostgreSQL full-text search
or
BM25-capable search engine
```

Now run two searches:

```text
query
 │
 ├── vector search
 │
 └── BM25 / lexical search
```

Compare each independently against the benchmark.

---

# 7. Phase 3 — Hybrid Search + RRF

Do not directly compare raw BM25 scores with cosine-similarity scores.

Instead combine rankings using Reciprocal Rank Fusion.

```text
Vector ranking
    +
BM25 ranking
    ↓
   RRF
    ↓
combined ranking
```

Conceptually:

```text
RRF score =
Σ 1 / (k + rank)
```

You mainly care about rank positions rather than incompatible raw scoring scales.

Compare:

```text
Vector only
BM25 only
Hybrid + RRF
```

Example experiment:

| Strategy | Recall@5 | MRR |
|---|---:|---:|
| Vector | 0.77 | 0.70 |
| BM25 | 0.72 | 0.73 |
| Hybrid + RRF | 0.88 | 0.82 |

Do not assume hybrid is better.

Measure it.

---

# 8. Phase 4 — Add a Reranker

First-stage retrieval optimizes speed.

A reranker can evaluate a smaller candidate set more carefully.

Pipeline:

```text
BM25 top 20
+
Vector top 20
       ↓
      RRF
       ↓
top 20-30 candidates
       ↓
    reranker
       ↓
     top 5
       ↓
      LLM
```

Evaluate:

```text
Hybrid + RRF
vs
Hybrid + RRF + reranker
```

Also measure latency.

Example:

| Pipeline | Recall@5 | MRR | p95 retrieval |
|---|---:|---:|---:|
| Vector | 0.78 | 0.71 | 120 ms |
| Hybrid + RRF | 0.88 | 0.82 | 190 ms |
| + reranker | 0.92 | 0.88 | 360 ms |

The useful engineering question becomes:

> Is the reranker worth the additional latency and cost?

---

# 9. Phase 5 — Add RAG Evaluation

Retrieval metrics alone are not enough.

Use RAGAS or a similar evaluation framework to evaluate the generated answer.

Possible metrics:

```text
Context Precision
Context Recall
Faithfulness
Answer / Response Relevancy
Noise Sensitivity
```

Keep two separate evaluation layers:

```text
Retrieval evaluation
        +
Generation evaluation
```

Do not rely only on an LLM judge.

---

# 10. Multi-Tenant Data Isolation

Now make the system production-like.

Assume multiple companies use the same platform.

```text
Tenant A
Tenant B
Tenant C
```

Each can upload similar documents.

Example:

```text
Tenant A:
enterprise refund window = 30 days

Tenant B:
enterprise refund window = 7 days
```

Tenant A asks:

```text
"What is our enterprise refund window?"
```

The system must NEVER retrieve Tenant B's chunk.

---

# 11. Retrieval Must Be Tenant-Scoped

Bad design:

```text
search entire database
        ↓
retrieve candidates
        ↓
filter tenant afterwards
```

Better:

```text
authenticated user
        ↓
derive tenant_id
        ↓
retrieval executes WITH tenant filter
```

For example:

```text
tenant_id = server-side auth context
```

Never trust a tenant ID supplied directly by the client.

---

# 12. Data Model

Example chunk table:

```text
knowledge_chunks
----------------
id
tenant_id
source_id
document_id
document_version
source_type
content
metadata
embedding
search_vector
created_at
```

Possible source types:

```text
UPLOADED_DOCUMENT
WEB_PAGE
RESOLVED_TICKET
```

---

# 13. Multi-Tenant Failure Tests

These tests are part of the project, not optional extras.

## Same document name

```text
Tenant A → handbook.pdf
Tenant B → handbook.pdf
```

Expected:

```text
no collision
```

## Same semantic content

Both tenants contain almost identical policies.

Expected:

```text
Tenant A retrieves only A
Tenant B retrieves only B
```

## Same query

```text
Tenant A: "What is our refund policy?"
Tenant B: "What is our refund policy?"
```

Results must still remain isolated.

## Concurrent requests

Simulate many users querying simultaneously.

Verify there is no shared state contamination.

---

# 14. Cache Correctness

A dangerous cache key would be:

```text
cache_key = query
```

Because two tenants could ask:

```text
"What is our refund policy?"
```

A better cache key should include relevant context.

Example:

```text
tenant_id
+
corpus_version
+
permission_version
+
retrieval_pipeline_version
+
normalized_query
```

Example:

```text
rag:
tenant-123:
corpus-v8:
permissions-v3:
pipeline-v5:
refund-policy
```

---

# 15. Document Versioning

Suppose:

```text
refund-policy-v1:
refund window = 30 days
```

is replaced by:

```text
refund-policy-v2:
refund window = 14 days
```

The system must not continue serving the old answer.

Introduce:

```text
document_version
corpus_version
```

Then test stale-answer scenarios.

---

# 16. Permission Changes

Later, add document-level permissions.

Example:

```text
Support agents:
can read product docs

Finance users:
can read pricing contracts
```

If access is revoked:

```text
existing caches
existing retrieval indexes
existing conversations
```

must not leak the content afterwards.

This gives you a deeper production-style security problem.

---

# 17. Recommended Experiments

Every change should produce a before/after comparison.

Example experiment table:

| Experiment | Recall@5 | MRR | Faithfulness | p95 latency | Tenant leaks |
|---|---:|---:|---:|---:|---:|
| Dense baseline | 0.76 | 0.69 | 0.91 | 120 ms | 0 |
| BM25 | 0.73 | 0.72 | 0.90 | 75 ms | 0 |
| Hybrid + RRF | 0.87 | 0.81 | 0.94 | 185 ms | 0 |
| + reranker | 0.92 | 0.87 | 0.96 | 350 ms | 0 |

The final project should tell a story through experiments.

---

# 18. Optional Improvement — Query Rewriting

Example conversation:

```text
User:
"How does SSO authentication work?"

Later:
"What about expiration?"
```

The literal query:

```text
"What about expiration?"
```

contains too little context.

Rewrite:

```text
"What are the expiration rules related to the SSO authentication flow?"
```

Compare:

```text
Hybrid
vs
Hybrid + query rewrite
```

Again, evaluate rather than assuming it helps.

---

# 19. Optional Improvement — Contextual Embeddings

Normal chunk:

```text
"The timeout is 30 seconds."
```

This is ambiguous when embedded alone.

Contextualized representation:

```text
"This chunk comes from the Payment Retry Configuration section
of the Checkout Service operations manual.

The timeout is 30 seconds."
```

Compare retrieval performance with and without contextualization.

---

# 20. Synthetic Dataset Strategy

You do not need a real SaaS company's data.

Create three fictional companies.

Example:

```text
AcmeDesk
OrbitCRM
CloudCart
```

Give each:

```text
10-20 docs
20-30 help-center pages
50-100 resolved support tickets
```

Deliberately create overlapping terminology.

Example:

```text
All three have:
SSO
refunds
billing
API limits
account lockouts
webhooks
```

But their actual rules differ.

This makes cross-tenant retrieval bugs easy to detect.

---

# 21. Project Architecture

Suggested first version:

```text
Frontend
   ↓
Backend API
   ↓
Authentication
   ↓
tenant context
   ↓

PostgreSQL
├── metadata
├── chunks
├── full-text search
└── pgvector embeddings

Retrieval Service
├── vector search
├── lexical search
├── RRF
└── reranker

Generation Service
└── LLM

Evaluation Service
├── Recall@K
├── MRR
├── NDCG
├── RAGAS
├── isolation tests
└── cache/version tests
```

Start with PostgreSQL + pgvector if possible.

Avoid adding unnecessary infrastructure until there is a reason.

---

# 22. Suggested Development Order

```text
1. synthetic support dataset
2. ingestion
3. vector baseline
4. benchmark dataset
5. retrieval metrics
6. BM25 / lexical retrieval
7. RRF hybrid retrieval
8. reranker
9. RAGAS
10. tenant isolation
11. cache correctness
12. versioning
13. concurrency tests
14. query rewriting
15. contextual embeddings
16. dashboard / experiment comparison
```

---

# 23. Final Project Positioning

Do not present it as:

> Built a customer-support chatbot using RAG.

Present it as:

> Built and benchmarked a multi-tenant retrieval system for customer-support knowledge, combining semantic and lexical retrieval through RRF, reranking, deterministic IR metrics, RAG evaluation, and tenant-isolation tests.

The chatbot can exist.

It should **not be the impressive part**.

---

# Are RAG Systems Usually Chatbots?

Not necessarily.

A chatbot is simply the easiest interface to demonstrate RAG.

The core idea of RAG is:

```text
some task requires external knowledge

        ↓

retrieve relevant information

        ↓

give that information to a model

        ↓

use model output downstream
```

The downstream output does not need to be chat.

RAG can be used inside many background workflows.

---

# Non-Chatbot RAG Patterns

## 1. Automated Support Triage

Incoming support ticket:

```text
"My card was charged twice but my order only appears once."
```

Pipeline:

```text
ticket
   ↓
retrieve:
- similar resolved tickets
- payment docs
- incident history
   ↓
LLM
   ↓
structured result
```

Example:

```json
{
  "category": "PAYMENTS",
  "subcategory": "DUPLICATE_CHARGE",
  "priority": "HIGH",
  "team": "Payments Support"
}
```

The system then routes the ticket.

No chatbot needed.

---

# 2. Agent Copilot

Instead of AI replying directly to the customer:

```text
incoming ticket
      ↓
retrieve similar cases
      ↓
retrieve internal docs
      ↓
draft response
      ↓
human agent
      ↓
approve / edit
```

This is often safer and more realistic than fully autonomous customer support.

The AI acts as a retrieval-powered assistant.

---

# 3. Knowledge Base Generation

Resolved support tickets contain valuable knowledge.

Example:

```text
30 customers ask the same undocumented question
```

System:

```text
resolved tickets
      ↓
cluster similar issues
      ↓
retrieve related docs
      ↓
identify knowledge gap
      ↓
generate proposed help-center article
      ↓
human review
```

This turns support history into documentation.

---

# 4. Review / Feedback Summarization

Another useful RAG-style workflow:

```text
thousands of customer reviews
        ↓
retrieve / cluster relevant reviews
        ↓
LLM summarizes themes
```

Example output:

```text
Customers like:
- battery life
- camera quality

Common complaints:
- overheating
- slow fingerprint reader
- weak modem performance
```

This is retrieval + summarization rather than chat.

---

# 5. Domain-Specific Research

RAG becomes much more useful when restricted to a domain-specific corpus.

Examples:

```text
legal cases
academic papers
sports reporting
medical literature
gaming guides
aviation reports
financial reports
product reviews
```

The value comes from retrieving the correct domain evidence before generation.

---

# 6. Controlled / Synthetic RAG Evaluation

A useful way to test whether a RAG system truly follows retrieved context:

Create synthetic documents containing deliberately fictional facts.

Example document:

```text
"Researchers at Acme Labs solved the Navier-Stokes Millennium Problem in 2025."
```

Then ask questions where the model's pretraining may conflict with the synthetic corpus.

Expected behavior:

```text
answer according to retrieved corpus
```

rather than:

```text
ignore retrieval and rely on pretrained knowledge
```

This can test:

```text
grounding
faithfulness
retrieval correctness
instruction following
```

---

# General Product Lesson

A useful AI product does not necessarily invent a completely new category.

A common pattern is:

```text
existing workflow
      ↓
identify slow / repetitive step
      ↓
add retrieval + AI where uncertainty exists
      ↓
automate deterministic steps normally
      ↓
reduce effort or turnaround time
```

Example:

```text
manual workflow = 20 minutes
AI-assisted workflow = 3 minutes
```

That is often more valuable than building an "AI-first" product with no clear problem.

---

# Podcast Transcription + RAG Idea

This is a genuinely reasonable personal RAG use case.

Problem:

> Podcasts contain valuable long-form discussions, but listening to several hours of episodes just to find one discussion or recommendation takes too much time.

Instead:

```text
podcast audio
      ↓
speech-to-text transcription
      ↓
speaker / timestamp segmentation
      ↓
chunking
      ↓
embedding + lexical index
      ↓
RAG
```

Then ask:

```text
"What did they say about AI agents?"

"Which episode discussed sleep?"

"What books have the hosts recommended?"

"What was their argument about remote work?"

"Find the episode where they talked about NVIDIA's moat."

"Give me the timestamps where they discussed PostgreSQL."
```

This can become more than "chat with podcast."

---

# Better Podcast Product Direction

## Podcast Knowledge Engine

Input:

```text
RSS feed
or
podcast episode URLs
```

System automatically:

```text
download episode
      ↓
transcribe
      ↓
detect speakers
      ↓
segment topics
      ↓
extract timestamps
      ↓
index
```

Search:

```text
semantic search
+
BM25
+
RRF
+
reranker
```

Output:

```text
answer
+
episode
+
speaker
+
timestamp
+
source transcript
```

Example:

```text
Question:
"What arguments has this podcast made against remote work?"

Answer:
The hosts mainly raise three concerns...

Evidence:
Episode 142 — 31:20
Episode 159 — 18:42
Episode 188 — 52:11
```

Now the project is useful even if the user never uses a chatbot UI.

---

# Podcast Features That Could Make It Interesting

## Topic Timeline

```text
"AI Agents"
```

returns:

```text
Jan 2025 → skeptical
Jun 2025 → experimentation
Jan 2026 → optimistic
Sep 2026 → concern about reliability
```

This lets users see how opinions change over time.

---

## Recommendation Extraction

Automatically detect:

```text
books
movies
games
products
articles
people
other podcasts
```

Then:

```text
"Show every book this host has recommended."
```

---

## Contradiction / Opinion Tracking

Example:

```text
"What has the host said about Apple?"
```

System retrieves discussions across years and summarizes:

```text
2024 → positive about hardware
2025 → negative about AI strategy
2026 → positive about privacy direction
```

This is much more interesting than episode summarization.

---

## Ask Before Listening

A user gets a recommended 2-hour podcast.

Instead of immediately listening:

```text
episode
   ↓
transcribe/index
   ↓
ask:
"What topics are covered?"

"What are the strongest arguments?"

"Is there anything about AI engineering?"

"Which 20 minutes should I listen to?"
```

The system could produce:

```text
Recommended sections:

18:20-32:40
AI agents and software development

1:07:10-1:19:30
Future of search

1:44:00-1:51:20
Career advice
```

This solves a real problem:

> Deciding whether a long podcast is worth your time.

---

# Podcast Project vs Customer-Support Retrieval Project

They teach different things.

## Customer-Support RAG

Best for learning:

```text
multi-tenancy
hybrid search
RRF
reranking
RAGAS
security/isolation
production retrieval
```

Strongest for:

```text
interviews
backend / AI engineering
retrieval engineering
```

## Podcast Knowledge Engine

Best for learning:

```text
speech-to-text
long-document ingestion
semantic retrieval
timestamps
speaker metadata
topic extraction
RAG
```

Strongest for:

```text
personal usefulness
interesting demo
multimodal / audio AI
```

---

# Recommended Direction

For the retrieval-engineering portfolio project:

> Build the **customer-support RAG** first.

It provides a controlled environment where you can properly benchmark:

```text
Vector
vs
BM25
vs
Hybrid + RRF
vs
Hybrid + RRF + reranker
```

and test production issues such as:

```text
multi-tenant isolation
stale caches
document updates
permissions
concurrent requests
```

Keep the **Podcast Knowledge Engine** as a separate potential project.

It is particularly attractive if you want a later project involving:

```text
audio
transcription
RAG
semantic search
personalized information retrieval
```

---

# Core Principle

Do not optimize for:

> "How much AI can I put into this?"

Optimize for:

> "Where does probabilistic retrieval/reasoning solve something that deterministic software alone handles poorly?"

A good project should be able to answer:

```text
Why use embeddings here?

Why use BM25?

Why RRF?

Why rerank?

How do you know retrieval improved?

How do you stop data leakage?

What happens when documents change?

What happens when retrieval is wrong?

What happens when the model hallucinates?
```

If you can defend those decisions, the project demonstrates substantially more AI engineering skill than a typical RAG chatbot.
