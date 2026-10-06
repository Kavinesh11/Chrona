# Presentation Notes — DistilBERT-GNN Network Extension

**Written for:** our team, to study from and to present `DistilBERTGNN_Network_Extension.pptx` (27 slides).
Each entry = **what the slide shows** + **what to say / what to know** (talking points and likely follow-ups).
Suggested length: ~10–12 min talk (≈25–30 s per content slide, longer on 16–20).

---

## Section 1 — Introduction & Context (slides 1–4)

### Slide 1 — Title
- **Shows:** project title, the SNA case-study framing, team, "results on the real 68,841-tweet benchmark".
- **Say:** "We reproduced and *extended* DistilBERT-GNN, treating social-media event detection as a **social network analysis** problem, and we ran it on the real benchmark — not a toy."

### Slide 2 — We Modeled / From / To Answer
- **Shows:** three one-line framings — a *message network*, from *Twitter event streams*, to answer *three questions* (does structure beat text? what to filter? what does learning add over a classical baseline?).
- **Say:** State the research questions explicitly — the last one (learning vs. a classical baseline) is our novel angle.

### Slide 3 — Our Focus
- **Shows:** definition of the SNA view — a tweet is a *node* whose meaning comes from relational context (who posted, shared entities, shared vocabulary); four key terms (HIN, homogeneous graph, node filtering, community detection).
- **Say:** "A tweet in isolation is short and ambiguous; its *neighbourhood* in the graph disambiguates it. That's the core SNA idea."

### Slide 4 — The Data & Sources
- **Shows:** source + justification (QSGNN Twitter benchmark, Ren et al. 2022), the benchmark properties (Twitter, 28 days, 68,841 messages, 503 events, 22 blocks), and the HIN→homogeneous derivation with the adjacency formula.
- **Say:** "We reuse an established benchmark on purpose — it keeps our numbers comparable to Word2Vec, LDA, BERT, PP-GCN, KPGNN, QSGNN. The graph is *derived* by a deterministic rule, not hand-built: two tweets link if they share a user, entity, or keyword."
- **Know:** `A[i,j] = min{ [Σ_k W_mk · W_mkᵀ][i,j], 1 }` — the min-with-1 makes it a binary (unweighted) adjacency.

## Section 2 — Methodology (slides 5–13)

### Slide 5 — How the Graph is Built
- **Shows:** 6 steps — message nodes, user/entity/keyword edges, sentiment filtering, centrality filtering, attention-weighted passing (2-layer 4-head GAT), incremental maintenance.
- **Say:** Walk the pipeline left→right. Emphasise the two *filtering* options (content-based sentiment vs. graph-based centrality) — both are SNA-relevant choices.

### Slide 6 — How Methods Differ
- **Shows:** a progression feature-pivot → document-pivot → graph-based → DistilBERT-GNN; content-only vs graph-based cards.
- **Say:** "Feature- and document-pivot methods ignore *who* posted and *how* posts relate. Graph-based methods exploit that structure — that's the family we're in."

### Slide 7 — The Challenges? (divider)
- **Say:** transition — "Why is this hard?"

### Slide 8 — Challenges, part 1
- **Shows:** vocabulary gap, short messages, continuous arrival, node noise.
- **Say:** each is a reason a naive approach fails; they motivate contextual embeddings + graph structure + incremental updates + filtering.

### Slide 9 — Challenges, part 2 + "challenges compound"
- **Shows:** network memory (how much history to keep) and filter choice (content vs structure); the compounding callout.
- **Say:** "These aren't independent — keeping too much history dilutes new signal, and no filter means noise drowns the community."

### Slide 10 — Our Argument
- **Shows:** three cards — structure carries signal, filtering matters as much as connecting, the network must be maintained not just grown.
- **Say:** this is the thesis the rest of the deck defends.

### Slide 11 — We Evaluated / Across / Using
- **Shows:** ten baselines, 22 blocks (M0–M21), three metrics (NMI/AMI/ARI).
- **Say:** define the evaluation protocol before showing numbers.

### Slide 12 — What the Numbers Show
- **Shows:** offline NMI bar chart across all baselines (these are the **paper's reported** numbers; DistilBERT-GNN = 0.72).
- **Say:** "As reported in the paper — graph-based methods (teal) beat text-only (grey) on this benchmark; DistilBERT-GNN is best."
- **Know (honesty):** these are the paper's published numbers, the context our own measurements sit against.

### Slide 13 — Findings cards
- **Shows:** structure wins, noise removal is the edge, keep-latest beats keep-all.
- **Say:** three qualitative takeaways from the paper's ablations.

## Section 3 — Our Extension & Real Results (slides 14–21)

### Slide 14 — The Extension! (divider)
- **Say:** "Here's what *we* add: we ran the pipeline on the real data and measured the classical baselines the model is meant to beat."

### Slide 15 — Same Graph, Two Lenses
- **Shows:** Lens A = NetworkX + Gephi (Louvain, pure topology, no learning); Lens B = DistilBERT-GNN (GAT + contrastive). 
- **Say:** "The same message graph can be read two ways. Comparing them isolates what *learning* adds over *structure alone*."

### Slide 16 — What We Ran
- **Shows:** stat tiles 68,841 / 503 / 22 + the real pipeline chain; ~8 min on CPU, no node filtering.
- **Say:** "The real, unmodified pipeline on the real benchmark — full graph, so every real tweet-to-tweet tie is visible."

### Slide 17 — The Results ★ (key slide)
- **Shows:** grouped bar chart — Structure (Louvain) vs Content (DistilBERT KMeans) means: NMI 0.64/0.64, AMI 0.47/0.53, ARI 0.36/0.36 + two finding cards.
- **Say:** "Two headline facts. One: they're **nearly tied on average** — within 0.01 NMI. Two: **the winner flips block to block** (structure wins block 5, content wins block 17). Neither is enough alone, and they're complementary — which is the single strongest argument for a model that fuses both."
- **Know:** mean graph modularity 0.656 → real, strong community structure.

### Slide 18 — Seeing the Real Graph
- **Shows:** real Gephi render of block 5, nodes = tweets, sized by degree, coloured by true event.
- **Say:** "Structure is visible to the eye — same-event tweets pull together, singletons scatter. Louvain recovers 18 communities for 13 events, NMI 0.88 on this block."

### Slide 19 — One Graph, Three Partitions
- **Shows:** block 5 coloured three ways — ground-truth event, NetworkX Louvain, Gephi modularity.
- **Say:** "All three colourings land on the same clusters. The labels, our NetworkX partition, and Gephi's own modularity **agree** — strong evidence the structure really does encode the events."

### Slide 20 — What the Numbers Look Like
- **Shows:** left = PCA of real DistilBERT embeddings (block 5); right = degree distribution (block 0).
- **Say:** "Left: content separates *some* events but overlaps in the centre — why content-only caps near 0.64. Right: the graph is heavy-tailed, median degree 4 with a hub bump at ~65 — a few entity/user hubs carry it, exactly where message-passing helps."

### Slide 21 — The Live Dashboard
- **Shows:** five Streamlit tabs on real data (metrics, 3D embeddings, topology, event lifecycle, node-filter).
- **Say:** "Everything is interactive and real — only the headline KPI tiles are the paper's static reference numbers; the curves, embeddings, topology and degrees are all computed from our run."

## Section 4 — Tooling, Discussion, Conclusion (slides 22–27)

### Slide 22 — Our Tooling
- **Shows:** NetworkX / Gephi / new scripts. 
- **Say:** name the four scripts we wrote (`export_gephi`, `analyze_real`, `gephi_render`, `build_dashboard_run`) and that Gephi was driven live via its AI Server HTTP API on :8081.

### Slide 23 — Discussion & Implications
- **Shows:** real-world relevance, why SNA+NLP together, what our result implies, limitations.
- **Say:** hit real-world use (early-warning event detection), the complementarity implication (a per-window re-weighting could beat a fixed blend), and be upfront about limitations.

### Slide 24 — Our Conclusion
- **Shows:** the synthesis + the takeaway banner.
- **Say:** "On real data, structure and content are a tie that disagrees per block — the strongest case for a model that fuses both."

### Slide 25 — What We Did / What Comes Next
- **Shows:** three done + three next.
- **Say:** done = ran it, measured baselines, visualized three ways; next = full GNN run in the pinned env, scale past test mode, write it up.

### Slide 26 — References
- **Say:** name-drop the key ones if asked (Blondel = Louvain, Veličković = GAT, Newman = modularity, Bastian = Gephi, Hagberg = NetworkX).

### Slide 27 — Thank You
- **Say:** one-line recap + invite questions.

---

## 60-second version (if time is cut)
Slides 4 → 16 → 17 → 19 → 24. "Real benchmark → we ran it → structure and content tie but disagree per block → you can see the communities in Gephi → so fusing both is the right design."

## Rubric mapping (so nothing is missed)
| Rubric criterion | Slides |
|---|---|
| Introduction & Context | 1, 2, 3 |
| Network Data Collection & Sources | 4 |
| Network Modeling & Methodology | 5, 6, 8, 9, 15 |
| Analysis & Interpretation | 12, 13, 17, 19, 20 |
| Visualization & Graph Representation | 18, 19, 20, 21 |
| Discussion & Implications | 23 |
| Conclusion & Summary | 24, 25 |
| Overall Clarity & Presentation | whole deck (consistent design system) |
