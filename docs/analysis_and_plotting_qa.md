# Comprehensive Q&A — Social Network Analysis, Graph Plotting & DistilBERT-GNN

This document is a **deep study resource** covering every concept in the case study, mapped to the three-unit SNA syllabus. Answers are thorough — written so you *understand*, not just recite. All numbers are from the real 68,841-tweet benchmark.

---

## Unit 1 — Networks and Society

### A. Foundations of Social Network Analysis

**Q1. What is Social Network Analysis and why does it matter for event detection?**

Social Network Analysis (SNA) studies social structures by modelling them as graphs — nodes (actors) connected by edges (relationships). Traditional NLP treats each tweet as an isolated document and classifies it independently. SNA says: "the *connections* between tweets carry information that no single tweet contains."

Consider: tweet A says "flooding downtown." Tweet B says "evacuation order issued." Tweet C mentions the same city as A and was posted by the same user as B. Individually, these are ambiguous. But the *relational structure* — shared entity (city), shared user — links all three into a coherent event subgraph. SNA extracts this relational signal.

In our case study, the message graph connects tweets sharing users, entities, or keywords. Community detection on this graph recovers events with NMI 0.644 — *without reading any text*. That's the power of SNA: structure alone carries event information.

**Q2. What are the three levels of network analysis?**

- **Micro level (node-level):** Properties of individual nodes. In our study: degree centrality, betweenness centrality, eigenvector centrality, PageRank of individual tweets. Questions answered: "Which tweet is most important?" "Which tweet bridges two events?"

- **Meso level (group-level):** Properties of subgroups within the network. In our study: Louvain communities, modularity of partitions, event clusters. Questions answered: "Which tweets belong to the same event?" "How well-separated are the communities?"

- **Macro level (network-level):** Properties of the entire graph. In our study: density (0.072), transitivity (0.88), assortativity (+0.65), diameter (5.2), degeneracy (19.7). Questions answered: "Is this a small-world network?" "How sparse is the tweet-connection structure?"

Our case study spans all three levels — we compute micro (centrality), meso (community detection + evaluation), and macro (network measures table) analyses on the real data.

**Q3. What tools exist for graph visualisation and analysis, and which did we use?**

Major tools from your syllabus:

- **Gephi** — desktop application for interactive graph exploration. We used it via the AI Server HTTP API (plugin by MattArtzAnthro, port 8081). Capabilities: ForceAtlas2 force-directed layout, modularity detection, node ranking (size by degree), partition colouring, and PNG export. Gephi excels at publication-quality figures for small-to-medium graphs (<100K nodes).

- **NetworkX** — Python library for graph construction and analysis. We used it for Louvain community detection, centrality computation (degree, betweenness, eigenvector, PageRank), modularity scoring, and network measure calculation (density, clustering, transitivity, assortativity, path length, diameter, degeneracy). NetworkX is scriptable and reproducible — ideal for batch analysis across 22 blocks.

- **Streamlit + pyvis** — for our interactive dashboard. Pyvis renders interactive graph visualizations in the browser using vis.js physics (ForceAtlas2). Streamlit wraps everything in a 5-tab web app.

Other tools from the syllabus (not used here but worth knowing): Pajek (classic, Slovenian origin, good for large networks), UCINET (social science focus), NodeXL (Excel plugin), Cytoscape (biology/bioinformatics focus), igraph (R/Python, fast for large graphs), graph-tool (C++/Python, fastest for very large graphs).

---

### B. Network Measures (all computed on our real data)

**Q4. What is network density and what does our value tell us?**

`density = 2m / [n(n-1)]` where m = number of edges, n = number of nodes.

Density is the fraction of all *possible* edges that actually exist. Range: 0 (no edges) to 1 (complete graph / every pair connected).

Our values: block 5 density = 0.095, mean across 22 blocks = 0.072. Only ~7-10% of possible tweet pairs are connected. This is **sparse** — typical for real-world networks. It means most tweets discuss *different* events and don't share attributes.

Why sparsity matters: in a dense graph (density near 1), every node is connected to every other, so there are no meaningful communities — everything is one big blob. Sparse graphs have *structure* — dense pockets (events) connected by sparse bridges. Community detection is most valuable precisely in this sparse regime.

**Q5. What is the clustering coefficient and what does 0.75 mean?**

The **local clustering coefficient** of node v is:

`C(v) = 2 × (number of edges among v's neighbours) / [k_v × (k_v - 1)]`

where k_v is v's degree. It measures: "what fraction of my neighbours are also connected to each other?" If all your neighbours form a clique, C(v) = 1. If none of them are connected, C(v) = 0.

The **average clustering coefficient** is the mean of C(v) over all nodes.

Our block 5 value: 0.75. This means: if tweet A shares an attribute with tweets B and C, there's a 75% chance B and C also share an attribute. This is *very high* — it creates dense triangles within events. Intuitively: if two tweets both mention "earthquake" (connected to a hub tweet about the earthquake event), they probably also share other attributes (location entity, keywords).

**Q6. What is transitivity and how does it differ from the clustering coefficient?**

`transitivity = 3 × (number of triangles) / (number of connected triples)`

A connected triple is three nodes A-B-C where edges A-B and B-C exist. A triangle is when A-C also exists. Transitivity asks: "of all connected triples in the entire graph, what fraction are closed (form triangles)?"

The difference from the average clustering coefficient: clustering coefficient averages *per-node* values (each node weighted equally), while transitivity is a *global* ratio (each triple weighted equally). High-degree nodes dominate transitivity because they participate in more triples.

Our values: block 5 transitivity = 0.83, mean = 0.88. Extremely high — nearly 9 out of 10 connected triples are closed. This is because our graph is built from shared attributes: if A shares an entity with B, and B shares a keyword with C, and that entity and keyword both relate to the same event, then A and C very likely also share something.

**Q7. What is degree assortativity and what does positive assortativity mean?**

Assortativity (Newman, 2002) is the Pearson correlation coefficient between the degrees of nodes at either end of an edge:

`r = [Σ_edges (k_i × k_j) - [Σ_edges (k_i + k_j)/2]²] / [Σ_edges (k_i² + k_j²)/2 - [Σ_edges (k_i + k_j)/2]²]`

Simplified: it measures whether high-degree nodes tend to connect to other high-degree nodes (positive, **assortative**) or to low-degree nodes (negative, **disassortative**).

Our values: block 5 = +0.42, mean = +0.65. **Positively assortative**: hubs connect to hubs.

What this means in our context: hub tweets (those sharing many attributes — central to a major event) connect to *other* hub tweets about the same event. Peripheral tweets (few connections, minor events) connect to other peripheral tweets. The network self-organises by "importance level."

Most social networks are assortative (popular people befriend other popular people). Most technological networks (internet, WWW) are *disassortative* (high-degree routers connect to many low-degree end nodes). Our message graph behaves like a social network, which makes sense — it's built from social data.

**Q8. What is degeneracy (k-core decomposition) and what does k=14 or k=19.7 tell us?**

A **k-core** is the maximal subgraph where every node has degree ≥ k *within that subgraph*. You find it by repeatedly removing nodes with degree < k until no more can be removed. The **degeneracy** is the maximum k for which a non-empty k-core exists.

Algorithm (peeling): remove the node with the smallest degree; its "coreness" is its degree at removal time. Repeat until the graph is empty. The maximum coreness value is the degeneracy.

Block 5 degeneracy = 14: there exists a group of tweets where *every* tweet in the group connects to at least 14 others *within the group*. This is the densest core — likely the main event's tweets, all sharing multiple attributes.

Mean degeneracy = 19.7 across 22 blocks. Some blocks have even denser cores (up to ~30). This tells us our graphs have deeply inter-connected event cores, not just shallow connections.

Why degeneracy matters: it reveals the "depth" of community structure. A graph with degeneracy 2 has weak communities (nodes barely connected). Degeneracy 14-20 means robust, multiply-reinforced communities — exactly the structure community detection algorithms can exploit.

**Q9. What are the properties of real-world networks and does our graph exhibit them?**

From your syllabus, real-world networks typically exhibit:

1. **Heavy-tailed degree distribution:** Most nodes have few connections; a few hubs have many. ✅ Our graph: median degree 4, but hubs reach degree 65+. The degree distribution histogram (slide 20) shows this clearly.

2. **Small-world property:** High clustering coefficient + short average path length (relative to a random graph of the same size). ✅ Our graph: avg clustering 0.75 (high) + avg path length 2.94 (short, ~3 hops). This is the Watts-Strogatz small-world signature.

3. **Community structure:** Nodes cluster into densely-connected groups with sparse connections between groups. ✅ Our graph: modularity 0.656 (strong), Louvain recovers event communities at NMI 0.88 on block 5.

4. **Hubs and authorities:** A few nodes are disproportionately important. ✅ Our graph: tweet #12 has degree centrality 0.212 (connected to 21% of all tweets), betweenness 0.156 (sits on 15.6% of all shortest paths).

5. **Positive assortativity (in social networks):** ✅ Our graph: assortativity +0.65.

Our message graph exhibits *all five* hallmark properties of real-world social networks.

---

### C. Network Growth Models

**Q10. How does the Erdős–Rényi random graph model work, and does our graph match it?**

The **Erdős–Rényi model** (G(n,p)): start with n nodes, connect each pair independently with probability p. The resulting degree distribution is **binomial** (approximately Poisson for large n), concentrated around the mean degree np.

Our graph does **not** match this model. The degree distribution is heavy-tailed (median 4, hubs at 65+), not bell-shaped. In a random graph with the same density (0.072), we'd expect degrees clustered around 7.1 with very few nodes above 15. Our hubs at 60+ are impossible under this model.

**Q11. What is the Watts-Strogatz small-world model?**

Start with a **ring lattice**: n nodes in a circle, each connected to its k nearest neighbours. This has high clustering (your neighbours are also neighbours of each other) but long path lengths (to reach the far side of the ring, you must traverse many hops).

Now **rewire** each edge with probability p: disconnect one end and reconnect it to a random node. Even a small p (1–10% of edges rewired) dramatically shortens average path length (shortcuts across the ring) while barely reducing clustering (most local structure is preserved).

The result: **high clustering + short paths = small-world.** Our graph has exactly this: avg clustering 0.75 (high, like the lattice part) + avg path length 2.94 (short, like the rewired part). The "rewiring" in our context comes from tweets that mention entities from *multiple* events — they create cross-community shortcuts.

**Q12. What is the Barabási-Albert preferential attachment model?**

Start with a small seed graph. Add new nodes one at a time; each new node connects to m existing nodes with probability proportional to their *current degree*:

`P(new node connects to node i) = degree(i) / Σ_j degree(j)`

"The rich get richer" — nodes that already have many connections are more likely to get new ones. This generates a **power-law degree distribution**: P(k) ~ k^(-γ), where γ is typically 2–3.

Our graph shows this behaviour: popular entities (shared by many tweets) attract more tweets that mention them, and tweets sharing those popular entities form hubs. The degree distribution's heavy tail (slide 20) is the fingerprint of preferential attachment.

**Q13. What about Price's model, local-world models, accelerating growth, and aging?**

These are refinements from your syllabus:

- **Price's model (1976):** The directed version of preferential attachment (predates Barabási-Albert by 23 years!). New papers cite old ones proportional to their citation count. Generates a power-law in-degree distribution.

- **Local-world models:** New nodes don't see the entire network — they connect preferentially within a local neighbourhood. More realistic: a new tweet "sees" only tweets in its temporal window, not all 68,841. Our incremental architecture (22 blocks, each seeing only nearby messages) is implicitly a local-world model.

- **Accelerating growth:** The number of edges per new node increases over time (not constant m). As events grow, later tweets share more entities with the existing graph, adding more edges per tweet. We observe this indirectly: later blocks have higher mean degree.

- **Aging:** Old nodes become less attractive over time. In our context: old events stop attracting new tweets as they fade from discussion. The `remove_obsolete=2` (keep-latest) strategy implements aging by discarding old nodes entirely.

---

### D. Node Centrality Measures (all computed on real block 5)

**Q14. What is degree centrality and who are our top-degree nodes?**

`C_D(v) = degree(v) / (n - 1)`

Simply: what fraction of all other nodes is v connected to? The most direct measure of "importance" — more connections = more central.

Block 5 top nodes: #12 (0.212), #30 (0.168), #83 (0.152). Tweet #12 is connected to 21.2% of all tweets in the block — a major hub sharing attributes with one-fifth of the network.

Limitation: degree centrality treats all connections equally. A node connected to 20 peripheral nodes and a node connected to 20 hubs both have the same degree centrality, but the latter is arguably more "important." Eigenvector centrality and PageRank address this.

**Q15. What is betweenness centrality and why does it matter for event detection?**

`C_B(v) = Σ_{s≠v≠t} [σ_st(v) / σ_st]`

where σ_st is the total number of shortest paths between s and t, and σ_st(v) is how many pass through v. Normalised by dividing by (n-1)(n-2)/2.

Betweenness measures "brokerage" — how much does v sit on the paths connecting *other* nodes? High betweenness = a bridge between communities.

Block 5: tweet #12 again leads (0.156), followed by #30 (0.096), #32 (0.083). Tweet #12 is not just a hub (many connections) but a *broker* (connecting otherwise-separate communities). Removing it would increase the average path length between the communities it bridges.

For event detection, high-betweenness tweets are the ones that connect two events — they might mention entities from both, or be posted by a user active in multiple event discussions. These "bridge" tweets are ambiguous and hardest to classify correctly. The GNN's attention mechanism must learn to handle them.

This relates to **Granovetter's strength of weak ties** (Unit 1 syllabus): weak ties (low-weight, inter-community edges) carry novel information between groups. High-betweenness nodes sit on these weak ties.

**Q16. What is eigenvector centrality?**

`C_E(v) = (1/λ₁) × Σ_{u ∈ N(v)} C_E(u)`

A node's importance is proportional to the importance of its neighbours. This is the eigenvector corresponding to the largest eigenvalue λ₁ of the adjacency matrix. The equation is recursive: important nodes are those connected to other important nodes.

Block 5: tweets #2, #25, #37, #39, #40 all score 0.258 — they form a clique (all connected to each other) within a single event. In a clique, everyone is equally important because everyone is connected to the same set of important nodes.

Eigenvector centrality captures "prestige" — being connected to well-connected nodes. It's the ancestor of PageRank.

**Q17. What is PageRank and how does it relate to our work?**

`PR(v) = (1-d)/n + d × Σ_{u ∈ N(v)} PR(u) / out_degree(u)`

where d is the damping factor (typically 0.85). Interpretation: imagine a "random surfer" who starts at a random node and at each step either follows a random outgoing edge (probability d) or teleports to a random node (probability 1-d). PageRank is the long-run fraction of time the surfer spends at each node.

Originally designed by Brin and Page for ranking web pages (hence the name). In our undirected message graph, PageRank is similar to eigenvector centrality but with the damping/teleport mechanism that prevents "authority sinks" (cliques that trap all the importance).

Block 5: tweet #12 leads (0.018), same as degree and betweenness. In undirected graphs, PageRank and degree centrality correlate strongly. The value is small (0.018) because PageRank distributes probability across all 100 nodes (sums to 1).

From your syllabus, PageRank is under **Link Analysis (Unit 1)**. The paper's `--filter_method centrality` option uses centrality measures (including PageRank-like importance) to decide which nodes to keep before GNN training.

**Q18. What are PersonalizedPageRank, DivRank, SimRank, and PathSIM?**

These are from your syllabus under link analysis:

- **PersonalizedPageRank (PPR):** Instead of teleporting to a *random* node, the surfer teleports back to a specific "seed" node (or set of seeds). This biases the ranking toward nodes near the seed. Application: "given tweet X, which other tweets are most relevant to it?" — useful for ego-centric event detection.

- **DivRank:** A variant of PageRank that promotes *diversity* — it penalises nodes that are too similar to already-top-ranked nodes. Application: finding diverse representative tweets for an event (not just the 10 most central, which might all say the same thing).

- **SimRank:** Measures similarity between nodes: "two nodes are similar if they are connected to similar nodes." Recursive like eigenvector centrality. Application: finding tweets that play similar structural roles in different events.

- **PathSIM:** A meta-path-based similarity for heterogeneous networks. A meta-path is a sequence of node types (e.g., tweet→user→tweet = "two tweets by the same user"). PathSIM counts the number of meta-path instances connecting two nodes. In our HIN, relevant meta-paths include: T→U→T (same user), T→E→T (same entity), T→K→T (same keyword). The homogeneous projection `A = min(Σ W·Wᵀ, 1)` is essentially a binarised union of all single-hop meta-paths.

We don't compute PPR/DivRank/SimRank/PathSIM explicitly, but the concepts underlie the graph construction (PathSIM-style meta-paths define our edges) and the GNN's attention (a learned form of node similarity).

---

### E. Network Growth and Structure

**Q19. What is the "small-world" property and does our network have it?**

A network is **small-world** if:
1. Its clustering coefficient is much higher than a random graph with the same n and m.
2. Its average path length is comparable to (or only slightly longer than) a random graph with the same n and m.

For a random Erdős–Rényi graph with n=100 and density 0.095: expected clustering ≈ 0.095 (same as density), expected avg path length ≈ log(100)/log(9.4) ≈ 2.05.

Our block 5: clustering = 0.75 (8× higher than random!), avg path length = 2.94 (only 1.4× longer than random). **Yes, this is small-world.** Dense local clusters (events) connected by short-range bridges.

**Q20. What is the "scale-free" property?**

A network is **scale-free** if its degree distribution follows a power law: P(k) ~ k^(-γ). "Scale-free" means there's no characteristic scale — the distribution looks the same at any zoom level.

Our degree distribution is heavy-tailed but we don't formally fit a power-law exponent. Qualitatively, it has the scale-free signature: many low-degree nodes, a few extreme hubs, and no "typical" degree. The mean (23.7) is far from the median (4), which is characteristic of power-law distributions.

Strict scale-free-ness is debated in the literature (Broido & Clauset, 2019 showed many networks claimed as scale-free don't pass rigorous statistical tests). We describe our distribution as "heavy-tailed" rather than claiming strict power-law, which is the defensible position.

---

## Unit 2 — Community Structure and Link Prediction

### F. Community Detection

**Q21. What is community detection and why is it central to our case study?**

Community detection is the task of partitioning a network's nodes into groups (communities) such that nodes within a group are more densely connected to each other than to nodes outside the group.

In our case study, **community = event.** Tweets about the same event share users, entities, and keywords, creating dense within-event connections. Tweets about different events share fewer attributes, creating sparse between-event connections. Community detection on the message graph directly recovers events — no text analysis needed.

This is the most direct application of SNA to our problem: the community structure *is* the event structure.

**Q22. What is the Louvain algorithm and how does it work?**

Louvain (Blondel et al., 2008) is a greedy, hierarchical community detection algorithm:

**Phase 1 — Local moves:** Start with each node in its own community. For each node, compute the modularity gain from moving it to each neighbour's community. Move it to the community with the maximum gain (if positive). Repeat until no move improves modularity.

**Phase 2 — Aggregation:** Collapse each community into a single "super-node." Edges between communities become edges between super-nodes (weighted by the number of original edges). Self-loops represent within-community edges.

**Repeat:** Apply Phase 1 to the super-node graph. Continue until no further modularity increase is possible.

Properties:
- **Modularity-maximising:** the objective function is modularity Q (defined below).
- **No need to specify k:** unlike K-Means, Louvain discovers the number of communities automatically.
- **Fast:** O(n log n) empirically, scales to millions of nodes.
- **Stochastic:** the order of node processing matters; different random seeds can give different partitions. We fix a seed for reproducibility.

In our study: Louvain finds 225 communities in block 0 (500 nodes), 18 in block 5 (100 nodes). The over-fragmentation in block 0 (225 for 10 events) is because many singleton/peripheral tweets form their own tiny communities. Block 5's 18 communities for 13 events is much closer — hence the higher NMI (0.88 vs 0.49).

**Q23. What is modularity and what is a "good" value?**

`Q = (1/2m) × Σ_{ij} [A_{ij} - (k_i × k_j)/(2m)] × δ(c_i, c_j)`

where A_{ij} is the adjacency, k_i and k_j are degrees, m is total edges, c_i is node i's community, and δ is the Kronecker delta (1 if same community, 0 otherwise).

In English: modularity compares the actual number of within-community edges to the *expected* number if edges were placed randomly (preserving degree). Q > 0 means more within-community edges than expected; Q < 0 means fewer.

Ranges: theoretically [-0.5, 1]. Practically:
- Q ≈ 0: no community structure (random)
- Q > 0.3: usually considered "meaningful" structure
- Q > 0.7: strong community structure
- Q = 1: perfect separation (no between-community edges)

Our values: block 5 modularity = 0.76, mean = 0.656. **Strong community structure** — well above the 0.3 threshold. The events genuinely create dense, well-separated subgraphs.

**Q24. What are disjoint vs overlapping communities?**

**Disjoint (hard) partitioning:** Each node belongs to exactly one community. Louvain produces disjoint communities. In our context: each tweet belongs to exactly one event.

**Overlapping (soft) communities:** A node can belong to multiple communities simultaneously. Algorithms: OSLOM, BigCLAM, NMF-based methods. In our context: a tweet could potentially relate to two events (e.g., a tweet about both an earthquake *and* a political response to it). We use disjoint detection because the benchmark's ground-truth labels assign each tweet to one event.

**Q25. What is local community detection?**

Instead of partitioning the entire graph, local methods find the community around a specific **seed node** without examining the whole network. Useful for very large graphs where global methods are too slow.

Algorithm sketch: start from the seed, greedily expand by adding the neighbour that most improves a local quality function (e.g., local modularity, conductance). Stop when no addition improves quality.

Not used in our study (our blocks are small enough for global Louvain), but relevant for scaling beyond test mode.

**Q26. How do you evaluate community detection? What are NMI, AMI, ARI?**

See the detailed explanation under Slide 11 in the presentation notes. Summary:

| Metric | What it measures | Chance level | Perfect |
|---|---|---|---|
| NMI | Shared information between predicted and true partitions, normalised | 0 | 1 |
| AMI | NMI corrected for agreement by chance | 0 | 1 |
| ARI | Pairwise agreement (same/different cluster), chance-corrected | 0 | 1 |

**Why all three?** NMI can be inflated by many-cluster partitions (Louvain's 225 for 10 events). AMI penalises this. ARI is the strictest pairwise measure. Reporting all three gives a robust evaluation.

Our structure baseline: NMI 0.644 / AMI 0.468 / ARI 0.362 (mean over 22 blocks).
Our content baseline: NMI 0.638 / AMI 0.533 / ARI 0.361.

The AMI gap (0.468 vs 0.533) correctly penalises Louvain's over-fragmentation. NMI and ARI say they're tied; AMI says content is slightly better because it fragments less.

---

### G. Link Prediction

**Q27. What is link prediction and how does it connect to our work?**

Link prediction asks: given the current graph, which edges are likely to form in the future? Methods score pairs of unconnected nodes:

- **Common neighbours:** |N(u) ∩ N(v)| — how many shared neighbours?
- **Jaccard coefficient:** |N(u) ∩ N(v)| / |N(u) ∪ N(v)| — normalised version.
- **Adamic-Adar:** Σ_{w ∈ N(u)∩N(v)} 1/log(degree(w)) — shared neighbours, weighted by their rarity.
- **Preferential attachment:** degree(u) × degree(v) — hubs attract new edges.

In our context, link prediction relates to the **incremental** aspect: when block i+1 arrives, which new tweets will connect to existing ones? The shared-attribute mechanism makes this predictable: a new tweet mentioning entity "Tokyo" will likely connect to existing tweets that also mention "Tokyo."

The GNN implicitly performs a form of link prediction: its learned embeddings place tweets that *should* be connected (same event) nearby in the embedding space. The contrastive loss explicitly trains for this — positive pairs (same event) are pulled together, negative pairs pushed apart.

---

### H. Cascade Behaviours and Epidemic Models

**Q28. What are information cascades and how do they appear in our data?**

An information cascade occurs when a piece of information spreads through a network: person A shares content, person B (connected to A) reshares it, person C (connected to B) reshares again, and so on. The spread follows the network structure.

In our data, events *are* cascades: an earthquake happens → first-hand witnesses tweet → news outlets pick up → reaction tweets → the event "cascades" through the tweet network. Our **event lifecycle tab** (dashboard tab 4) shows this: events appear (emergence), grow (cascade peak), and decay (fading from discussion).

The incremental block structure captures this temporal dynamics: block i might see an event emerge (3 tweets), block i+1 sees it peak (15 tweets), block i+2 sees it decay (2 tweets). The keep-latest maintenance strategy (remove_obsolete=2) naturally handles this by discarding decayed events.

**Q29. What are the basic epidemic models (SIR, SIS, SEIR)?**

These model how "infections" (information, rumours, events) spread through a network:

- **SIR (Susceptible → Infected → Recovered):** Nodes start susceptible. Infected nodes spread to neighbours with probability β. Infected nodes recover (stop spreading) with probability γ. Once recovered, immune. Application: a one-time news event — people hear about it, talk about it briefly, then move on.

- **SIS (Susceptible → Infected → Susceptible):** No immunity. Recovered nodes can be re-infected. Application: recurring topics (people discuss, forget, then discuss again when it resurfaces).

- **SEIR (Susceptible → Exposed → Infected → Recovered):** Adds an "exposed" state (knows about the event but hasn't tweeted yet). More realistic for social media: people see tweets before creating their own.

Our event detection doesn't explicitly use these models, but the *temporal pattern* of events (emerge → peak → decay) mirrors SIR dynamics. The "recovered" state corresponds to tweets that no longer generate new connections (the event is over). The keep-latest strategy is analogous to removing "recovered" nodes from the active graph.

**Q30. What is the difference between simple and complex contagion?**

**Simple contagion:** A single contact is enough to "infect" (one exposure to the tweet/event is enough to make you tweet about it). Spreads fast, follows shortest paths.

**Complex contagion:** Multiple independent exposures are needed (you need to see multiple friends tweet about the event before you tweet about it). Spreads slower but through dense clusters.

In our graph, event cascades may involve both: some events spread via simple contagion (a single dramatic tweet goes viral), others via complex contagion (an event needs multiple mentions from trusted sources before people engage). The clustering coefficient (0.75) facilitates complex contagion — dense local structure means multiple exposures happen naturally.

---

## Unit 3 — Graph Representation Learning and Applications

### I. Graph Representation Learning

**Q31. What is graph representation learning and how does DistilBERT-GNN do it?**

Graph representation learning maps each node to a low-dimensional vector (embedding) that captures both its features and its structural context in the graph. The goal: nodes that are "similar" (same community, same structural role, connected) should have similar embeddings.

DistilBERT-GNN does this in two stages:

1. **Feature extraction (DistilBERT):** Each tweet → 770-d vector capturing its textual content. This is the "initial representation" — content only, no structure.

2. **Structure-aware refinement (GAT):** The GAT layers aggregate information from each node's graph neighbours, weighted by learned attention coefficients. After 2 layers, each node's embedding incorporates information from all nodes within 2 hops. The attention mechanism learns *which neighbours matter most* — a form of adaptive, learnable aggregation.

The output: 8-dimensional embeddings (--out_dim 8 by default) where same-event tweets cluster together. These are then fed to DBSCAN or K-Means for the final event assignments.

**Q32. What are Graph Neural Networks (GNNs) and how do they differ from regular neural networks?**

Regular neural networks (MLPs, CNNs, RNNs) operate on fixed-size, grid-structured inputs (vectors, images, sequences). Graphs have irregular structure: different nodes have different numbers of neighbours, there's no canonical ordering.

GNNs generalise neural networks to graph-structured data using **message passing:**

```
h_v^(l+1) = UPDATE(h_v^(l), AGGREGATE({h_u^(l) : u ∈ N(v)}))
```

At each layer l, each node v:
1. Collects "messages" from its neighbours (AGGREGATE)
2. Combines them with its own representation (UPDATE)
3. Produces a new representation h_v^(l+1)

Different GNN variants differ in how they AGGREGATE:
- **GCN (Graph Convolutional Network):** Mean of neighbours' features, weighted by degree normalisation.
- **GraphSAGE:** Sample a fixed number of neighbours, mean/max/LSTM aggregate.
- **GAT (Graph Attention Network):** Weighted sum of neighbours' features, where weights are *learned attention coefficients* — the model learns which neighbours to attend to.

Our model uses **GAT** (Veličković et al., 2018) because:
- Attention is adaptive: different events may require attending to different types of neighbours.
- Multi-head attention (4 heads) captures multiple relationship types simultaneously.
- GAT handles the heterogeneous importance of edges without needing explicit edge types.

**Q33. What is contrastive learning and how does the triplet + global-local loss work?**

Contrastive learning trains embeddings by pulling similar items together and pushing dissimilar items apart, without requiring explicit class labels during training.

**Triplet loss:** For each training example (anchor), select a positive (same event) and a negative (different event):

`L_triplet = max(0, d(anchor, positive) - d(anchor, negative) + margin)`

If the positive is closer than the negative by at least `margin`, loss is 0 (good). Otherwise, the loss pushes the positive closer and the negative farther.

**Global-local pair loss:** Additionally compares node embeddings to a "global" representation of their cluster, ensuring that nodes are close to their own cluster's centroid and far from other clusters' centroids.

The combined loss creates embeddings where same-event tweets form tight, well-separated clusters in the embedding space — exactly what DBSCAN or K-Means needs for accurate event assignment.

**Q34. What are node2vec, DeepWalk, and LINE?**

These are **unsupervised** graph embedding methods (predecessors to GNNs):

- **DeepWalk (Perozzi et al., 2014):** Perform random walks on the graph, then treat the walk sequences like sentences in Word2Vec (Skip-gram). Nodes that co-occur in random walks get similar embeddings. Captures structural similarity.

- **node2vec (Grover & Leskovec, 2016):** Like DeepWalk but with biased random walks controlled by parameters p (return) and q (in-out). Low q = BFS-like walks (captures local structure), high q = DFS-like walks (captures structural roles). More flexible than DeepWalk.

- **LINE (Tang et al., 2015):** Preserves first-order proximity (direct connections) and second-order proximity (shared neighbours) using two separate objectives.

These methods learn from *structure only* — they don't use node features (tweet text). GNNs like GAT improve on them by incorporating both structure *and* features. Our DistilBERT embeddings give the GAT rich initial features; the message-passing adds structural context. This combination is why GNN-based methods outperform DeepWalk/node2vec/LINE on tasks where both content and structure matter.

---

### J. Anomaly Detection in Networks

**Q35. What is anomaly detection in networks and how does it relate to event detection?**

Network anomaly detection identifies nodes, edges, or subgraphs that deviate from the "normal" network pattern. Types:

- **Node anomalies:** Nodes with unusual feature values or connectivity patterns. In our context: a spam tweet with abnormally high degree (connected to everything) or a bot account with suspicious posting patterns.

- **Edge anomalies:** Edges that are unexpected given the network structure. In our context: a connection between tweets that share no obvious attributes (potential data error or unusual cross-event reference).

- **Subgraph anomalies:** Groups of nodes forming unusual patterns. In our context: coordinated inauthentic behaviour (bot networks posting synchronized tweets to amplify a narrative).

Event detection is related but distinct: events are *expected* communities, not anomalies. However, anomaly detection could complement event detection by identifying:
- Bot-driven fake events (coordinated subgraphs with anomalous timing/content)
- Noise nodes that should be filtered (anomalously high connectivity to unrelated events)
- Emerging events (a sudden new cluster forming = an anomaly in the temporal pattern)

**Q36. What role does node filtering play, and is it a form of anomaly detection?**

Yes, loosely. The paper's node filtering strategies identify and remove tweets that are "uninformative" — a form of noise detection:

- **Sentiment-confidence filtering:** Tweets where DistilBERT is uncertain (confidence near 0.5) are "anomalous" in the sense that they're ambiguous — the model can't determine their sentiment, suggesting they're off-topic, sarcastic, or too vague. These are removed.

- **Centrality-based filtering:** Tweets with very low centrality are structurally peripheral — they don't participate in any event community meaningfully. These are "anomalous" in the structural sense (outliers in the degree/centrality distribution).

Both filters remove ~50% of tweets and improve downstream clustering. This is conceptually similar to anomaly detection: identify and remove data points that don't fit the expected pattern (informative, event-relevant tweets).

---

### K. Applications

**Q37. How does this work relate to detecting malicious activities on online social networks?**

From your Unit 3 syllabus: malicious activities on OSNs include spam, phishing, sockpuppetry, and coordinated manipulation. Our pipeline's techniques directly transfer:

- **Sockpuppet detection:** Sockpuppets (fake accounts controlled by one entity) create anomalous structural patterns — multiple accounts posting similar content, sharing the same entities, forming a suspiciously tight cluster. Our community detection + centrality analysis would reveal these: an artificial "event" community with unusually high internal density and suspicious timing.

- **Coordinated inauthentic behaviour:** Bot networks amplify messages in coordinated waves. The event lifecycle analysis (dashboard tab 4) would show an abnormal pattern: too-fast emergence, too-uniform spread, too-synchronised posting times.

- **Spam detection:** Spam tweets often have anomalously high degree (they mention many popular entities/hashtags to maximise visibility) but low eigenvector centrality (they're not connected to other *important* tweets). This signature (high degree, low eigenvector) is detectable by our centrality analysis.

**Q38. How does this relate to modelling the spread of a pandemic?**

The SIR/SIS/SEIR models from Q29 were originally developed for disease spread. The same mathematics applies to information spread on social media:

- **Nodes** = people (or tweets) 
- **Edges** = contacts (or shared attributes)
- **Infection** = exposure to information about an event
- **β (transmission rate)** = how likely a tweet's neighbours are to also tweet about the event
- **γ (recovery rate)** = how quickly people stop tweeting about the event

Our event lifecycle data could be fit to SIR curves: emergence (exponential growth, R₀ > 1), peak (inflection point), and decay (R_t < 1, event fading). The message graph's structure determines how fast information spreads: high clustering facilitates local saturation, while bridge nodes (high betweenness) enable cross-community spread.

For actual pandemic tracking: replace tweets with epidemiological reports, entities with locations, and events with disease outbreaks. The same graph-based clustering would group related outbreak reports into coherent "events" (outbreaks). This is active research in computational epidemiology.

**Q39. What is collusion detection and how do network methods help?**

Collusion = secret agreement between parties to deceive others. In network terms: colluding entities form a hidden community with unusually strong internal connections and coordinated behaviour.

Detection approaches:
- **Structural:** Look for groups with anomalously high internal density relative to their external connections (high local modularity).
- **Temporal:** Colluding accounts act in suspiciously synchronised patterns (coordinated posting times, sequential actions).
- **Content:** Colluding accounts share unusually similar language/content.

Our graph-based approach (build a graph from shared attributes → detect communities → evaluate their properties) is directly applicable. A colluding group would appear as a community with suspicious properties: too-high density, too-similar content, too-regular timing.

---

## Cross-Cutting Concepts

### L. The DistilBERT-GNN Pipeline

**Q40. Walk through the full pipeline step by step.**

1. **Raw data:** 68,841 tweets with metadata (user, timestamp, text).

2. **Preprocessing/NER:** Extract named entities (people, places, organisations) and keywords from tweet text. Assign each tweet a user, entity set, and keyword set.

3. **DistilBERT embeddings** (`generate_initial_features.py`): Feed each tweet's text through DistilBERT (66M-parameter distilled BERT). Output: 768-d contextual embedding per tweet, concatenated with 2 additional features → 770-d vector.

4. **HIN construction:** Build a heterogeneous information network with 4 node types (tweet, user, entity, keyword) and typed edges.

5. **Homogeneous projection** (`custom_message_graph.py`): Project HIN to tweet-tweet graph: A[i,j] = 1 if tweets i and j share any attribute. Result: sparse, binary, symmetric adjacency matrix.

6. **Block splitting:** Divide into 22 chronological blocks (M0=500, M1–M21=100 each).

7. **Node filtering** (`node_filter.py`): Apply sentiment-confidence or centrality-based filtering to remove ~50% of uninformative nodes.

8. **GAT encoding** (`model.py`, `layers.py`): 2-layer, 4-head Graph Attention Network transforms 770-d input embeddings into 8-d output embeddings using attention-weighted message passing.

9. **Contrastive loss:** Triplet loss + global-local pair loss trains the embeddings to cluster by event.

10. **Clustering:** DBSCAN or K-Means on the 8-d embeddings produces event assignments.

11. **Evaluation:** Compare predicted clusters to ground-truth labels using NMI/AMI/ARI.

12. **Incremental loop** (`main.py`): Repeat steps 7-11 for each new block. Every `window_size` blocks (default 3), retrain the model (step 8-9) on fresh data.

**Q41. What is the novel contribution of our extension (beyond reproducing the paper)?**

We add a **classical SNA baseline comparison** that the original paper does not include:

**Lens A (structure-only):** Louvain community detection on the homogeneous message graph — no text, no learning, no embeddings. Just graph topology.

**Lens B (content-only):** K-Means clustering on the raw DistilBERT embeddings — no graph, no GNN. Just text content.

**Key finding:** They tie on average (NMI 0.644 vs 0.638), but the winner *flips per block*. This is the strongest empirical argument for the GNN: neither signal alone is sufficient, and they're complementary exactly where it matters. A model that fuses both — weighting structure when it's stronger, content when it's stronger — should beat either alone. That's what DistilBERT-GNN's attention mechanism does.

**Q42. Why does structure win on some blocks and content on others?**

Structure (Louvain) wins when events have *distinct structural signatures* — they use unique entities, unique users, unique hashtags. The graph clearly separates them into communities. Example: block 5 (NMI 0.88) has 13 well-separated events with distinct entity profiles.

Content (DistilBERT) wins when events have *distinct textual content* but ambiguous structure — they use overlapping entities/users but talk about different things. The text disambiguates what the structure can't. Example: block 17 where multiple events share popular entities but have distinguishable language.

The GNN's attention mechanism can, in principle, learn this per-block/per-node: attend more to graph neighbours when structure is informative, rely more on the node's own DistilBERT features when content is informative.

---

### M. Reproducibility and Tooling

**Q43. How do you reproduce the full analysis?**

```bash
# 1. Install dependencies (from repo root)
pip install -r requirements.txt

# 2. Generate DistilBERT embeddings (from DistilBERTGNN/)
cd DistilBERTGNN
python generate_initial_features.py

# 3. Build incremental message graphs
python custom_message_graph.py  # set test=True for quick run

# 4. Export GEXF files for Gephi
python ../tools/export_gephi.py --all-blocks --out-dir ../real_analysis_outputs

# 5. Run structure vs content baseline comparison
python ../tools/analyze_real.py --data-path incremental_test_100messagesperday --out ../real_analysis_outputs

# 6. Render in Gephi (requires Gephi Desktop + AI Server running on :8081)
python ../tools/gephi_render.py --gexf ../real_analysis_outputs/real_block5.gexf \
  --out-dir ../real_analysis_outputs/gephi \
  --columns ground_truth_event louvain_community modularity_class

# 7. Launch interactive dashboard
python ../tools/run_dashboard.py  # opens http://localhost:8501

# 8. (Optional) Full GNN training — requires torch 2.2 + dgl 2.2
python main.py --filter_method sentiment --remove_obsolete 2 --window_size 3
```

**Q44. What are the limitations and what would you do differently with more time?**

**Limitations:**
1. One platform (Twitter), one time window (28 days) — unknown generalisability to other social media or time periods.
2. Test-mode caps (100 msgs/block) — real-world blocks would be larger and computationally heavier.
3. 503 pre-labeled events — not open-set (can't discover new event *types* beyond training).
4. Full GNN metrics pending — we measured baselines but not the actual model on this machine (torch/dgl environment issue).
5. Louvain is stochastic — we report a single seeded run, not a stability distribution.
6. Centrality measures computed on test-mode blocks — results on full-scale blocks may differ.

**With more time:**
1. Pin the torch 2.2 + dgl 2.2 environment and run the full DistilBERT-GNN to get its real numbers alongside our baselines.
2. Lift the 100-msg cap to full-scale analysis.
3. Add Louvain stability analysis (100 random seeds, report mean ± std of NMI).
4. Build a per-window "structure-vs-content router" that automatically weights the signals based on graph properties (density, modularity) — exploiting the complementarity we discovered.
5. Compare more community detection methods (Leiden, Infomap, label propagation).
6. Formal power-law fit on the degree distribution (using the Clauset-Shalizi-Newman method).
