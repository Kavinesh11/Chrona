# Analysis & Graph-Plotting — Q&A

**Written for:** our team, to prepare for examiner / audience questions on the analysis and the
graph-plotting (NetworkX + Gephi + dashboard) parts of the case study. Answers are kept short and
defensible; numbers are the real ones from `real_analysis_outputs/analysis.json`.

---

## A. Social Network Analysis fundamentals

**Q1. Why is event detection a *network* problem and not just text classification?**
A tweet alone is short, noisy, ambiguous. What disambiguates it is relational context — who posted it,
which tweets mention the same entities, which share vocabulary. Modelling those relations as a graph lets
structure carry signal that text alone doesn't. Our result shows it: structure-only clustering scores on
par with a strong language model.

**Q2. What is the node and what is the edge in your graph?**
Nodes = tweets (messages). An edge connects two tweets if they share at least one user, named entity, or
keyword. Formally the homogeneous adjacency is `A[i,j] = min{ [Σ_k W_mk·W_mkᵀ][i,j], 1 }`, where `W_mk`
is the tweet-to-attribute incidence matrix; the product counts shared attributes and `min(…,1)` binarises it.

**Q3. What is a Heterogeneous Information Network (HIN) and why project it to a homogeneous graph?**
The HIN has four node *types* (tweet, user, entity, keyword) and typed edges. We project it to a
tweet–tweet (homogeneous) graph because community detection and the GNN operate on message-to-message
relations; the projection summarises "these two tweets are related" via shared attributes.

**Q4. Which SNA techniques did you actually use, and why each?**
- **Graph construction** (HIN → homogeneous) — makes implicit relations explicit and computable.
- **Community detection** (Louvain) — events *are* communities; recovers them with no labels.
- **Modularity** — measures how strong that community structure is.
- **Centrality** (degree/closeness/betweenness) — ranks node importance; used as one node-filtering option.
- **Degree distribution** — characterises the network (hubs vs periphery).
- **ForceAtlas2 layout** — positions nodes so structure is visible.

## B. Community detection & modularity

**Q5. What is the Louvain algorithm?**
A greedy, modularity-maximising community detection method (Blondel et al. 2008). It repeatedly moves nodes
to the neighbouring community that most increases modularity, then aggregates communities into super-nodes
and repeats. Fast, scales to large graphs, no need to pre-set the number of communities.

**Q6. What is modularity? What's a "good" value?**
Modularity Q measures how many more edges fall *inside* communities than you'd expect by chance (range ~−0.5
to 1). Q ≈ 0 means no community structure; Q > ~0.3 is usually considered meaningful structure. Our real
graphs average **Q ≈ 0.656** — strong, real community structure.

**Q7. Louvain gives one partition — is it stable?**
Louvain is stochastic (order-dependent) and can give slightly different partitions per run. We fix a seed
for reproducibility. Gephi's own "community stability" tooling re-runs it and measures how often node pairs
co-cluster; for a rigorous claim you'd report mean stability, not a single draw. (We note this as a caveat.)

**Q8. Louvain found 225 communities for 500 nodes in block 0 but only 10 events — why?**
On a sparse block, Louvain over-fragments: many tiny or singleton communities form because low-degree tweets
have no strong community to join. That's exactly why block 0's NMI is lower (0.49) — fragmentation inflates
the community count and hurts agreement with the 10 ground-truth events.

**Q9. Did you try other community methods?**
Louvain is our structure-only baseline. Gephi also exposes Leiden-style/modularity and we cross-checked the
"modularity class" against Louvain — they agree (slide 19). Alternatives (Infomap, label propagation) are
future work.

## C. Evaluation metrics (NMI / AMI / ARI)

**Q10. What do NMI, AMI, ARI measure?**
All three compare a predicted clustering to the ground-truth labels (0 = random, 1 = perfect):
- **NMI** (Normalized Mutual Information) — shared information between the two partitions, normalised.
- **AMI** (Adjusted MI) — NMI corrected for chance (important when cluster counts differ).
- **ARI** (Adjusted Rand Index) — agreement on *pairs* of points (same/different cluster), chance-corrected.

**Q11. Why report all three instead of just accuracy?**
Clustering has no fixed label-to-cluster mapping, so accuracy is ill-defined. NMI/AMI/ARI are
permutation-invariant and comparable across different numbers of clusters. ARI is the strictest; AMI guards
against the "many tiny clusters inflate NMI" effect.

**Q12. Why is AMI sometimes lower than NMI for the structure baseline?**
Because Louvain produces *more* clusters than there are events; NMI can be optimistic under high cluster
counts, while AMI penalises that chance agreement — so AMI < NMI signals over-fragmentation.

## D. The two baselines & the headline result

**Q13. What exactly are the two baselines you compare?**
- **Structure-only:** Louvain community detection on the message graph — no text, no embeddings, no learning.
- **Content-only:** K-Means on the real DistilBERT embeddings (k = number of ground-truth events in the
  block) — no graph.
Both are scored against the ground-truth events with NMI/AMI/ARI.

**Q14. What are the numbers?**
Mean over 22 real blocks: structure Louvain **NMI 0.644 / AMI 0.468 / ARI 0.362**, content KMeans/DistilBERT
**NMI 0.638 / AMI 0.533 / ARI 0.361**. Nearly tied on NMI/ARI; content a bit ahead on AMI.

**Q15. What is the key finding / "so what"?**
Two things: (1) the two signals are **tied on average** (ΔNMI < 0.01), and (2) the **winner flips block to
block** (structure wins block 5: 0.88 vs 0.77; content wins block 17: 0.52 vs 0.32). So neither signal is
sufficient alone and they're complementary — the empirical motivation for a model (DistilBERT-GNN) that
fuses structure *and* content per node.

**Q16. Isn't it disappointing that structure doesn't beat content outright?**
No — a tie is the point. If one dominated, you'd just use that one. The *disagreement per block* is what
makes a learned fusion valuable: the model can lean on whichever signal is stronger for each window.

**Q17. Why didn't you report the full DistilBERT-GNN model numbers?**
The full GNN training (`main.py`) needs the paper's pinned environment (torch 2.2 / dgl 2.2). This machine's
dgl wheel has no binary for the installed torch build, so GNN message-passing can't run here. We measured the
two *classical baselines* the model is meant to beat; the model run is listed as next step. (The 0.72 NMI on
slide 12 is the paper's reported number.)

**Q18. Why is block 7 / block 15 low for *both* baselines?**
Those blocks have few events (8) but dense, ambiguous connectivity — tweets share generic vocabulary across
events, so neither structure nor content separates them cleanly. They're the "hard" windows where a learned
model would add the most.

## E. Graph plotting — NetworkX

**Q19. What is NetworkX and what did you use it for?**
A Python library for building and analysing graphs. We used it to load the sparse adjacency
(`nx.from_scipy_sparse_array`), run Louvain (`nx.algorithms.community.louvain_communities`), compute
modularity and centralities, and generate the degree-distribution and embedding figures.

**Q20. Explain the degree-distribution plot (slide 20 right).**
X-axis = node degree (how many other tweets a tweet is connected to), Y-axis = how many tweets have that
degree. It's **heavy-tailed**: ~240 tweets have degree 1–3 (median 4), with a secondary bump near degree 65
(a cluster of highly inter-connected tweets sharing many attributes). Mean degree 23.7.

**Q21. Why does the degree distribution matter?**
It shows the network is non-trivial — not uniform, not just disconnected pairs. A few hubs tie many tweets
together while most are peripheral. That hub-and-spoke structure is precisely the regime where GNN
message-passing (aggregating from neighbours) adds value.

**Q22. Is it a power law / scale-free network?**
It's heavy-tailed and hub-dominated, consistent with the scale-free-like behaviour typical of social graphs,
but we don't formally fit a power-law exponent here — we describe it qualitatively (median 4, hub bump ~65).

**Q23. Explain the PCA embedding scatter (slide 20 left).**
Each point is a tweet's 770-d DistilBERT embedding projected to 2-D by PCA, coloured by true event. Some
events form tight clusters (e.g. the orange event on the left), but the centre is a mush of overlapping
events — visually explaining why content-only clustering tops out around 0.64 NMI.

**Q24. PCA vs t-SNE — which and why?**
The static figure uses PCA (linear, fast, preserves global variance, deterministic). The dashboard lets you
switch to t-SNE (non-linear, better local cluster separation but slower and stochastic). PCA is the honest
default for "how separable is the raw space".

## F. Graph plotting — Gephi

**Q25. What is Gephi and how did you drive it?**
Gephi is an interactive graph-visualisation and analysis tool. We drove it programmatically via the "Gephi
AI Server" plugin's HTTP API on `127.0.0.1:8081` — our `gephi_render.py` posts to `/project/new`,
`/import/gexf`, `/statistics/degree`, `/statistics/modularity`, `/appearance/ranking/size`,
`/appearance/partition/color`, `/layout/run` (ForceAtlas2) and `/export/png`. No manual clicking.

**Q26. What is a GEXF file?**
Graph Exchange XML Format — Gephi's native graph format. Our `export_gephi.py` writes one per block with
node attributes attached: `ground_truth_event`, `degree`, `degree_centrality`, and `louvain_community`.

**Q27. What is ForceAtlas2 and what do positions mean?**
A force-directed layout: edges act like springs pulling connected nodes together, all nodes repel each
other. At equilibrium, densely-connected groups (communities) sit together and weakly-connected nodes drift
to the margins. So *spatial proximity ≈ structural similarity* — that's why same-event tweets cluster
visually (slide 18).

**Q28. Why are nodes different sizes / colours in the Gephi figure?**
Size = degree (bigger = more connections = more central/hub-like). Colour = a partition: ground-truth event,
Louvain community, or Gephi modularity class, depending on the figure. Slide 19 shows all three agree.

**Q29. What does "all three partitions agree" (slide 19) prove?**
That the community structure is real and label-aligned: an unsupervised method (Louvain), Gephi's independent
modularity, and the human ground-truth labels all carve the graph the same way. Strong evidence structure
encodes events.

**Q30. Why only 100-node blocks in Gephi? Isn't the real data bigger?**
We run the repo's documented "test mode" (100 msgs/block, 500 for block 0) — real tweets, capped per block.
This keeps the graph interactively viewable; the full `test=False` build produces ~16M-edge graphs that no
interactive tool renders usefully. Scaling past the cap (with OpenOrd layout for large graphs) is future work.

## G. The dashboard

**Q31. What's in the dashboard and is it real?**
Five Streamlit tabs on the real data: (1) metric curve over 22 blocks, (2) 3-D DistilBERT embedding space,
(3) interactive message-topology map (pyvis), (4) event emergence/decay lifecycle, (5) node-filter + degree
inspector. All are computed from our run; only the three headline KPI tiles on tab 1 are the paper's static
reference numbers.

**Q32. The metric curve dips to near zero at blocks 7 and 15 — bug?**
No — that's the real content-baseline score for those hard blocks (matches slide 17's note). The curve is
the DistilBERT-feature clustering baseline per block, so the dips are genuine.

## H. Incremental / streaming aspect

**Q33. What does "incremental" mean here?**
Messages arrive over time in 22 chronological blocks (M0–M21). Rather than re-clustering the whole history,
the method processes each new block and (in the full model) maintains the graph with a "keep-latest"
strategy. Our lifecycle tab shows events emerging and decaying across blocks.

**Q34. Why "keep latest" instead of keeping all history?**
The paper's ablation (and our framing) shows keeping all old nodes dilutes the signal for new events and
costs more memory. Only the most recent block is needed to detect what's happening now.

## I. Honesty / limitations (examiners love these)

**Q35. What are the main limitations?**
One platform, one 28-day window; test-mode caps (100 msgs/block); 503 pre-labelled events (not open-set);
full GNN metrics pending the pinned environment; Louvain reported as a single seeded run rather than a
stability distribution.

**Q36. Did you collect your own data?**
No — we deliberately reuse the QSGNN Twitter benchmark so results are comparable to all prior methods. That's
standard, defensible practice for a reproduction/extension case study.

**Q37. What would you do with more time?**
Run the full DistilBERT-GNN in the pinned env to put its real NMI/AMI/ARI beside our baselines; lift the
per-block cap; add Louvain stability; and build a per-window structure-vs-content *router* to exploit the
complementarity we found.

## J. Reproducing it (if asked "how do I run this?")
```bash
# 1. features + graphs + Gephi export, one command (needs the raw dataset in place):
python tools/run_gephi_pipeline.py --all-blocks --out-dir real_analysis_outputs
# 2. the structure-vs-content analysis + comparison chart:
python tools/analyze_real.py --data-path DistilBERTGNN/incremental_test_100messagesperday --out real_analysis_outputs
# 3. render a block in Gephi (Gephi desktop + AI Server on :8081 must be running):
python tools/gephi_render.py --gexf real_analysis_outputs/real_block5.gexf --out-dir real_analysis_outputs/gephi --columns ground_truth_event louvain_community modularity_class
# 4. the live dashboard:
python tools/run_dashboard.py          # http://localhost:8501
```
