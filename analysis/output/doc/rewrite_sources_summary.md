## Presentation Notes

### Slide 1: Investigating the Impact of Semantic Similarity on Stylometric Attribution Using Controlled Artificial Texts and Delta Distances
- George Mikros Radek Čech, Petra Mutlová

### Slide 2: Where this paper started…

### Slide 3: An interesting Medieval Latin Corpus
- Polemical debates about the reform of the Church Authors from two opposing parties Latin texts written by authors with various linguistic background Pro-reform party: Jan Hus (incl. false attributions) Jacobellus of Mies Nicholas of Dresden Peter of Dresden Anti-reform party: Andrew of Brod Peter of Pulkau Nicholas of Dinkelsbühl

### Slide 4: [No title]

### Slide 5: [No title]

### Slide 6: Semantic Similarity as destabilizing stylometric factor

### Slide 7: Research Problem & Background

### Slide 8: Research Questions & Objectives
- Primary Research Questions How does semantic similarity affect the reliability of Delta distances in distinguishing between authors? How do Word Embedding features compare to Delta distances in authorship attribution under varying levels of semantic overlap? Can we identify patterns in the relationship between Delta distances and Word Embeddings across different semantic contexts? What methodological recommendations can be derived for stylometric analysis in semantically similar texts? Research Objectives Create controlled artificial text corpora with varying levels of semantic overlap Compare the performance of Delta distances and Word Embeddings across scenarios Identify non-evident patterns and relationships between methods Develop methodological recommendations for stylometric analysis

### Slide 9: Methodology Overview

### Slide 10: Experimental Design

### Slide 11: [No title]

### Slide 12: Stylo Analyses
- Scenario 1
- Scenario 3
- Scenario 5

### Slide 13: Delta Distance Analysis
- Implications Delta distances excel with high intra-author similarity Performance degrades with increasing inter-author similarity Separation ratio predicts attribution difficulty Critical Threshold Ratio near 1.0 indicates unreliable attribution
- Separation Ratio Label pairs: Split the 1,770 text-pairs into 570 within-author and 1,200 between-author comparisons. Average distances: Find the mean distance inside each group (mean_intra, mean_inter). Ratio = separation: Divide mean_inter by mean_intra; the larger the ratio, the wider the “gap” between authors’ writing spaces and the easier attribution becomes. Key Findings Scenario 1: Clear separation between authors (ratio 3.22) Scenario 2: Moderate separation (ratio 1.21) Scenario 5: Poor separation (ratio 1.12)

### Slide 14: Word Embeddings Pipeline & Research Integration
- Model & Setup SpaCy en_core_web_lg (300-d vectors trained on Common Crawl + Wikipedia) nlp.max_length = 12 000 000, UTF-8, batch processing of all .txt files Document Embeddings 300-d vector per token → mean-pool over all tokens (content + function + punctuation) Produces a 60 × 300 feature matrix for each scenario Export for ML One CSV per scenario: filename column + 300 dimensions; no header/index → plug-and-play with scikit-learn Methodological Rationale All-token mean = richer stylistic + semantic signal than content-words-only Identical pipeline across scenarios ensures fair, reproducible comparisons Research Contribution Embeddings add semantic awareness that complements traditional Δ-distance features, strengthening attribution analyses

### Slide 15: Word Embedding Analysis
- PCA Effectiveness Results • Scenario 1: 78% variance in 2 PCs, good author separation • Scenario 3: 100% variance in 2 PCs, perfect author separation • Scenario 5: 29% variance in 2 PCs, poor author separation Key Insights • 7.5× efficiency difference between best (Scenario 3) and worst (Scenario 5) cases • PCA effectiveness predicts classification performance across all scenarios • Semantic context determines dimensionality reduction quality

### Slide 16: Clustering Quality of the WE
- What the scores mean S1 & S3 ⇒ perfect author separation; tight, well-spaced clusters S5 ⇒ clusters overlap; performance worse than random (negative ARI) Performance pattern Binary behaviour: embeddings either excel (S1, S3) or collapse (S5) Optimal clustering at moderate semantic overlap (S3) Complete failure when inter-author texts are highly similar (S5) Research implications Clustering quality tightly mirrors attribution accuracy (1.0 vs 28 %) Word-embedding methods are highly context-sensitive—useful warning flag Silhouette & ARI give fast, reliable previews of method success Key takeaway Embeddings deliver perfect authorship clustering unless semantic overlap crosses a critical threshold—then performance drops off a cliff.

### Slide 17: Feature Space Comparison
- Why? Question: "Do both methods rank text similarities the same way?" Answer: Matrix correlation directly measures agreement in similarity rankings Research Significance: High correlation = Methods agree on text relationships Low correlation = Methods capture different stylistic aspects Decreasing trend = Methods become less aligned as semantic overlap increases Process We measured how similarly the two methods rank all pairwise text relationships The systematic decrease (0.907 → 0.445 → 0.670) validates our complementary methods hypothesis Scenario 3 shows maximum complementarity with the lowest correlation All correlations are statistically significant (p < 0.001) with 1,770 text pairs per scenario

### Slide 18: Comparing Delta distances and Word Embeddings
- One decision rule for both 5-Nearest-Neighbour (5-NN) classifier on distance matrices only Eliminates algorithm bias—lets the distance do the work Prepare distances Δ: ready-made distance matrix Embeddings: cosine distance = 1 – cosine similarity on 300-d vectors Evaluation workflow Leave-one-out 5-NN accuracy (k = 5). Global cluster quality via silhouette on full distance matrix. Results 5-NN Accuracy – Δ stays high (1.00 → 0.92 → 0.93); embeddings collapse under heavy obfuscation (1.00 → 1.00 → 0.07). Silhouette – embeddings lead early (0.92 → 1.00) but turn negative in Scenario 5 (–0.02); Δ erodes steadily yet remains weakly positive (0.67 → 0.13 → 0.10). Interpretation Δ preserves local author pockets even in noisy text ⇒ great for small-radius voting. Embeddings preserve global geometry until extreme disguise ⇒ great for clustering, brittle locally under heavy noise.

### Slide 19: Combining Delta distances with Word Embeddings in Scenario 5
- Methodology Input: 300-d spaCy embeddings & stylometric Δ-distance matrix (60×60) Normalize each matrix to [0,1] so scales are comparable. Blend distances: d_combo = α·Δ + (1–α)·Embedding for α∈[0…1] Evaluate with k-NN (k=5) leave-one-out author classification. Sweep α in 0.1 steps → record accuracy for each blend. Results Delta captures robust global author signals; embeddings add fine-grained semantics. A weighted hybrid can exploit both strengths and cancel weaknesses. Under tougher 5-NN voting, pure embeddings collapsed (≈7 %); pure Delta scored 93 %. Blending 30 % embeddings with 70 % Delta distances reached perfect 100 % accuracy. Guidance: tune α≈0.7 for high-k robustness; reconsider for other corpora. Recommendation Use Delta for robust majority voting, embeddings for clean global structure, and blend the two when you need both.

### Slide 20: Broader Implications

### Slide 21: Conclusions

### Slide 22: Thank you!

## Paper Draft Notes

### Introduction
- 1. Introduction
- Authorship attribution rests on the assumption that texts preserve measurable features of individual linguistic behaviour. In stylometric research, these features are typically operationalised through quantitative properties such as the distribution of linguistic units, word or sentence length and vocabulary richness. However, the extent to which such features are detectable depends on a number of textual and contextual factors. One such factor, which became central to the present study, is the degree of similarity between the texts being compared.
- The motivation for this research emerged from an attempt to apply stylometric methods to a corpus of Latin texts written by authors in medieval Bohemia. In this case, the methods did not bring stable or readily interpretable results. Instead of yielding consistent authorial groupings, they proved highly sensitive to parameter settings. Even minor changes in feature selection or analytical thresholds led to substantially different attribution outcomes. This instability raised a methodological question that extends beyond the particular corpus under investigation: why do methods that are often effective in authorship attribution fail under certain textual conditions? We assume that a crucial factor could lie in the high degree of similarity between the texts under comparison.
- This assumption is grounded in the specific nature of medieval Latin textual production and in a historically different understanding of authorship, both of which may reduce the visibility of individual authorial features. Medieval Latin texts were produced and transmitted within a manuscript culture in which authorship did not function in the same way as in modern literary practice. Texts written in Latin were characterised by frequent anonymity and medieval authorship was often collective or anonymous. Moreover, medieval textual production was strongly shaped by practices of compilation and textual reuse. Authors commonly incorporated entire passages from other texts, as well as shorter formulations and argumentative patterns, without explicitly marking them as borrowed material. Therefore, medieval texts cannot always be considered products of individual authorship in the modern sense, but rather as dynamic textual formations produced through the selection, rearrangement, and adaptation of existing material. However, in the late Middle Ages (ca 1300–1500), authorial originality came to the foreground and modern individualised authorship gradually began to emerge. Another factor contributing to textual similarity is the linguistic background of the authors. Specifically, Latin was not a native language but a learned language acquired through a shared educational tradition. The teaching was based on similar grammatical manuals and rhetorical models, and it also involved the memorisation of large passages from the same texts. This shared educational background may have further contributed to the convergence of linguistic choices and stylistic patterns across authors.
- These considerations lead us to the central question of the present study: to what extent can textual similarity itself limit the applicability of stylometric methods? To address this question, we use a controlled experimental design based on artificial text corpora with systematically varied degrees of similarity. Both intra-author and inter-author overlap are manipulated in order to examine how different configurations of textual similarity influence author differentiation. The results show that increasing semantic similarity diminishes the reliability of Delta distances in distinguishing authors.
- The paper is organised as follows….

### Methods

### Results
