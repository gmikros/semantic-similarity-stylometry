# Reproducibility Audit

- Paper tables found: 5

## Paper Table 1
- Columns: ['scenario', 'intra_overlap', 'inter_overlap', 'unique_texts', 'vocab_size', 'mean_words_/_doc', 'mfw_used']
- Best script match: table2_delta_separation (column-overlap=1)
- Status: UNRESOLVED (low column overlap)

## Paper Table 2
- Columns: ['scenario', 'within_author_δ', 'between_author_δ', 'separation_ratio', 'role_in_the_design']
- Best script match: table2_delta_separation (column-overlap=1)
- Status: UNRESOLVED (low column overlap)

## Paper Table 3
- Columns: ['scenario', 'representation', '1nn_loo_acc.', 'ari_k=3', 'silhouette']
- Best script match: table3_method_performance_plus_silhouette (column-overlap=2)
- Status: UNRESOLVED (low column overlap)

## Paper Table 4
- Columns: ['sc.', 'contrast', 'acc._a', 'acc._b', 'δ_acc._[95%_ci]', 'p_perm.', 'p_exact']
- Best script match: table2_delta_separation (column-overlap=0)
- Status: UNRESOLVED (low column overlap)

## Paper Table 5
- Columns: ['scenario', 'delta_~_openai', 'delta_~_spacy', 'openai_~_spacy']
- Best script match: table2_delta_separation (column-overlap=1)
- Status: UNRESOLVED (low column overlap)

## Figure Audit (embedded vs generated)
- Embedded images in paper: 6
- Generated figure files: 5
- word/media/image1.png -> fig4_delta_pair_type_decomposition.png | hamming=100 | UNSURE
- word/media/image2.png -> fig4_delta_pair_type_decomposition.png | hamming=93 | UNSURE
- word/media/image3.png -> fig1_design_targets_and_realized_geometry.png | hamming=108 | UNSURE
- word/media/image4.png -> fig4_delta_pair_type_decomposition.png | hamming=40 | LIKELY MATCH
- word/media/image5.png -> fig5_knn_accuracy_curves_s3_s5.png | hamming=55 | POSSIBLE
- word/media/image6.png -> fig6_matrix_correlations_mantel.png | hamming=48 | POSSIBLE