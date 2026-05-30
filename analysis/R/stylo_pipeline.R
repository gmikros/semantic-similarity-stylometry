#!/usr/bin/env Rscript
# Classic stylometry with the `stylo` package: 500-MFW Cosine Delta ("wurzburg").
#
# Reproduces, per scenario:
#   * <scn>_table_with_frequencies.txt   (relative MFW frequencies)
#   * <scn>_distance_table_<k>mfw_0c.csv  (Cosine-Delta distance matrix)
#   * <scn>_wurzburg_EDGES.csv / _NODES.csv (bootstrap consensus network for Gephi)
#
# Usage:  Rscript R/stylo_pipeline.R <scenario> [mfw]
#   e.g.  Rscript R/stylo_pipeline.R 3 500
#         Rscript R/stylo_pipeline.R 4 393

suppressMessages({
  if (!requireNamespace("stylo", quietly = TRUE)) {
    stop("Package 'stylo' is required. Install with install.packages('stylo').")
  }
  library(stylo)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Provide a scenario, e.g. Rscript R/stylo_pipeline.R 3 500")
scenario <- args[[1]]
mfw <- if (length(args) >= 2) as.integer(args[[2]]) else 500L

# --- resolve paths relative to this script ---------------------------------
this_file <- sub("^--file=", "",
                 grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
analysis_dir <- normalizePath(file.path(dirname(this_file), ".."))
project_dir <- normalizePath(file.path(analysis_dir, ".."))

split_scenarios <- c("1", "2")
corpus_dir <- file.path(project_dir, "Corpus", paste("Scenario", scenario))
if (scenario %in% split_scenarios) corpus_dir <- file.path(corpus_dir, "output")
if (!dir.exists(corpus_dir)) stop(paste("Corpus dir not found:", corpus_dir))

out_dir <- file.path(analysis_dir, "output")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

prefix <- paste0("Scenario_", scenario)

# stylo names its native artifacts (EDGES/NODES, etc.) by MFW only, which would
# collide across scenarios. Run stylo from a per-scenario subfolder; the explicit
# scenario-prefixed tables below are still written to the main output dir.
scn_dir <- file.path(out_dir, prefix)
dir.create(scn_dir, showWarnings = FALSE, recursive = TRUE)
old_wd <- getwd()
setwd(scn_dir)
on.exit(setwd(old_wd), add = TRUE)

cat(sprintf("Scenario %s | %d MFW | Cosine Delta | corpus: %s\n",
            scenario, mfw, corpus_dir))

# --- single-point Cosine Delta analysis ------------------------------------
res <- stylo(
  gui              = FALSE,
  corpus.dir       = corpus_dir,
  corpus.lang      = "English",
  analyzed.features = "w",
  ngram.size       = 1,
  mfw.min          = mfw,
  mfw.max          = mfw,
  mfw.incr         = 0,
  culling.min      = 0,
  culling.max      = 0,
  distance.measure = "wurzburg",   # Cosine Delta
  analysis.type    = "CA",
  write.png.file   = FALSE,
  display.on.screen = FALSE
)

# Frequency table (relative frequencies of the MFW set)
freqs <- res$table.with.all.freqs
write.table(t(freqs[, seq_len(min(mfw, ncol(freqs)))]),
            file = file.path(out_dir, paste0(prefix, "_table_with_frequencies.txt")),
            quote = TRUE, col.names = NA)

# Distance table
dist_mat <- as.matrix(res$distance.table)
write.table(dist_mat,
            file = file.path(out_dir, paste0(prefix, "_distance_table_", mfw, "mfw_0c.csv")),
            quote = TRUE, col.names = NA)
# comma-separated wide format (matches Wide_Format_Distance_Matrix_for_Scenario_N.csv)
wide <- data.frame(Author = rownames(dist_mat), dist_mat, check.names = FALSE)
write.csv(wide, file = file.path(out_dir, paste0("Wide_Format_Distance_Matrix_for_", prefix, ".csv")),
          row.names = FALSE)

cat("Wrote frequency + distance tables to", out_dir, "\n")

# --- bootstrap consensus network (Gephi EDGES/NODES) -----------------------
# `stylo.network` averages several MFW bands into a document co-occurrence network
# and writes <...>_EDGES.csv / _NODES.csv for Gephi.
mfw_min <- max(100L, mfw %/% 5L)
tryCatch({
  stylo.network(
    mfw.min          = mfw_min,
    mfw.max          = mfw,
    mfw.incr         = max(20L, (mfw - mfw_min) %/% 10L),
    corpus.dir       = corpus_dir,
    corpus.lang      = "English",
    analyzed.features = "w",
    ngram.size       = 1,
    culling.min      = 0,
    culling.max      = 0,
    distance.measure = "wurzburg"
  )
  cat("Wrote bootstrap consensus network EDGES/NODES to", out_dir, "\n")
}, error = function(e) {
  cat("Network step skipped:", conditionMessage(e), "\n")
})
