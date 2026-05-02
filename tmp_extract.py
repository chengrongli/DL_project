import traceback

try:
    from data.repo_extractor import extract_pairs_from_repo
except Exception:
    with open("extract_result.txt", "w") as f:
        f.write("import_error\n")
        f.write(traceback.format_exc())
    raise SystemExit(1)

try:
    result = extract_pairs_from_repo(
        repo_root="data/raw_lpc_repo",
        out_dir="data/pairs/train",
        index_path="data/index_train.csv",
        include_patterns=["*walk.png"],
    )
    output = [
        f"total={result.total_candidates}",
        f"success={result.successful}",
        f"failed={len(result.failed_paths)}",
    ]
except Exception:
    output = ["error", traceback.format_exc()]

with open("extract_result.txt", "w") as f:
    f.write("\n".join(output))
