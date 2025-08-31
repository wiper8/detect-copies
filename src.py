import os
import re
import itertools
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def list_repos(path, regex):
    repos = []
    
    for root, dirnames, _ in os.walk(path):
        for dir in dirnames:
            if bool(re.match(regex, dir, re.IGNORECASE)):
                repos.append(os.path.relpath(os.path.join(root, dir), path))
    return repos

def list_files(repo_path):
    files = []
    exceptions = [".git"]
    for root, dirnames, filenames in os.walk(repo_path):
        dirnames[:] = [d for d in dirnames if d not in exceptions]
        for f in filenames:
            if not f in exceptions:
                files.append(os.path.relpath(os.path.join(root, f), repo_path))
    return files

def filter_code(files):
    return [f for f in files if not bool(re.match(r"(.*csv|.*rds|.*validate.*|.*xlsx|.*LICENSE|tests.*|.*pdf)", f, re.IGNORECASE))]

def file_similarity(path1, path2):
    try:
        with open(path1, "r", encoding="utf-8", errors="ignore") as f1, \
             open(path2, "r", encoding="utf-8", errors="ignore") as f2:
            text1, text2 = f1.read(), f2.read()
    except Exception:
        return 0.0  # binaire ou problème d’encodage

    if not text1.strip() and not text2.strip():
        return 1.0

    # TF-IDF + cosinus
    vectorizer = TfidfVectorizer().fit([text1, text2])
    tfidf = vectorizer.transform([text1, text2])
    cos = cosine_similarity(tfidf[0:1], tfidf[1:2])[0,0]

    return cos

def repo_similarity(repo1, repo2):
    files1 = list_files(repo1)
    files2 = list_files(repo2)
    files1 = filter_code(files1)
    files2 = filter_code(files2)
    files1 = set(files1)
    files2 = set(files2)
    
    common = files1 & files2    
    all = files1 | files2
    if len(all) == 0:
        jaccard_repos = 0
    else:
        jaccard_repos = len(common) / len(all)

    sims = []
    for f in common:
        sims.append(file_similarity(os.path.join(repo1, f), os.path.join(repo2, f)))
    
    score = sum(sims) / max(len(common), 1)
    score = max(score, 0.0)
    # print(f"Jaccard: {jaccard_repos * 100:.1f}%, Cosine: {score * 100:.1f}%")
    return jaccard_repos, score

def every_repo_similarity(repos, path):
    n = len(repos)
    if n < 2:
        raise Exception(f"Only {n} repos found")
    
    full_paths = [os.path.join(path, repo) for repo in repos]
    
    # Compute pairwise similarities
    results = []
    for repo1, repo2 in itertools.combinations(full_paths, 2):
        sim_repo, sim_files = repo_similarity(repo1, repo2)
        results.append([repo1, repo2, sim_repo, sim_files])

    results = pd.DataFrame(results, columns = ["repo1", "repo2", "sim_repo", "sim_files"])
    return results.sort_values(by="sim_files")