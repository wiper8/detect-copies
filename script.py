from src import *

path = "../../../Desktop"
repos = list_repos(path, r"[0-9]{9}_[Tt]ravail[Ll]ong")

print(every_repo_similarity(repos, path))
