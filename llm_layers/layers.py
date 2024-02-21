import os, appdirs, sys, traceback, csv, torch
from tabulate import tabulate
from huggingface_hub import list_models, get_paths_info, repo_info, snapshot_download
from huggingface_hub.utils._errors import GatedRepoError

def getLayersFile():
    return appdirs.user_config_dir() + "/llm_layers"

def loadLayersFile(file=getLayersFile()):
    """Returns a list of dictionaries, one for each row in the layers file."""
    f = open(file, "r")
    return list(csv.DictReader(filter(lambda row: row[0] != "#", f), delimiter="\t"))


def show(file=getLayersFile()):
    try:
        ds = loadLayersFile(file)
    except:
        print(traceback.format_exc())
        print("error: Couldn't load the layers file " + file, file=sys.stderr)
        return ""
    return tabulate(ds, headers="keys")
    
def reposFromFile(filename):
    """Given a filename (e.g. a gguf file), returns a list of huggingface repository ids with that file in it. The list may be empty."""
    # hf will not allow search in repos directly, so we first have to find the repo with some trickery and guesswork
    # of course, we don't know filename, it might be totally ok as is
    ws = filename.lower().split(".")
    # add the dots back in
    for i in range(0, len(ws)):
        if i == 0 or i == len(ws)-1:
            continue
        ws[i] = "." + ws[i]
        
    best = []
    query = ""
    for w in ws:
        query += w
        xs = list(list_models(search=query))
        if xs == []:
            # a longer string won't get any more results
            break
        if best == [] or (len(xs) < len(best)):
            # the more specific result is better, and fewer results is arguably more specific
            best = xs

    # get and check files
    winners = []
    for modelinfo in best:
        try:
            for fileinfo in get_paths_info(modelinfo.id, filename):
                # want exact match here
                if fileinfo.path == filename:
                    winners.append(modelinfo.id)
        except GatedRepoError:
            continue
        except HTTPError:
            continue
                
    return winners
            
def pickWinner(winners, filename):
    """Given a list of repo ids and a file they contain, pick the best one to acquire later"""
    # This is a very simple algorithm, it's either theBloke or whoever has more downloads+likes
    # hf has the brilliant strategy of throwing an exception when querying a gated repo, so no list comprehension here.
    repo_data = []
    for repo_id in winners:
        try:
            repo = repo_info(repo_id)
        except GatedRepoError: # this is why we can't have nice things
            continue
        except HTTPError:
            # this will mean we get no repo data if inet is down, which is fine
            continue
        
        repo_data.append((repo_id, repo))                                

    winner = { "name" : "", "score" : 0}
    for (repo_id, repo) in repo_data:
        if repo.author == "TheBloke":
            # clear winner, return early
            return repo_id

        score = repo.downloads + repo.likes
        if winner["score"] < score:
            # new winner
            winner["name"] = repo_id
            winner["score"] = score

    return winner["name"]

    
    

def get_hf_repo_for_file(filename):
    """Given the name of a file in some huggingface repository, returns the repository id of a repository containing that file. If multiple repositories contain the file, the result is determined by ranking the candidate repositories according to various factors. If theBloke is among candidates, the result is always theBloke. Returns empty string if no repository contains the file."""
    return pickWinner(reposFromFile(filename), filename)

def load_layers_file(file=getLayersFile()):
    """Returns a list of dictionaries, one for each row in the layers file."""
    return loadLayersFile(file)

def download_for_layers_file(filename):
    """Takes filename of a layer file and downloads all listed model files using the huggingface api."""
    try:
        ds = load_layers_file(filename)
    except FileNotFoundError:
        print("error: File not found " + filename, file=sys.stderr)
        return

    for d in ds:
        repo = get_hf_repo_for_file(d["name"])
        if repo:
            print("Getting " + repo + " ...")
            snapshot_download(repo,
                              allow_patterns=[d["name"], "*README*", "*readme*", "*LICENSE*", "*license*", "*.txt", "*.md", "*.json"])
            
                                    

def get_total_vram_mb():
    has_cuda = torch.cuda.is_available()
    if not(has_cuda):
        return 0
    n = torch.cuda.device_count()
    vram = 0
    for i in range(0, n):
        vram += torch.cuda.get_device_properties(i).total_memory / 1e6

    return round(vram)
    

            
