import os, appdirs, sys, traceback, csv
from tabulate import tabulate

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
    
