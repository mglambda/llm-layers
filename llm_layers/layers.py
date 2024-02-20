def getLayersFile():
    return appdirs.user_config_dir() + "/llm_layers"

def loadLayersFile():
    """Returns a list of dictionaries, one for each row in the layers file."""
    f = open(getLayersFile(), "r")
    return list(csv.DictReader(filter(lambda row: row[0] != "#", f), delimiter="\t"))
