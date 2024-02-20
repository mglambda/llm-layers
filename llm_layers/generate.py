#!/usr/bin/env python
import sys, os, stat, glob, argparse, appdirs, traceback, datetime, re, csv
from llm_layers.layers import *

def main():
    parser = argparse.ArgumentParser(description="ghostbox-generate-startup-scripts - Create server startup scripts for GGUF file directory.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_directory", type=str, default="~/.cache/huggingface", help="Directory where you store your GGUF files or repositories. This will be searched recursively. If you have a folder full of huggingface repositories, that is what this parameter wants.")
    parser.add_argument("output_directory", type=str, help="Directory to put generated scripts into. Existing scripts will be overriden.")
    parser.add_argument("-l", '--layers', type=int, default=1, help="Default number of layers to offload to GPU by default. You can just open the generated script afterwards and change this easily, or you can adjust the LLM_LAYERS environment variable. Will also be written to the llm_layers file.")
    parser.add_argument("--context", type=int, default=2048, help="Default context size for loaded models. You can change this via the LLM_MAX_CONTEXT_LENGTH environment variable for all scripts. Will also be written to the llm_layers file.")
    parser.add_argument("--layers_file", type=str, default=getLayersFile(), help="File to write individual model loading information to. This file will be checked by the generated scripts for layers and context to use. You can still override these settings by supplying your own command line parameters.")
    parser.add_argument("-p", "--prefix", type=str, default="run.", help="String to prepend each script's filename. Hint: Try putting the number of layers here.")
    parser.add_argument("-s", "--suffix", type=str, default=".sh", help="String to append to each resulting string.")
    parser.add_argument("-x","--llama_executable", type=str, default="", help="Path to a llama.cpp executable. Server or main usually work. You can adjust this later with the LLM_SERVER environment variable.")
    parser.add_argument("--additional_arguments", type=str, default="--parallel 1 --mlock --no-mmap >/dev/null 2>/dev/null &", help="Any additional arguments that will be passed onto the server executable.")
    args = parser.parse_args()
    args.layers_file = os.path.expanduser(args.layers_file)
    
    if args.llama_executable == "":
        printerr("warning: No executable provided. You will either have to set LLM_SERVER in the environment or regenerate the scripts with --llama_executable set. Otherwise the scripts won't work.")
        
    cmd = sys.argv[0] + f" '{args.model_directory}' '{args.output_directory}' --prefix '{args.prefix}' --suffix '{args.suffix}' --llama_executable '{args.llama_executable}' --layers {args.layers} --context {args.context} --layers_file '{args.layers_file}' --additional_arguments '{args.additional_arguments}'"

        
    mdir = os.path.expanduser(args.model_directory)
    sdir = os.path.expanduser(args.output_directory)

    if not(os.path.isdir(mdir)):
        fail("Not a directory: " + mdir)

    if not(os.path.isdir(sdir)):
        fail("Not a directory: " + sdir)

    models = getGGUFFiles(mdir)
    for model in models:
        file = model["file"]
        scriptname = makeScriptName(file, args.prefix, args.suffix)
        scriptfile = os.path.normpath(sdir + "/" + scriptname)
        if os.path.isdir(scriptfile):
            printerr("warning: Skipping " + scriptfile + ": Is a directory.")
            continue

        if os.path.isfile(scriptfile):
            printerr("Overwriting " + scriptfile)
        else:
            printerr("Generating " + scriptfile)
            
        f = open(scriptfile, "w")
        f.write(mkBashScript(model, args.layers_file, layers=args.layers, server=args.llama_executable, additional_arguments=args.additional_arguments))
        f.flush()
        st = os.stat(scriptfile)
        os.chmod(scriptfile, st.st_mode | stat.S_IEXEC)

    print("Script files have been writen to " + sdir)
    layersfile = args.layers_file
    if layersfile != "":
        doLayersFile(layersfile, models, args, cmd=cmd)

    print("""If you want, add the following lines to your ~/.bashrc to set the values for all scripts.

```
export LLM_SERVER="/PATH/TO/SERVER" # llama.cpp server executable
export LLM_MAX_CONTEXT_LENGTH=2048 # default max context
export LLM_LAYERS=12 # default number of layers offloaded to GPU, if no entry in the layers config file is present
```

You can still provide command line parameters to the script to override environment variables or entries in the layers file like this
  ./run.xxx.sh -c 4092 -ngl 24
Happy hacking!""")



def doLayersFile(layersfile, models, args, cmd=""):
    if os.path.isdir(layersfile):
        printerr("Layers file " + layersfile + " is a directory. What is this nonsense?")
    else:
        if os.path.isfile(layersfile):
            # dictionaries of model data
            data = []
            try:
                data = list(csv.DictReader(filter(lambda line: line[0] != "#", open(layersfile, "r")), delimiter="\t"))
            except:
                printerr("error reading " + layersfile + ": \n" + traceback.format_exc())
        else:
            data = []
        newData = []
        for model in models:
            file = model["file"]
            model["name"] = os.path.basename(os.path.normpath(file))
            if list(filter(lambda d: d["name"] == model["name"], data)) != []:
                # don't override anything
                continue
            # assign defaults
            model["gpu_layers"] = args.layers
            model["context"] = args.context
            newData.append(model)

    if writeLayersConfig(layersfile, data + newData, cmd=cmd):
        printerr("Could not write layersfile " + layersfile)
    else:
        print("Wrote layers to " + layersfile)
        print("You can edit the layers file to adjust the context size and number of layers offloaded to the GPU on an individual, per-model basis.")
    
def makeScriptName(modelfile, prefix="", suffix=""):
    return prefix + os.path.basename(modelfile) + suffix

def getGGUFFiles(mdir, extensions=["gguf"]):
    """Recursively walks through directories collecting gguf models. Returns a list of dictionaries with key "name" being the gguf model filename."""
    models = []
    seen = set()
    files = glob.glob(mdir + "/*")
    # filter out mmrpoj files. This is a heuristic, but it usually works
    mmproj = ""
    candidates = list(filter(lambda w: "mmproj" in w.lower() and os.path.isfile(w) and w.lower().endswith(".gguf"), files))
    if candidates != []:
        # just pick the first one
        mmproj = candidates[0]
        for c in candidates:
            seen.add(c)

    # try to guess prompt format
    prompt_format = ""
    for file in files:
        if os.path.basename(file).lower() == "readme.md":
            prompt_format = guessPromptFormat(open(file, "r").read())
            # FIXME: new, we also want to figure out model type. No idea how yet.
        model_type = "default"
            

    # now go through the remaining ones
    for file in files:
        if os.path.isfile(file):
            if file.lower().endswith(".gguf") and not(file in seen):
                d = { "file" : file, "mmproj" : mmproj, "prompt_format" : prompt_format, "type" : model_type}
                seen.add(file)
                models.append(d)
        else:
            models = models + getGGUFFiles(file)
    return models
                

    


def writeLayersConfig(layerfile, data, cmd=""):
    header = "# Generated on " + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + " with\n# " + cmd + "\n# Listed values are what will be used for a particular model by the backend, not the maximum model capability. Regenrating this file will keep existing settings, though your comments will be lost"
    header += "\n"
    fields = "name gpu_layers context prompt_format type".split(" ")
    # csv is such a joy, it will throw if the dict has more keys than defined fields
    goodData = []
    for d in data:
        goodD = {}
        for key in fields:
            goodD[key] = d[key]
        goodData.append(goodD)
        
    try:
        f = open(layerfile, "w")
        f.write(header)
        writer = csv.DictWriter(f, fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(sorted(goodData, key=lambda d: d["name"]))
        f.close()
    except:
        printerr("Caught exception\n" + traceback.format_exc())
        return True
    return False


def mkBashScript(modelData, layersfile, layers=1, server="server", additional_arguments=""):
    modelpath = modelData["file"]
    # watch out bash script begins here
    w = """#!/bin/bash
MODEL=XXX_THE_MODEL_XXX
CONFIGLAYERS=$( cat 'XXX_THE_LAYERSFILE_XXX' | grep -P "XXX_THE_MODELNAME_XXX\\t" | awk {'print $2'} )
CONFIGCONTEXT=$( cat 'XXX_THE_LAYERSFILE_XXX' | grep -P "XXX_THE_MODELNAME_XXX\\t" | awk {'print $3'} )
MMPROJ_FILE=XXX_THE_MMPROJ_FILE_XXX
    
if [ -n "$CONFIGLAYERS" ]
then
    echo "Found layers in layers file $XXX_THE_LAYERSFILE_XXX"
    LAYERS=$CONFIGLAYERS
elif [ -n "$LLM_LAYERS" ]
then
    echo "Found LLM_LAYERS environment variable."
    LAYERS=$LLM_LAYERS
else
    echo "Could not find LLM_LAYERS environment variable."
    LAYERS=XXX_THE_LAYERS_XXX
fi

echo "Setting layers to $LAYERS"

if [ -n "$CONFIGCONTEXT" ]
then
    echo "Found context length in layers file $XXX_THE_LAYERSFILE_XXX"
    MAX_CONTEXT_LENGTH=$CONFIGCONTEXT
elif [ -n "$LLM_MAX_CONTEXT_LENGTH" ]
then
	MAX_CONTEXT_LENGTH="$LLM_MAX_CONTEXT_LENGTH"
	echo "Found LLM_MAX_CONTEXT_LENGTH in environment"
else
    echo "Could not find LLM_MAX_CONTEXT_LENGTH in environment. Defaulting."	
    MAX_CONTEXT_LENGTH=2048	   
fi

echo "Setting Context to $MAX_CONTEXT_LENGTH"

if [ -n "$MMPROJ_FILE" ]
then
    MMPROJ_ARG="--mmproj $MMPROJ_FILE"
    echo "Passing $MMPROJ_ARG to server."
fi
    
if [ -n "$LLM_SERVER" ]
then
	echo "Found LLM_SERVER in environment."
	SERVER="$LLM_SERVER"
else
	echo "Could not find LLM_SERVER in environment. Defaulting."
	SERVER="XXX_THE_SERVER_XXX"
fi
echo "Setting SERVER to $SERVER"
echo "End of run script. Starting server."
PATH=./:$PATH
$SERVER -c $MAX_CONTEXT_LENGTH -m $MODEL -ngl $LAYERS $MMPROJ_ARGS XXX_THE_ADDITIONALARGS_XXX $@
"""
    return w.replace("XXX_THE_MODEL_XXX", os.path.expanduser(modelpath)).replace("XXX_THE_LAYERS_XXX", str(layers)).replace("XXX_THE_SERVER_XXX", os.path.expanduser(server)).replace("XXX_THE_MODELNAME_XXX", re.escape(os.path.basename(os.path.expanduser(modelpath)))).replace("XXX_THE_LAYERSFILE_XXX", layersfile).replace("XXX_THE_ADDITIONALARGS_XXX", additional_arguments).replace("XXX_THE_MMPROJ_FILE_XXX", modelData["mmproj"])

def fail(w):
    printerr(w)
    sys.exit()


def printerr(w):
    print(w, file=sys.stderr)
    




def guessPromptFormat(w, formats="chat-ml alpaca user-assistant-newlines mistral".split(" ")):
    ws = w.split("\n")
    needles = ["prompt template", "prompt format", "prompt_format"]
    for o_line in ws:
        line = o_line.lower()
        for needle in needles:
            if needle in line:
                for o_format in formats:
                    format = o_format.lower()
                    if format in line:
                        return format

    # additional heuristics if we haven't found it above
    wl = w.lower()
    if "<|im_start|>" in wl:
        return "chat-ml"

    if "### instruction:" in wl:
        return "alpaca"

    if "### user:" in wl:
        return "user-assistant-newlines"

    if "[INST]" in wl and "[/INST]" in wl:
        return "mistral"
    
    # no idea
    return ""
    
if __name__ == "__main__":
    main()

