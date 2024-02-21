#!/usr/bin/env python
import sys, os, stat, glob, argparse, appdirs, traceback, datetime, re, csv, tempfile, random
from llm_layers import getData
from llm_layers.layers import *
from functools import *

# switch output on and off globally
print_enabled = True
printerr_enabled = True
# global scope for a temp file we want to stick around for program lifetime
temp_layersfile = None

def main():
    parser = argparse.ArgumentParser(description="ghostbox-generate-startup-scripts - Create server startup scripts for GGUF file directory.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument("--color", action=argparse.BooleanOptionalAction, default=True, help="Enable colored output.")
    parser.add_argument("--pretty", action=argparse.BooleanOptionalAction, default=True, help="Enable prettier output using tabular. Disable this alongside -d for copy/pastable output.")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True, help="Download models in layers file from huggingface. -d or --dry_run implies --no-download.")
    parser.add_argument("-g", "--generate", action=argparse.BooleanOptionalAction, default=False, help="Generate a layers file and run scripts. This will never overwrite settings in an existing layers file, but may add new entries. Existing scripts will be overwritten without mercy.")
    parser.add_argument("-d", "--dry_run", action=argparse.BooleanOptionalAction, default=True, help="Will not write anything to disk, but print a sample layers file to stdout. This is the default when run without either -d or -g. If both parameters are provided, -g will dominate.")
    parser.add_argument("--model_directory", type=str, default="~/.cache/huggingface", help="Directory where you store your GGUF files or repositories. This will be searched recursively. If you have a folder full of huggingface repositories, that is what this parameter wants.")
    parser.add_argument("--output_directory", type=str, default="~/.local/bin", help="Directory to put generated scripts into. Existing scripts will be overriden.")
    parser.add_argument("-l", '--layers', type=int, default=1, help="Default number of layers to offload to GPU by default. You can just open the generated script afterwards and change this easily, or you can adjust the LLM_LAYERS environment variable. Will also be written to the llm_layers file.")
    parser.add_argument("--context", type=int, default=2048, help="Default context size for loaded models. You can change this via the LLM_MAX_CONTEXT_LENGTH environment variable for all scripts. Will also be written to the llm_layers file.")
    parser.add_argument("--layers_file", type=str, default=getLayersFile(), help="File to write individual model loading information to. This file will be checked by the generated scripts for layers and context to use. You can still override these settings by supplying your own command line parameters. If this file already exists, it will not be overwritten, though new entries may be added to it. Also, it will be used as a --include_layers_file. The default value is platform dependent, often ~/.config/llm_layers. It is quit reasonable to leave the default and keep regenerating that file.")
    parser.add_argument("-b", "--best_for_machine", action=argparse.BooleanOptionalAction, default=False, help="Include models based on the current system hardware. The selection is highly opinionated and subject to change over time. This option is disabled by default, unless the --layers_file does not exist, in which case the program assumes it's the first time you are running it, enabling -b. You can disable this behaviour by passing --no-best_for_machine explicitly.")
    parser.add_argument("-I", '--include_layers_file', action="append", default=[], help="Additional layer files to source from. Data will be gathered and added to the resulting --layer_file. Include layer files will not be written to. If multiple layer files contain entires with the same 'name' field, the result is undefined. This option can be supplied multiple times.")
    parser.add_argument("-p", "--prefix", type=str, default="run.", help="String to prepend each script's filename. Hint: Try putting the number of layers here.")
    parser.add_argument("-s", "--suffix", type=str, default=".sh", help="String to append to each resulting string.")
    parser.add_argument("-x","--executable", type=str, default="", help="Path to a backend (e.g. llama.cpp) executable. Server or main usually work. You can adjust this later with the LLM_SERVER environment variable.")
    parser.add_argument("--additional_arguments", type=str, default="--parallel 1 --mlock --no-mmap >/dev/null 2>/dev/null &", help="Any additional arguments that will be passed onto the server executable.")
    args = parser.parse_args()
    args.layers_file = os.path.expanduser(args.layers_file)

    if not(os.path.isfile(args.layers_file)):
        if not("--no-best_for_machine" in sys.argv):
            args.best_for_machine = True
            
    if args.generate and args.dry_run:
        args.dry_run = False
        
    if args.dry_run:
        if "--download" not in sys.argv:
            args.download = False
        drymsg = "# Running with -d (--dry_run), Nothing permanent will be written to disk. Here is"
        if args.pretty:
            drymsg += "a pretty version of the potential layers file.\n"
        else:
            drymsg += "an example version of the layers file. You can copy and paste this output if you like.\n"
        drymsg += "# Run with -g to actually generate the layers file and the scripts.\n"
        global print_enabled
        print_enabled = False
        global temp_layersfile
        temp_layersfile = tempfile.NamedTemporaryFile()
        
    if args.executable == "" and args.generate:
        printerr("warning: No executable provided. You will either have to set LLM_SERVER in the environment or regenerate the scripts with --executable set. Otherwise the scripts won't work.")

        # FIXME: cmd is no longer correct
    cmd = sys.argv[0] + f" '{args.model_directory}' '{args.output_directory}' --prefix '{args.prefix}' --suffix '{args.suffix}' --executable '{args.executable}' --layers {args.layers} --context {args.context} --layers_file '{args.layers_file}' --additional_arguments '{args.additional_arguments}'"


    # ok looks like we will generate a layers file
    mdir = os.path.expanduser(args.model_directory)
    sdir = os.path.expanduser(args.output_directory)

    if not(os.path.isdir(mdir)):
        fail("Not a directory: " + mdir)

    models = getGGUFFiles(mdir, args)
    # recommendations
    include_models = []
    if args.best_for_machine:
        printout("Determining hardware...")
        vram = get_total_vram_mb()
        printout("Found " + str(vram) + "MB of maximum video ram.\nChoosing appropriate loadout...")
        choice = choiceForVRam(vram)
        if choice is not None:
            if "id" in choice:
                cool_name = choice["id"]
            else:
                cool_name = choice["file"]
            printout("Done. Chose the '" + cool_name + "' loadout for your hardware.")
            if "description" in choice.keys():
                printout("Description: " + choice["description"])
            best_models = load_layers_file(choice["file"])
            include_models += best_models
        else:
            printerr("error: No loadouts found. Failed to select a loadout for your machine.")

    # we always write this, though it might be a temp file
    if args.layers_file != "":
        # get the includes
        for includefile in args.include_layers_file:
            # note the dry=true, this is not real writing layers file yet, we just want the include list
            include_models += doLayersFile(includefile, ensureUniqueModels(models + include_models), args, dry=True)
            
        layers_models = doLayersFile(args.layers_file, ensureUniqueModels(models + include_models), args, cmd=cmd)
    else:
        layers_models = []

    # downloading from hf
    if args.download:
        print("Downloading models... (you may want to grab a coffee)")
        if args.dry_run:
            layersfile = temp_layersfile
        else:
            layersfile = args.layers_file
        download_for_layers_file(layersfile)
        # regenerate file based models
        models = getGGUFFiles(mdir, args)

        
        # writing the scripts - need to do this *after* downloading. Also note that we only write scripts for models that actually exist and have been found bygetGGUFFiles
    if args.generate:
        writeScriptFiles(models, sdir, args)
        
    printout("""If you want, add the following lines to your ~/.bashrc to set the values for all scripts.

```
export LLM_SERVER="/PATH/TO/SERVER" # llama.cpp server executable
export LLM_MAX_CONTEXT_LENGTH=2048 # default max context
export LLM_LAYERS=12 # default number of layers offloaded to GPU, if no entry in the layers config file is present
```

You can still provide command line parameters to the script to override environment variables or entries in the layers file like this
  ./run.xxx.sh -c 4092 -ngl 24
Happy hacking!""")

    # if it was a dryrun, we want to show the table here.
    if args.dry_run:
        print_enabled = True
        if args.pretty:
            printout(drymsg + "\n" + show(temp_layersfile.name))
        else:
            printout(drymsg + open(temp_layersfile.name, "r").read())
        
        
        


def doLayersFile(layersfile, models, args, cmd="", dry=False):
    """This function reads and writes a layers file, based on a filename, models, and some optional parameters. Returns model information found in layersfile.
    Parameter
    layersfile : str
    A filename with model data, that will be both read from and written to.
    models : list
    A list of dictionaries containing model data
    args : namespace
    An argparse command line argument object.
    cmd : str
    The command used to start the program, which is printed at the top of the layer file.
    Returns : list
    A list of dictionaries with model data of only the models found in the layersfile."""
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
            if "file" in model:
                file = model["file"]
                model["name"] = os.path.basename(os.path.normpath(file))
            if list(filter(lambda d: d["name"] == model["name"], data)) != []:
                # don't override anything
                continue
            newData.append(model)

    if writeLayersConfig(layersfile, data + newData, cmd=cmd, dry=(args.dry_run or dry)):
        printerr("Could not write layersfile " + layersfile)
    else:
        printout("Wrote layers to " + layersfile)
        printout("You can edit the layers file to adjust the context size and number of layers offloaded to the GPU on an individual, per-model basis.")
    return data
    
def makeScriptName(modelfile, prefix="", suffix=""):
    return prefix + os.path.basename(modelfile) + suffix

def getGGUFFiles(mdir, args, extensions=["gguf"]):
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
                d = { "file" : file, "name" : os.path.basename(file), "mmproj" : mmproj, "prompt_format" : prompt_format, "type" : model_type, "context" : args.context, "gpu_layers" : args.layers}

                seen.add(file)
                models.append(d)
        else:
            models = models + getGGUFFiles(file, args, extensions=extensions)
    return models
                

    


def writeLayersConfig(layersfile, data, cmd="", dry=False):
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
        if dry:
            global temp_layersfile
            f = open(temp_layersfile.name, "w")
        else:
            f = open(layersfile, "w")
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

def printout(w, **kwargs):
    if print_enabled:
        print(w, **kwargs)
        
    
def printerr(w):
        if printerr_enabled:
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



def writeScriptFiles(models, sdir, args):
    if not(os.path.isdir(sdir)):
        printerr("Creating directory " + sdir)
        try:
            os.makedirs(sdir)
        except:
            fail("error: Could not create directory " + sdir)
        
    
    for model in models:
        file = model["file"]
        scriptname = makeScriptName(file, args.prefix, args.suffix)
        scriptfile = os.path.normpath(sdir + "/" + scriptname)
        if os.path.isdir(scriptfile):
            printerr("warning: Skipping " + scriptfile + ": Is a directory.")
            continue

        if os.path.isfile(scriptfile):
            printout("Overwriting " + scriptfile)
        else:
            printout("Generating " + scriptfile)
            
        f = open(scriptfile, "w")
        f.write(mkBashScript(model, args.layers_file, layers=args.layers, server=args.executable, additional_arguments=args.additional_arguments))
        f.flush()
        st = os.stat(scriptfile)
        os.chmod(scriptfile, st.st_mode | stat.S_IEXEC)

    printout("Script files have been writen to " + sdir)    

def ensureUniqueModels(models):
    """Takes a list of models as dictionaries and removes entries with duplicate "name" fields. Returns the list without offending entries.
    Current behaviour when a duplicate is encountered is to keep the layerfile model when conflict is between a model from a layerfile and a model read from the filesystem (which would just get default values assigned to it). In any other case, the model further down the list wins."""
    def f(acc, model):
        if model["name"] in [other["name"] for other in acc]:
            # there are duplicates, now we need to decide who wins
            if "file" in model.keys():
                # other model in acc wins
                return acc
            else:
                # i win, remove other and add myself
                return list(filter(lambda other: other["name"] != model["name"], acc)) + [model]
        return acc + [model]
            
    return reduce(f, models, [])



def choiceForVRam(vram):
    path = getData("loadouts")
    files = glob.glob(path + "/*")
    loadouts = []
    for file in files:
        d = {}
        if os.path.isdir(file):
            continue
        if os.path.isfile(file):
            try:
                d["file"] = file
                lines = filter(lambda w: w != "" and w[0] == "#", open(file, "r").read().split("\n"))
                # we will not validate the file, that is responsibility of layerfile readers etc, we just parse some metadata in comments
                validkeys = "id vram description".split(" ")
                for line in lines:
                    ws = line.split(":")
                    if len(ws) > 1:
                        key = ws[0].replace("#", "").strip()
                        if key == "vram":
                            d[key] = megabyteIntFromVRamString(ws[1].strip())
                        elif key in validkeys:
                            d[key] = ":".join(ws[1:]).strip()
            except:
                # couldn't read or something
                printerr(traceback.format_exc() + "\nSomething... formatting something...")
                continue
        loadouts.append(d)

    #ok got all loadouts and they're valid now we find the best one
    def predicate(loadout):
        if "vram" not in loadout.keys():
            return False

        if loadout["vram"] > vram:
            return False
        return True

    def rank(loadouts):
        # no idea how to rank them yet
        random.shuffle(loadouts)
        return loadouts
    
    xs = rank(list(filter(predicate, loadouts)))
    if xs == []:
        return None
    return xs[0]


def megabyteIntFromVRamString(w):
    return int(round(parse_size(w) / 1e6))

def parse_size(w):
    """Takes a string of byte size like 200kb and returns the number of bytes as an int. Returns -1 on no parse."""
    units = {"B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12}
    # Alternative unit definitions, notably used by Windows:
    # units = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30, "TB": 2**40}
    w = w.strip()
    for i in range(0, len(w)):
        if not(w[i].isdigit()):
            break

    if i == len(w):
        number = w
        unit = "GB"
    else:
        number = w[:i].strip()
        unit = w[i:].strip().upper()
        
        try:
            n = int(number)
        except:
            return -1
    if unit not in units:
        return -1
    return int(round(float(n) * units[unit]))
                                
