#!/usr/bin/env python
from tabulate import tabulate
from ghostbox.util import loadLayersFile, getLayersFile

def main():
    try:
        ds = loadLayersFile()
    except:
        print("Couldn't load the layers file '" + getLayersFile() + "'.")
        return

    print(tabulate(ds, headers="keys"))
    
        
if __name__ == "__main__":
    main()
