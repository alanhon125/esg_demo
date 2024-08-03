import json
import sys
import itertools
import math

if __name__== "__main__":
    input_fname = sys.argv[1]
    output_prefix = sys.argv[2]
    
    partition = 5

    with open(input_fname, "r") as f:
        raw = json.load(f)
    
    chunk_size = math.ceil(len(raw) / partition)
    cnt = 1
    for i in range(0, len(raw), chunk_size):
        partitioned = list(itertools.islice(raw, i, i + chunk_size))
        with open(output_prefix + f"_{cnt}.json", "w") as json_file:
            json.dump(partitioned, json_file, indent=4, separators=(',',': '))
        cnt = cnt + 1
