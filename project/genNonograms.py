import subprocess
from os import listdir
from pathlib import Path
import json

def main(pbn = None, local_puzzle = True, out_dir="Puzzles/generated/local"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if local_puzzle:
        example_folder = subprocess.run(["pynogram", "--show-examples-folder"], stdout=subprocess.PIPE)
        example_folder = example_folder.stdout.decode('utf-8').replace('\r', '').replace('\n', '')
        for f in listdir(example_folder):
            print("Process puzzle:", f)
            puzzle = subprocess.run(["pynogram", "-b", f, "--max-depth", "1", "--timeout", "1", "--draw-final", "--max-solutions", "1"], stdout=subprocess.PIPE)
            puzzle = puzzle.stdout.decode('utf-8')
            build_json(out_dir / f, puzzle)

    if pbn != None:
        if isinstance(pbn, list):
            for f in pbn:
                print("Process puzzle:", f)
                puzzle = subprocess.run(["pynogram", f"--pbn={str(f)}", "--max-depth", "1", "--max-solutions", "1", "--draw-final", "--timeout", "1",], stdout=subprocess.PIPE)
                build_json(out_dir / str(pbn), puzzle.stdout.decode('utf-8'))
        else:
            print("Process puzzle:", pbn)
            puzzle = subprocess.run(["pynogram", f"--pbn={str(pbn)}", "--max-depth", "1", "--max-solutions", "1", "--draw-final", "--timeout", "1",], stdout=subprocess.PIPE)
            build_json(out_dir / str(pbn), puzzle.stdout.decode('utf-8'))
        
def build_json(out_dir, puzzle_str):
    rows = []
    columns = []
    puzzle_str = puzzle_str.replace('\r', '\n').replace('\n\n', '\n').split('\n')
    
    for idx, col in enumerate(puzzle_str[0]):
        if idx % 2 == 0 and col != '#':
            columns.append([])
    
    for idx_r, row in enumerate(puzzle_str):
        if row != '':
            print(row)
            if row[0] == '#': 
                is_col = True
            else:
                is_col = False
                rows.append([])
            
            counter = 0
            for idx_c, col in enumerate(row):
                if idx_c % 2 == 0 and col.isnumeric():
                    number = col
                    if idx_c < len(row)-1 and row[idx_c+1].isnumeric():
                        number += row[idx_c+1]
                    
                    if is_col:
                        columns[counter].append(int(number))
                    else:
                        rows[-1].append(int(number))
                
                if idx_c % 2 == 0 and col != '#':
                    counter += 1
    
    puzzle_json = {"width":len(columns), "height":len(rows), "rows":rows, "columns":columns}
    
    with open(out_dir.with_suffix('.json'), 'w') as f:
        json.dump(puzzle_json, f)
        
if __name__ == "__main__":
    #main(pbn = [3, 4], local_puzzle = False, out_dir="Puzzles/generated/pbn")
    main()
