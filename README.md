# Natural_Computing_assignments

## How to run the algorithms
First clone the repository:
```
git clone https://github.com/thijschanka/Natural_Computing_assignments.git
```
Next move into the correct folder:
```
cd Natural_Computing_assignments/project/
```
Then install the required packages:
```
pip install -r .\requirements.txt
```
Finally run the inference script:
```
python ./inference.py -e evaluation.json -k 5x5 -a bt -s 42
```

## Settings
The inference script has the following settings:
| Full Command  | Short Command | Description   |
| ------------- | ------------- | ------------- |
| --eval_json  | -e  | The relative location of the evaluation file to run the script with |
| --accepted_key  | -k  | the keys/puzzle groups from the evaluation file to run, in none given it will run all |
| --seed | -s | Wheter to seed the code and if given with what value |
| --algorithm | -a | which algorithm to use,pso=particle swarm optimisation, ea=evolutionary algorithm, bt=backtracking |
| --particles | -p | how many particles to use or the population size if ea |
| --iterations | -i | how many iterations to run |

## Example results
If we would run the following 3 lines of code:
```
./inference.py -e evaluation.json -k 5x5 -a bt -p 200 -s 42
./inference.py -e evaluation.json -k 5x5 -a ea -p 200 -s 42
./inference.py -e evaluation.json -k 5x5 -a pso -s 42 -p 65536
```
We will get the results files:
- ./Results/5x5_bt_results.csv
- ./Results/5x5_ea_results.csv
- ./Results/5x5_pso_results.csv

