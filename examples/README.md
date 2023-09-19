# Notes 
1. Examples only run 1000 Langevin steps to test performance.
2. For BottleBrush and Star9ArmsGyroid, the domain sizes are not accurately equilibrated.
3. Feature size of the neural network in 'Star9ArmsGyroid' is 64.
4. A line with 'sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))' is added to use 'deep_langevin_fts.py' in the root folder.
5. Only 'Star9arms' and 'Gyroid' are using newly trained model.