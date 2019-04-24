This ZIP file contains the following folders:

· Source:
	Containing the source files for the implementation of the work.
	- main.py : with code of the main
	- preprocessing.py : with functions to preprocess the data
	- Rule.py : containing the class that defines the rules
	- PRISM.py : contains the train and predict functions for the PRISM implementation

· Output:
	Contains the console output of the tested datasets

· Data:
	Contains the tested datasets

· Documentation:
	Contains the report explaining the work and the experiments done

To execute the Python script with the different datasets, you should run 
the main file main.py from the source folder passing in the first argument 
the file path and in the second argument the class label. 

Examples for the tested datasets:

python main.py ../data/lenses.csv t
python main.py ../data/iris.data class
python main.py ../data/breast-cancer.data Class
python main.py ../data/horse.csv outcome


