# SpringBox 
This program simulates active particles in a viscous fluid in two dimensions.
Activated particles interact by pairwise interactions, and effectively through
the coupling with the fluid. The fluid is treated entirely in the Stokesian
limit and is solved with 2D-Stokeslets.

Feel free to use the code to play with your systems. A good starting point for
your experimentation is probably the example.py file, which generates a simple
simulation. Please note that because this is a research code, I cannot
guarantee that any features will remain unchanged.

## Usage:
Make sure to have the environment variable set to include the folder where this
README file is included. On linux machines this can be achieved by adding the
following line to .bashr/.zshrc:
`export PYTHONPATH=$PYTHONPATH:/home/schildi/src/SpringBox`
Then the package can be imported as demonstrated by the examples in the folder
`examples`. To run a simulation execute
`python3 example.py`
