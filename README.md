# GFN2-xTB Parameter Fitter

A Python package for optimizing GFN2-xTB parameters using a Dynamic Domain Genetic Algorithm (DDGA) approach, specifically designed for improving the description of enzymatic reactions at the QM/MM level of theory.

## Description

This package implements a multi-objective evolutionary strategy for optimizing semiempirical Hamiltonians, particularly GFN2-xTB, to enhance their performance in enzymatic QM/MM simulations. The implementation includes working examples for replicating parameter optimization for Crotonyl-CoA carboxylase/reductase (CCR) and dihydrofolate reductase (DHFR) systems.

## Project Structure

```
gfn2-xtb_paramfitter/
├── algorithms/             # Core algorithm implementations
│   ├── __init__.py
│   └── DDGeneticAlgorithm.py
├── experiments/           # Example optimization scripts
│   ├── __init__.py
│   ├── DDGA_OPTIM_CCR.py
│   └── DDGA_OPTIM_DHFR.py
├── fitters/              # Parameter fitting modules
├── inputhandlers/        # Input processing utilities
├── objectivefunctions/   # Objective function definitions
│   ├── __init__.py
│   └── ECGFittingV1.py
├── parameters/           # Parameter configurations
├── readers/             # File reading utilities
├── runners/             # Execution modules
└── requirements.txt     # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/josevlibera2010/gfn2-xtb_paramfitter.git
cd gfn2-xtb_paramfitter
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The package includes example scripts in the `experiments/` directory that demonstrate the parameter optimization process for CCR and DHFR systems:

- `DDGA_OPTIM_CCR.py`: Example script for CCR parameter optimization
- `DDGA_OPTIM_DHFR.py`: Example script for DHFR parameter optimization

To run an optimization experiment:

```bash
export PYTHONPATH=/path/to/gfn2-xtb_paramfitter:${PYTHONPATH}

python DDGA_OPTIM_CCR.py
# or
python DDGA_OPTIM_DHFR.py
```

Each script contains detailed comments explaining the optimization process, parameter settings, and how to adapt the code for your own systems. You can use these scripts as templates for implementing optimizations for other enzymatic systems.

## Citation

Velázquez-Libera, J. L., Recabarren, R., Vöhringer-Martinez, E., Salgueiro, Y., Ruiz-Pernía, J. J., Caballero, J., & Tuñón, I. (2025). __Multi-objective evolutionary strategy for improving semiempirical  Hamiltonians in the study of enzymatic reactions at the QM/MM level of theory.__ _ChemRxiv_. [doi:10.26434/chemrxiv-2025-pvztk](https://chemrxiv.org/engage/chemrxiv/article-details/67ab41626dde43c90863e00a)  This content is a preprint and has not been peer-reviewed.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Authors

- José Luís Velázquez-Libera (@josevlibera2010)

## Contact

For questions and feedback:
- Email: josevlibera2010@gmail.com
