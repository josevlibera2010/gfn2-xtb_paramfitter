"""
Physical constants and unit conversion factors for quantum chemistry calculations.

All values are based on CODATA 2018 recommended values where applicable.
"""

# Energy conversion factors
HARTREE_TO_KCAL_MOL = 627.509391  # 1 Hartree = 627.509391 kcal/mol
EV_TO_HARTREE = 27.2113961318     # 1 eV = 1/27.211... Hartree (cclib uses eV internally)

# Length conversion factors
BOHR_TO_ANGSTROM = 0.529177249    # 1 Bohr = 0.529177249 Angstrom
ANGSTROM_TO_BOHR = 1.8897259885789  # 1 Angstrom = 1.8897259885789 Bohr