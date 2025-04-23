# Linear Programming Solver

A Python library for parsing and solving linear programming problems using LaTeX input and the Simplex method.

## Overview

This project consists of two main components:
1. **Parser and Standard Form Converter** (`lp_parser.py`): Parses LaTeX-formatted linear programming problems and converts them to standard form (minimization with equality constraints).
2. **Simplex Solver** (`simplex_solver.py`): Solves linear programming problems in standard form using the Simplex method, outputting detailed iterations in LaTeX.

**Created by Gabriel S. Delgado with assistance from Grok 3 by xAI.**

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Part 1: Parsing and Standardizing
```python
from lp_parser import process_linear_program
from IPython.display import display, Math

latex_input = r"""
\text{Max Z} &= 2x_1 + x_2 \\
\text{sujeito a:}&\quad x_1 + x_2 \leq 1 \\
&\quad x_1, x_2 \geq 0
"""

result = process_linear_program(latex_input)
display(Math('\\text{Problem in Standard Form:}' + result))
```

### Part 2: Solving with Simplex
```python
from simplex_solver import solve_simplex
from IPython.display import display, Math

latex_input = r"""
\displaystyle \text{Min -Z} &= -x_1 - x_2 \\
\text{sujeito a:} 
&\quad x_1 + x_2 + x_3 = 1 \\
&\quad x_1, x_2, x_3 \geq 0
"""

result = solve_simplex(latex_input)
display(Math('\\text{Simplex Solution:}' + result))
```

## Examples

See the `examples/` folder for Jupyter notebooks demonstrating both components.

## License

MIT License. See `LICENSE` for details.

## Acknowledgments

- Created by Gabriel S. Delgado.
- Assisted by Grok 3, developed by xAI.
