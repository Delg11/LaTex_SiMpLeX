# Simplex Algorithm for Linear Programming

This project implements the Simplex algorithm in Python to solve linear programming problems provided in LaTeX format. It supports both maximization and minimization problems, handles equality, less-than-or-equal, and greater-than-or-equal constraints, and uses the two-phase Simplex method when necessary (e.g., for problems requiring artificial variables). The output is formatted in LaTeX, making it ideal for rendering in environments like Overleaf, Google Colab Markdown cells, or Jupyter notebooks.

## Features
- **LaTeX Input Parsing**: Parses linear programming problems written in LaTeX, extracting objective functions and constraints.
- **Standard and Two-Phase Simplex**: Converts problems to standard form and solves them using the Simplex method, applying two-phase Simplex for problems with artificial variables or `≥` constraints.
- **Comprehensive Output**: Generates detailed LaTeX output, including initial tableaus, pivot operations, and final solutions, with detection of optimal, unbounded, or infeasible solutions.
- **Exact Arithmetic**: Uses Python's `fractions.Fraction` for precise calculations, avoiding floating-point errors.
- **Example Problems**: Includes seven example problems covering various cases, such as maximization, minimization, equality constraints, and infeasibility.

## Installation

### Prerequisites
- **Python**: Version 3.8 or higher.
- **Optional**: `IPython` for rendering LaTeX output in Jupyter or Google Colab (not required for core functionality).

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Delg11/LaTex_SiMpLeX.git
   cd simplex-algorithm
   ```
2. (Optional) Install `IPython` for Jupyter/Colab:
   ```bash
   pip install ipython
   ```
   Alternatively, create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install ipython
   ```

No additional dependencies are required, as the script uses Python's standard library (`re` and `fractions`).

## Usage

The main script, `simplex.py`, provides a `solve_linear_program` function that takes a LaTeX-formatted linear programming problem as input and returns LaTeX output with the solution process.

### Example
```python
from simplex import solve_linear_program

latex_problem = r'''
\text{Max Z} &= 2x_1 + 4x_2 \\
\text{sujeito a:}&\quad x_1 + 2x_2 \leq 4 \\
&\quad -x_1 + x_2 \leq 1 \\
&\quad x_1, x_2 \geq 0
'''

result = solve_linear_program(latex_problem)
print(result)
```

This example (Problem 5 in the script) outputs LaTeX code detailing the Simplex iterations, culminating in the optimal solution \( Z = 8, x_1 = 0, x_2 = 2 \).

### Running Examples
The script includes seven example problems. To run them:
1. Execute `simplex.py` directly:
   ```bash
   python simplex.py
   ```
2. The script will process each problem and print LaTeX output, which can be copied into a LaTeX renderer (see below).

### Input Format
The LaTeX input must follow this structure:
- Objective function: `\text{Max Z} &= <expression>` or `\text{Min Z} &= <expression>`.
- Constraints section: `\text{sujeito a:}`, followed by constraints like `<expression> \leq <value>`, `<expression> \geq <value>`, or `<expression> = <value>`.
- Variable bounds: e.g., `0 \leq x_1 \leq 5` or `x_1, x_2 \geq 0`.

See the `latex_inputs` list in `simplex.py` for examples.

## Rendering LaTeX Output
The output is LaTeX-formatted and requires a compatible environment to render tableaus and equations.

### Option 1: Overleaf
1. Create a new project in [Overleaf](https://www.overleaf.com).
2. Copy the LaTeX output from the script.
3. Paste it into a `.tex` file with the following preamble:
   ```latex
   \documentclass{article}
   \usepackage{amsmath, amssymb}
   \usepackage{xcolor}
   \usepackage{xfrac} % for \sfrac
   \begin{document}
   % Paste LaTeX output here
   \end{document}
   ```
4. Compile to view tableaus, pivot steps, and solutions.


## Example Output
For the example problem above (Max \( Z = 2x_1 + 4x_2 \)), the output includes:
- **Problem Statement**: Original and standard form.
- **Initial Tableau**: With basic variables and coefficients.
- **Iterations**: Pivot operations, entering/leaving variables, and updated tableaus.
- **Solution**: \( Z = 8, x_1 = 0.67, x_2 = 1.67 \), with a note on non-uniqueness.

For a rendered example, see the output for Problem 5 in Overleaf (optimal solution with two iterations).

## Testing
The script includes seven example problems testing various scenarios:
- **Problem 1**: Minimization with bounds (\( Z =  -26/3 \)).
- **Problem 2**: Minimization with equality constraints (\( Z = 0 \)).
- **Problem 3**: Maximization with equality constraints (\( Z = 2.5 \)).
- **Problem 4**: Minimization with inequalities (\( Z = -3 \)).
- **Problem 5**: Maximization with inequalities (\( Z = 8 \)).
- **Problem 6**: Maximization with inequalities (( Unbound )).
- **Problem 7**: Minimization with greater or equal inequalities (( Z=-1.50 )).


To verify, run `python simplex.py` and check the LaTeX output in Overleaf.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

Please include tests for new features and ensure LaTeX output remains compatible.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Credits
- **Developed by**: Gabriel S. Delgado
- **With Assistance from**: Grok 3, created by xAI

## Contact
For questions or suggestions, open an issue on GitHub or contact Gabriel S. Delgado via [your-preferred-contact-method].

---

Happy optimizing!
