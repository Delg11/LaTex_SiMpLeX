# Simplex Algorithm for Linear Programming

This project implements the Simplex algorithm in Python to solve linear programming problems provided in LaTeX format. It supports both maximization and minimization problems, handles equality, less-than-or-equal, and greater-than-or-equal constraints, and uses the two-phase Simplex method when necessary (e.g., for problems requiring artificial variables). The output is formatted in LaTeX, designed for rendering in Overleaf, providing detailed step-by-step tableaus, pivot operations, and solutions.

## Features
- **LaTeX Input Parsing**: Parses linear programming problems written in LaTeX, extracting objective functions and constraints.
- **Standard and Two-Phase Simplex**: Converts problems to standard form and solves them using the Simplex method, applying two-phase Simplex for problems with artificial variables or `≥` constraints.
- **Comprehensive Output**: Generates LaTeX output, including initial tableaus, pivot operations, and final solutions, with detection of optimal, unbounded, or infeasible solutions.
- **Exact Arithmetic**: Uses Python's `fractions.Fraction` for precise calculations, avoiding floating-point errors.
- **Example Problems**: Includes seven example problems covering various cases, such as maximization, minimization, equality constraints, and unbounded solutions.

## Installation

### Prerequisites
- **Python**: Version 3.8 or higher.

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Delg11/LaTex_SiMpLeX.git
   cd LaTex_SiMpLeX
   ```
2. No external dependencies are required, as the script uses Python's standard library (`re` and `fractions`).

Optionally, create a virtual environment to isolate the project:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

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

This example (Problem 5 in the script) outputs LaTeX code detailing the Simplex iterations, culminating in the optimal solution \( Z = 8, x_1 = 0.67, x_2 = 1.67 \), with a note on non-uniqueness.

### Running Examples
Below are seven example problems. To run them:
1. Execute the code:
   ```python
   from simplex import solve_linear_program
   
   latex_inputs = [
       r"""
       \text{Min Z} &= -2x_1 - x_2 \\
       \text{sujeito a:}&\quad 3x_1 + x_2 \leq 9 \\
       &\quad 2x_1 - 2x_2 \leq 3 \\
       &\quad 0 \leq x_1 \leq 1 \\
       &\quad 0 \leq x_2 \leq 8
       """,
       r"""
       \text{Min Z} &= -x_1 - 2x_2 + x_3 + 2x_4 + 3x_5 \\
       \text{sujeito a:} 
       &\quad 2x_1 - x_2 - x_3 - x_4 + 2x_5 = 0 \\
       &\quad 2x_1 - x_2 + 2x_3 - x_4 + x_5 = 0 \\
       &\quad x_1 + x_2 + x_3 + x_4 + x_5 = 0 \\
       &\quad x_1, x_2, x_3, x_4, x_5 \geq 0
       """,
       r"""
       \text{Max Z} &= x_1 \\
       \text{sujeito a:}&\quad x_1 + 3x_2 + 4x_3 + x_4 = 20 \\
       &\quad 2x_1 + x_3 = 5 \\
       &\quad -7x_1 + 3x_2 + x_4 = 0 \\
       &\quad x_1, x_2, x_3, x_4 \geq 0
       """,
       r"""
       \text{Min Z} &= -x_1 + 3x_2 \\
       \text{sujeito a:}&\quad 2x_1 + 3x_2 \leq 6 \\
       &\quad -x_1 + x_2 \leq 1 \\
       &\quad x_1, x_2 \geq 0
       """,
       r"""
       \text{Max Z} &= 2x_1 + 4x_2 \\
       \text{sujeito a:}&\quad x_1 + 2x_2 \leq 4 \\
       &\quad -x_1 + x_2 \leq 1 \\
       &\quad x_1, x_2 \geq 0
       """,
       r"""
       \text{Max Z} &= x_1 + 3x_2 \\
       \text{sujeito a:}&\quad x_1 - 2x_2 \leq 4 \\
       &\quad -x_1 + x_2 \leq 3 \\
       &\quad x_1, x_2 \geq 0
       """,
       r"""
       \text{Min Z} &= -2x_1 - x_2 \\
       \text{sujeito a:}&\quad 3x_1 + x_2 \geq 9 \\
       &\quad 2x_1 - 2x_2 \geq 3 \\
       &\quad 0 \leq x_1 \leq 1 \\
       &\quad 0 \leq x_2 \leq 8
       """
   ]
   
   latex_inputs = [s.strip() for s in latex_inputs]
   
    for i, latex_input in enumerate(latex_inputs):
       cleaned_input = latex_input.replace(r'\displaystyle', '')
       align_input = r'\begin{align*}' + cleaned_input + r'\end{align*}'
       result = solve_linear_program(latex_input)
       print(f"\\text{{Problema {i+1}:}} \\\\")
       print(align_input)
       print(result)
   ```
2. The script will process each problem and print LaTeX output, which can be copied into Overleaf (see below).

### Input Format
The LaTeX input must follow this structure:
- Objective function: `\text{Max Z} &= <expression>` or `\text{Min Z} &= <expression>`.
- Constraints section: `\text{sujeito a:}`, followed by constraints like `<expression> \leq <value>`, `<expression> \geq <value>`, or `<expression> = <value>`.
- Variable bounds: e.g., `0 \leq x_1 \leq 5` or `x_1, x_2 \geq 0`.

See the `latex_inputs` list in `simplex.py` for examples.

## Rendering LaTeX Output
The output is LaTeX-formatted and designed for rendering in Overleaf.

### Using Overleaf
1. Create a new project in [Overleaf](https://www.overleaf.com).
2. Copy the LaTeX output from the script.
3. Paste it into a `.tex` file with the following preamble:
   ```latex
   \documentclass{article}
   \usepackage{amsmath, amssymb}
   \usepackage{xfrac} % for \sfrac
   \usepackage{xcolor}
   \begin{document}
   % Paste LaTeX output here
   \end{document}
   ```
4. Compile to view tableaus, pivot steps, and solutions.

## Example Output
For the example problem above (Max \( Z = 2x_1 + 4x_2 \), Problem 5), the output includes:
- **Problem Statement**: Original and standard form.
- **Initial Tableau**: With basic variables and coefficients.
- **Iterations**: Pivot operations, entering/leaving variables, and updated tableaus.
- **Solution**: \( Z = 8, x_1 = 0.67, x_2 = 1.67 \), with a note on non-uniqueness.

For a rendered example, copy the output for Problem 5 into Overleaf.

## Testing
The script includes seven example problems testing various scenarios:
- **Problem 1**: Minimization with bounds (\( Z = -26/3 \)).
- **Problem 2**: Minimization with equality constraints (\( Z = 0 \)).
- **Problem 3**: Maximization with equality constraints (\( Z = 2.5 \)).
- **Problem 4**: Minimization with inequalities (\( Z = -3 \)).
- **Problem 5**: Maximization with inequalities (\( Z = 8 \)).
- **Problem 6**: Maximization with inequalities (Unbounded).
- **Problem 7**: Minimization with greater-than-or-equal inequalities (\( Z = -1.50 \)).

To verify, run example above and render the LaTeX output in Overleaf.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

Please include tests for new features and ensure LaTeX output remains compatible with Overleaf.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Credits
- **Developed by**: Gabriel S. Delgado
- **With Assistance from**: Grok 3, created by xAI

## Contact
For questions or suggestions, open an issue on GitHub or contact Gabriel S. Delgado via [GitHub](https://github.com/Delg11).

---

Happy optimizing!
