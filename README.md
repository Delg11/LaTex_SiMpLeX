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

Initial input:
```latex
    \text{Min Z} &= -2x_1 - x_2 \\
    \text{sujeito a:}&\quad 3x_1 + x_2 \leq 9 \\
    &\quad 2x_1 - 2x_2 \leq 3 \\
    &\quad 0 \leq x_1 \leq 1 \\
    &\quad 0 \leq x_2 \leq 8
```

Final output:
```latex
\text{Tableau inicial:}
\[
\begin{array}{|c|cccccc|c|}
\hline
\text{VB} & x_1 & x_2 & x_3 & x_4 & x_5 & x_6 & \text{b} \\
\hline
{x_2} & 3 & 1 & 1 & 0 & 0 & 0 & 9 \\
{x_4} & 2 & -2 & 0 & 1 & 0 & 0 & 3 \\
{x_1} & 1 & 0 & 0 & 0 & 1 & 0 & 1 \\
{x_6} & 0 & 1 & 0 & 0 & 0 & 1 & 8 \\
\hline
\text{-Z} & -2 & -1 & 0 & 0 & 0 & 0 & 0 \\
\hline
\end{array}
\]
Tableau após pivoteamento inicial para forma canônica:
\[
\begin{array}{|c|cccccc|c|}
\hline
\text{VB} & x_1 & x_2 & x_3 & x_4 & x_5 & x_6 & \text{b} \\
\hline
{x_2} & 0 & 1 & 1 & 0 & -3 & 0 & 6 \\
{x_4} & 0 & 0 & 2 & 1 & -8 & 0 & 13 \\
{x_1} & 1 & 0 & 0 & 0 & 1 & 0 & 1 \\
{x_6} & 0 & 0 & -1 & 0 & 3 & 1 & 2 \\
\hline
\text{-Z} & 0 & 0 & 1 & 0 & -1 & 0 & 8 \\
\hline
\end{array}
\]

\text{Iteração 1:}
\text{Candidato a entrar na base: } \(x_5\).\\
\text{Teste da razão para verificar quem sai da base:}\\
\( x_2:\dfrac{6}{-3}=\nexists, x_4:\dfrac{13}{-8}=\nexists, x_1:\dfrac{1}{1}=1, { \color{red}{x_6:\dfrac{2}{3}=\sfrac{2}{3}} } \)\\
\text{Candidato a sair da base: } \(x_6\).\\

\text{Novo Tableau:}
\[
\begin{array}{|c|cccccc|c|}
\hline
\text{VB} & x_1 & x_2 & x_3 & x_4 & x_5 & x_6 & \text{b} \\
\hline
{x_2} & 0 & 1 & 0 & 0 & 0 & 1 & 8 \\
{x_4} & 0 & 0 & \sfrac{-2}{3} & 1 & 0 & \sfrac{8}{3} & \sfrac{55}{3} \\
{x_1} & 1 & 0 & \sfrac{1}{3} & 0 & 0 & \sfrac{-1}{3} & \sfrac{1}{3} \\
{x_5} & 0 & 0 & \sfrac{-1}{3} & 0 & 1 & \sfrac{1}{3} & \sfrac{2}{3} \\
\hline
\text{-Z} & 0 & 0 & \sfrac{2}{3} & 0 & 0 & \sfrac{1}{3} & \sfrac{26}{3} \\
\hline
\end{array}
\]

\text{Iteração 2:}
Nenhum coeficiente negativo para variáveis não básicas na linha do objetivo. Solução ótima alcançada.
Como não temos mais valores de custo negativo, a solução atual,  \(z=-26/3\), é ótima. 
A solução \(\left(1/3,8,0,55/3,2/3,0\right)^T\) é a única partição básica ótima.
```

See the `examples/` folder for Jupyter notebooks demonstrating both components.

## License

MIT License. See `LICENSE` for details.

## Acknowledgments

- Created by Gabriel S. Delgado.
- Assisted by Grok 3, developed by xAI.
