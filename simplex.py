"""
Simplex Algorithm Implementation for Linear Programming

This module implements the Simplex algorithm to solve linear programming problems
provided in LaTeX format. It supports both maximization and minimization problems,
handles equality, less-than-or-equal, and greater-than-or-equal constraints, and
uses the two-phase Simplex method when necessary (e.g., for problems requiring
artificial variables). The output is formatted in LaTeX for easy rendering in
environments like Overleaf or Google Colab Markdown cells, providing detailed
step-by-step tableaus, pivot operations, and solutions.

Features:
- Parses LaTeX input to extract objective function and constraints.
- Converts problems to standard form (minimization with equality constraints).
- Builds and pivots Simplex tableaus, handling canonicalization.
- Supports two-phase Simplex for problems with artificial variables or >= constraints.
- Detects optimal, unbounded, or infeasible solutions.
- Generates LaTeX output for tableaus, iterations, and final solutions.

Usage:
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

Dependencies:
- Python 3.x
- `re` (standard library) for parsing LaTeX expressions.
- `fractions.Fraction` (standard library) for exact arithmetic.
- `IPython.display` (optional, for Jupyter/Colab display; not required for core functionality).

Developed by:
- Gabriel S. Delgado
- With assistance from Grok 3, created by xAI

License:
- MIT License (assumed; specify your preferred license when hosting on GitHub).

Example:
    The script includes seven example problems demonstrating various cases
    (minimization, maximization, equality constraints, infeasibility).
    Run the script directly to process these examples and print LaTeX output.

Notes:
- Ensure LaTeX input follows the expected format (see examples).
- Output is designed for rendering in LaTeX-compatible environments.
- For GitHub hosting, consider adding a README.md with setup instructions
  and example outputs rendered in Overleaf.
"""

import re
from fractions import Fraction
from IPython.display import display, Math

def parse_latex_problem(latex_input):
    """
    Parse a LaTeX linear programming problem into its components.
    Returns: problem_type ('Max' or 'Min'), objective, constraints, variables.
    """
    lines = [line.strip() for line in latex_input.split('\n') if line.strip()]
    problem_type = None
    objective = None
    constraints = []
    variables = set()

    for line in lines:
        if r'\text{Max' in line or r'\text{Min' in line:
            if r'\text{Max' in line:
                problem_type = 'Max'
            else:
                problem_type = 'Min'
            line = re.sub(r'&\s*\\quad\s*', '', line)
            obj_match = re.search(r'&=([^\\]+)\\', line)
            if obj_match:
                objective = obj_match.group(1).strip()
                vars_in_obj = re.findall(r'x_(\d+)', objective)
                variables.update([f'x_{i}' for i in vars_in_obj])
            break

    constraint_section = False
    for line in lines:
        cleaned_line = re.sub(r'&\s*\\quad\s*', '', line).strip()
        if r'\text{sujeito a:}' in line:
            constraint_section = True
            constraint_part = cleaned_line.replace(r'\text{sujeito a:}', '').strip()
            if constraint_part and (r'\leq' in constraint_part or r'\geq' in constraint_part or r'=' in constraint_part):
                constraints.append(constraint_part)
                vars_in_const = re.findall(r'x_(\d+)', constraint_part)
                variables.update([f'x_{i}' for i in vars_in_const])
            continue
        if constraint_section:
            upper_bound_match = re.match(r'(0\s*\\leq\s*)?x_(\d+)\s*\\leq\s*(\d+)', cleaned_line)
            if upper_bound_match:
                var_index = upper_bound_match.group(2)
                upper_bound = upper_bound_match.group(3)
                constraint = f"x_{var_index} \\leq {upper_bound}"
                constraints.append(constraint)
                variables.add(f'x_{var_index}')
            elif (r'\leq' in cleaned_line or r'\geq' in cleaned_line or r'=' in cleaned_line) and not (r'\geq 0' in cleaned_line and 'x_' in cleaned_line and '+' not in cleaned_line and '-' not in cleaned_line):
                constraints.append(cleaned_line)
                vars_in_const = re.findall(r'x_(\d+)', cleaned_line)
                variables.update([f'x_{i}' for i in vars_in_const])

    constraints = [re.sub(r'\\+$', '', c).strip() for c in constraints]
    return problem_type, objective, constraints, sorted(list(variables), key=lambda x: int(x.split('_')[1]))

def parse_expression(expr, all_variables):
    """
    Parse a LaTeX expression into coefficients for all variables.
    Returns: list of coefficients in order of all_variables.
    """
    expr = expr.replace(' + ', '+').replace(' - ', '-').replace('+ -', '-')
    if not expr.startswith('-') and not expr.startswith('+'):
        expr = '+' + expr
    terms = re.split(r'(?=[-+])', expr)
    terms = [term.strip() for term in terms if term.strip()]
    coefficients = {var: 0 for var in all_variables}

    for term in terms:
        match = re.match(r'([+-]?)(\d+)?([xsa]_)(\d+)', term.replace(' ', ''))
        if match:
            sign = match.group(1)
            coeff_str = match.group(2)
            var_prefix = match.group(3)
            var_index = match.group(4)
            var = f'{var_prefix}{var_index}'
            coeff = int(coeff_str) if coeff_str else 1
            if sign == '-':
                coeff = -coeff
            if var in coefficients:
                coefficients[var] = coeff
        elif re.match(r'([+-]?\d+)', term):
            continue

    return [coefficients[var] for var in all_variables]

def convert_to_standard_form(problem_type, objective, constraints, variables):
    """
    Convert the problem to standard form for Simplex.
    Returns: new objective, new constraints, all variables, slack variables, artificial variables, has_geq_constraints.
    """
    new_objective = objective
    new_constraints = []
    slack_var_count = max([int(var.split('_')[1]) for var in variables]) + 1 if variables else 1
    artificial_var_count = 1
    all_variables = variables.copy()
    slack_vars = []
    artificial_vars = []
    has_geq_constraints = False

    if problem_type == 'Max':
        coeffs = parse_expression(objective, variables)
        new_terms = []
        for coeff, var in zip(coeffs, variables):
            if coeff == 0:
                continue
            new_coeff = -coeff
            term_str = (f"{new_coeff}x_{var.split('_')[1]}" if new_coeff != -1 and new_coeff != 1 else
                        f"-x_{var.split('_')[1]}" if new_coeff == -1 else
                        f"x_{var.split('_')[1]}")
            new_terms.append(term_str)
        new_objective = ' + '.join(new_terms).replace('+ -', '- ')
    else:
        coeffs = parse_expression(objective, variables)
        new_terms = []
        for coeff, var in zip(coeffs, variables):
            if coeff == 0:
                continue
            term_str = (f"{coeff}x_{var.split('_')[1]}" if coeff != 1 and coeff != -1 else
                        f"-x_{var.split('_')[1]}" if coeff == -1 else
                        f"x_{var.split('_')[1]}")
            new_terms.append(term_str)
        new_objective = ' + '.join(new_terms).replace('+ -', '- ')

    for constraint in constraints:
        constraint = re.sub(r'&\s*\\quad\s*', '', constraint)
        upper_bound_match = re.match(r'(0\s*\\leq\s*)?x_(\d+)\s*\\leq\s*(\d+)', constraint)
        if upper_bound_match:
            var_index = upper_bound_match.group(2)
            upper_bound = upper_bound_match.group(3)
            slack_var = f's_{slack_var_count}'
            all_variables.append(slack_var)
            slack_vars.append(slack_var)
            new_constraint = f"x_{var_index} + {slack_var} = {upper_bound}"
            new_constraints.append(new_constraint)
            slack_var_count += 1
            continue

        if r'\leq' in constraint:
            left, right = constraint.split(r'\leq')
            ineq_type = 'leq'
        elif r'\geq' in constraint:
            left, right = constraint.split(r'\geq')
            ineq_type = 'geq'
            has_geq_constraints = True
        else:
            left, right = constraint.split('=')
            ineq_type = 'eq'
        left = left.strip()
        right = right.strip().split('\\')[0].strip()

        vars_in_left = re.findall(r'x_(\d+)', left)
        for var_idx in vars_in_left:
            var = f'x_{var_idx}'
            if var not in all_variables:
                all_variables.append(var)

        terms = parse_expression(left, all_variables)
        new_terms = []
        for coeff, var in zip(terms, all_variables):
            if coeff == 0:
                continue
            term_str = (f"{coeff}{var}" if coeff != 1 and coeff != -1 else
                        f"-{var}" if coeff == -1 else
                        f"{var}")
            new_terms.append(term_str)
        new_left = ' + '.join(new_terms).replace('+ -', '- ')

        if ineq_type == 'leq':
            slack_var = f's_{slack_var_count}'
            all_variables.append(slack_var)
            slack_vars.append(slack_var)
            new_constraint = f"{new_left} + {slack_var} = {right}" if new_left else f"{slack_var} = {right}"
            slack_var_count += 1
        elif ineq_type == 'geq':
            excess_var = f's_{slack_var_count}'
            artificial_var = f'a_{artificial_var_count}'
            all_variables.extend([excess_var, artificial_var])
            slack_vars.append(excess_var)
            artificial_vars.append(artificial_var)
            new_constraint = f"{new_left} - {excess_var} + {artificial_var} = {right}" if new_left else f"- {excess_var} + {artificial_var} = {right}"
            slack_var_count += 1
            artificial_var_count += 1
        else:
            artificial_var = f'a_{artificial_var_count}'
            all_variables.append(artificial_var)
            artificial_vars.append(artificial_var)
            new_constraint = f"{new_left} + {artificial_var} = {right}" if new_left else f"{artificial_var} = {right}"
            artificial_var_count += 1

        new_constraints.append(new_constraint)

    all_variables = sorted([v for v in all_variables if v.startswith('x_')], key=lambda x: int(x.split('_')[1])) + \
                    sorted([v for v in all_variables if v.startswith('s_')], key=lambda x: int(x.split('_')[1])) + \
                    sorted([v for v in all_variables if v.startswith('a_')], key=lambda x: int(x.split('_')[1]))
    return new_objective, new_constraints, all_variables, slack_vars, artificial_vars, has_geq_constraints

def build_initial_tableau(problem_type, objective, constraints, variables, slack_vars, artificial_vars, has_geq_constraints):
    """
    Build the initial Simplex tableau, identifying basic variables.
    Returns: basic_vars, tableau, needs_two_phase, constraint_coeffs, right_hand_sides, artificial_indices.
    """
    tableau = []
    basic_vars = []
    constraint_coeffs = []
    right_hand_sides = []
    needs_two_phase = bool(artificial_vars) or has_geq_constraints
    artificial_indices = []

    for constraint in constraints:
        left, right = constraint.split('=')
        left = left.strip()
        right = int(right.strip().split('\\')[0].strip())
        coeffs = parse_expression(left, variables)
        while len(coeffs) < len(variables):
            coeffs.append(0)
        constraint_coeffs.append(coeffs)
        right_hand_sides.append(right)

    obj_coeffs = parse_expression(objective, variables)
    while len(obj_coeffs) < len(variables):
        obj_coeffs.append(0)

    if not needs_two_phase:
        for i in range(len(constraints)):
            for j, var in enumerate(variables):
                if constraint_coeffs[i][j] == 1 and var in slack_vars and var not in basic_vars:
                    basic_vars.append(var)
                    tableau.append(constraint_coeffs[i] + [right_hand_sides[i]])
                    break
            else:
                needs_two_phase = True
                break

    if needs_two_phase:
        basic_vars = []
        tableau = []
        for i in range(len(constraints)):
            basic_var = None
            for j, var in enumerate(variables):
                if constraint_coeffs[i][j] == 1 and var in slack_vars and var not in basic_vars:
                    basic_var = var
                    break
            if not basic_var:
                for j, var in enumerate(variables):
                    if constraint_coeffs[i][j] == 1 and var in artificial_vars and var not in basic_vars:
                        basic_var = var
                        artificial_indices.append(j)
                        break
            if not basic_var:
                basic_var = f'a_{len(artificial_vars) + 1}'
                artificial_vars.append(basic_var)
                variables.append(basic_var)
                for coeffs in constraint_coeffs:
                    coeffs.append(0)
                constraint_coeffs[i][-1] = 1
                artificial_indices.append(len(variables) - 1)
                obj_coeffs.append(0)
            basic_vars.append(basic_var)
        for i in range(len(constraints)):
            tableau.append(constraint_coeffs[i] + [right_hand_sides[i]])

    tableau.append(obj_coeffs + [0])
    basic_vars.append('-z')

    for row in tableau:
        if len(row) != len(variables) + 1:
            raise ValueError(f"Inconsistent tableau row length: {len(row)} vs {len(variables) + 1}")

    return basic_vars, tableau, needs_two_phase, constraint_coeffs, right_hand_sides, artificial_indices

def pivot_tableau(basic_vars, tableau, pivot_row, pivot_col, variables):
    """
    Pivot the tableau around the specified row and column.
    Returns: new basic_vars, new tableau.
    """
    new_tableau = [[Fraction(coeff) for coeff in row] for row in tableau]
    pivot = new_tableau[pivot_row][pivot_col]

    for j in range(len(new_tableau[pivot_row])):
        new_tableau[pivot_row][j] /= pivot

    for i in range(len(new_tableau)):
        if i != pivot_row:
            factor = new_tableau[i][pivot_col]
            for j in range(len(new_tableau[i])):
                new_tableau[i][j] -= factor * new_tableau[pivot_row][j]

    new_basic_vars = basic_vars.copy()
    new_basic_vars[pivot_row] = variables[pivot_col]
    return new_basic_vars, new_tableau

def canonicalize_tableau(basic_vars, tableau, variables):
    """
    Pivot the tableau to ensure each basic variable's column is in canonical form.
    Returns: updated basic_vars, tableau.
    """
    new_tableau = tableau
    new_basic_vars = basic_vars

    for i in range(len(basic_vars) - 1):
        basic_var = basic_vars[i]
        var_idx = variables.index(basic_var)
        is_canonical = True
        for r in range(len(tableau) - 1):
            expected = 1 if r == i else 0
            if abs(tableau[r][var_idx] - expected) > 1e-10:
                is_canonical = False
                break
        if not is_canonical and abs(tableau[i][var_idx]) > 1e-10:
            new_basic_vars, new_tableau = pivot_tableau(new_basic_vars, new_tableau, i, var_idx, variables)

    return new_basic_vars, new_tableau

def format_tableau_latex(basic_vars, tableau, variables, is_phase_one=False):
    """
    Format the Simplex tableau in LaTeX, with two objective rows in Phase 1.
    Returns: LaTeX string.
    """
    num_vars = len(variables)
    col_spec = '|c|' + 'c' * num_vars + '|c|'
    latex_lines = [r'\[']
    latex_lines.append(r'\begin{array}{' + col_spec + '}')
    latex_lines.append(r'\hline')
    header = [r'\text{VB}'] + [var for var in variables] + [r'\text{b}']
    latex_lines.append(' & '.join(header) + r' \\')
    latex_lines.append(r'\hline')

    for i, row in enumerate(tableau[:-2] if is_phase_one else tableau[:-1]):
        row_str = [f'{{{basic_vars[i]}}}' if basic_vars[i].startswith(('x_', 's_', 'a_')) else basic_vars[i]]
        for coeff in row:
            if isinstance(coeff, Fraction):
                if coeff.denominator == 1:
                    row_str.append(str(coeff.numerator))
                else:
                    row_str.append(f'\\sfrac{{{coeff.numerator}}}{{{coeff.denominator}}}')
            else:
                row_str.append(str(coeff))
        latex_lines.append(' & '.join(row_str) + r' \\')

    latex_lines.append(r'\hline')
    for idx, obj_var in enumerate(['-z', '-w'] if is_phase_one else ['-z']):
        row = tableau[-2 + idx] if is_phase_one else tableau[-1]
        obj_row = [f'\\text{{{basic_vars[-2 + idx if is_phase_one else -1]}}}']
        for coeff in row:
            if isinstance(coeff, Fraction):
                if coeff.denominator == 1:
                    obj_row.append(str(coeff.numerator))
                else:
                    obj_row.append(f'\\sfrac{{{coeff.numerator}}}{{{coeff.denominator}}}')
            else:
                obj_row.append(str(coeff))
        if is_phase_one and obj_var == '-z':
            latex_lines.append(' & '.join(obj_row) + r' \\')
            latex_lines.append(r'\hline')
        else:
            latex_lines.append(' & '.join(obj_row) + r' \\')

    latex_lines.append(r'\hline')
    latex_lines.append(r'\end{array}')
    latex_lines.append(r'\]')
    return '\n'.join(latex_lines)

def analyze_tableau(basic_vars, tableau, variables, problem_type, is_phase_one=False):
    """
    Analyze the tableau to find entering and leaving variables.
    Returns: entering_var, leaving_var, analysis_latex, is_optimal, is_unbounded.
    """
    obj_row = tableau[-1][:-1] if is_phase_one else tableau[-1][:-1]
    non_basic_vars = [var for var in variables if var not in basic_vars[:-2 if is_phase_one else -1]]
    is_optimal = True
    pivot_candidates = []

    for var in non_basic_vars:
        idx = variables.index(var)
        if problem_type == 'Min' and obj_row[idx] < -1e-10:
            is_optimal = False
            pivot_candidates.append((obj_row[idx], idx, var))
        elif problem_type == 'Max' and obj_row[idx] > 1e-10:
            is_optimal = False
            pivot_candidates.append((obj_row[idx], idx, var))

    if is_optimal:
        return None, None, r"\text{Solução ótima alcançada.} \\" + r"", True, False

    pivot_candidates.sort(key=lambda x: x[0], reverse=(problem_type == 'Max'))
    for _, entering_idx, entering_var in pivot_candidates:
        ratios = []
        ratio_strings = []
        all_non_positive = True
        for i, row in enumerate(tableau[:-2 if is_phase_one else -1]):
            pivot_coeff = row[entering_idx]
            b_value = row[-1]
            if pivot_coeff > 1e-10 and b_value >= -1e-10:
                all_non_positive = False
                ratio = Fraction(b_value, pivot_coeff).limit_denominator()
                ratios.append((ratio, i))
                ratio_str = f"{basic_vars[i]}:\\dfrac{{{b_value}}}{{{pivot_coeff}}}"
                if ratio.denominator == 1:
                    ratio_str += f"={ratio.numerator}"
                else:
                    ratio_str += f"=\\sfrac{{{ratio.numerator}}}{{{ratio.denominator}}}"
                ratio_strings.append(ratio_str)
            else:
                ratio_strings.append(f"{basic_vars[i]}:\\dfrac{{{b_value}}}{{{pivot_coeff}}}=\\nexists")

        if not ratios and all_non_positive and not is_phase_one:
            unbounded_message = (
                f"Solução ilimitada identificada: a variável \\({entering_var}\\) "
                f"tem coeficiente \\({float(obj_row[entering_idx]):.2f} < 0\\) "
                f"na função objetivo, com coeficientes não-positivos \\(\\leq 0\\) "
                f"na sua coluna. Assim, \\({entering_var}\\) "
                f"pode aumentar indefinidamente, reduzindo \\(Z\\) sem limite. \\"
            )
            return None, None, unbounded_message, False, True

        if ratios:
            min_ratio, leaving_idx = min(ratios, key=lambda x: x[0])
            leaving_var = basic_vars[leaving_idx]

            formatted_ratio_strings = []
            for i in range(len(ratio_strings)):
                if i == leaving_idx:
                    formatted_ratio_strings.append(f"{{ \\color{{red}}{{{ratio_strings[i]}}} }}")
                else:
                    formatted_ratio_strings.append(ratio_strings[i])

            analysis_latex = [
                f"\\text{{Entrando: }} \\({entering_var}\\). \\\\",
                r"\text{Teste da razão:} \\",
                f"\\( {', '.join(formatted_ratio_strings)} \\) \\\\",
                f"\\text{{Saindo: }} \\({leaving_var}\\). \\\\",
                r""
            ]
            return entering_var, leaving_var, '\n'.join(analysis_latex), False, False

    return None, None, r"\text{Nenhum pivô válido encontrado para variáveis com coeficientes negativos.} \\" + r"", True, False

def two_phase_simplex(basic_vars, tableau, variables, artificial_vars, original_obj_coeffs, problem_type, artificial_indices, original_objective, new_constraints):
    """
    Apply the two-phase Simplex method with two-row tableau in Phase 1.
    Returns: LaTeX string with all iterations and solution.
    """
    output_latex = [r"\text{Fase 1: Minimizar a soma das variáveis artificiais} \\"]
    output_latex.append(r"\begin{align*}")
    w_terms = [var for var in artificial_vars]
    if w_terms:
        w_expr = ' + '.join(w_terms)
        output_latex.append(f"\\text{{Min W}} &= {w_expr} \\\\")
        output_latex.append(r'\text{sujeito a:}')
        for constraint in new_constraints:
            output_latex.append(f"&\\quad {constraint} \\\\")
        var_list = ', '.join(variables)
        output_latex.append(f"&\\quad {var_list} \\geq 0")
        output_latex.append(r"\end{align*}")
    else:
        output_latex.append(r"\text{Nenhuma variável artificial necessária.} \\")
        output_latex.append(r"\end{align*}")

    phase1_tableau = [[Fraction(coeff) for coeff in row] for row in tableau]
    phase1_basic_vars = basic_vars.copy()
    phase1_basic_vars[-1] = '-z'

    phase1_tableau.append([Fraction(0) for _ in range(len(variables))] + [Fraction(0)])
    phase1_basic_vars.append('-w')
    for idx in artificial_indices:
        if idx < len(variables):
            phase1_tableau[-1][idx] = Fraction(1)

    output_latex.append(r"\text{Tableau inicial da Fase 1 (Base artificial):} \\")
    output_latex.append(format_tableau_latex(phase1_basic_vars, phase1_tableau, variables, is_phase_one=True))

    for i in range(len(phase1_basic_vars) - 2):
        basic_var = phase1_basic_vars[i]
        if basic_var in artificial_vars:
            var_idx = variables.index(basic_var)
            if abs(phase1_tableau[-1][var_idx]) > 1e-10:
                for j in range(len(phase1_tableau[-1])):
                    phase1_tableau[-1][j] -= phase1_tableau[-1][var_idx] * phase1_tableau[i][j]

    output_latex.append(r"\text{Tableau após Pricing Out:} \\")
    output_latex.append(format_tableau_latex(phase1_basic_vars, phase1_tableau, variables, is_phase_one=True))

    iteration = 1
    while True:
        entering_var, leaving_var, analysis, is_done, is_unbounded = analyze_tableau(
            phase1_basic_vars, phase1_tableau, variables, 'Min', is_phase_one=True
        )
        output_latex.append(f"\\text{{Iteração {iteration} (Fase 1):}} \\\\")
        output_latex.append(analysis)

        if is_unbounded:
            output_latex.append(r"\text{Erro: Solução ilimitada detectada na Fase 1 (indica problema na formulação).} \\")
            return '\n'.join(output_latex)

        if is_done:
            if phase1_tableau[-1][-1] < -1e-10:
                output_latex.append(
                    r"\text{Soma das variáveis artificiais não é zero. O problema é inviável.} \\"
                )
                return '\n'.join(output_latex)
            while any(var in artificial_vars for var in phase1_basic_vars[:-2]):
                pivot_performed = False
                for i, basic_var in enumerate(phase1_basic_vars[:-2]):
                    if basic_var in artificial_vars:
                        var_idx = variables.index(basic_var)
                        for col, var in enumerate(variables):
                            if var not in artificial_vars and abs(phase1_tableau[i][col]) > 1e-10:
                                entering_var = var
                                leaving_var = basic_var
                                pivot_row = i
                                pivot_col = col
                                phase1_basic_vars, phase1_tableau = pivot_tableau(
                                    phase1_basic_vars, phase1_tableau, pivot_row, pivot_col, variables
                                )
                                output_latex.append(
                                    f"\\text{{Pivô adicional para remover variável artificial \\({basic_var}\\):}} \\"
                                )
                                output_latex.append(
                                    format_tableau_latex(phase1_basic_vars, phase1_tableau, variables, is_phase_one=True)
                                )
                                pivot_performed = True
                                break
                        if pivot_performed:
                            break
                if not pivot_performed:
                    break
            break

        pivot_row = phase1_basic_vars.index(leaving_var)
        pivot_col = variables.index(entering_var)
        phase1_basic_vars, phase1_tableau = pivot_tableau(
            phase1_basic_vars, phase1_tableau, pivot_row, pivot_col, variables
        )
        output_latex.append(f"\\text{{Novo Tableau (Fase 1):}} \\")
        output_latex.append(format_tableau_latex(phase1_basic_vars, phase1_tableau, variables, is_phase_one=True))

        iteration += 1

    output_latex.append(r"\text{Fase 1 concluída. Iniciando Fase 2.} \\")

    keep_indices = [i for i, var in enumerate(variables) if not var.startswith('a_')]
    phase2_variables = [variables[i] for i in keep_indices]
    phase2_tableau = [[row[i] for i in keep_indices] + [row[-1]] for row in phase1_tableau[:-1]]
    phase2_basic_vars = phase1_basic_vars[:-1]

    phase2_obj_coeffs = [original_obj_coeffs[i] for i in keep_indices]
    phase2_tableau[-1] = [Fraction(coeff) for coeff in phase2_obj_coeffs] + [Fraction(0)]

    for i in range(len(phase2_basic_vars) - 1):
        basic_var = phase2_basic_vars[i]
        if basic_var in phase2_variables:
            var_idx = phase2_variables.index(basic_var)
            if var_idx < len(phase2_tableau[-1]) - 1:
                coeff = phase2_tableau[-1][var_idx]
                if abs(coeff) > 1e-10:
                    for j in range(len(phase2_tableau[-1])):
                        phase2_tableau[-1][j] -= coeff * phase2_tableau[i][j]

    output_latex.append(r"\text{Tableau inicial da Fase 2:} \\")
    output_latex.append(format_tableau_latex(phase2_basic_vars, phase2_tableau, phase2_variables, is_phase_one=False))

    iteration = 1
    while True:
        entering_var, leaving_var, analysis, is_done, is_unbounded = analyze_tableau(
            phase2_basic_vars, phase2_tableau, phase2_variables, 'Min', is_phase_one=False
        )
        output_latex.append(f"\\text{{Iteração {iteration} (Fase 2):}} \\\\")
        output_latex.append(analysis)

        if is_unbounded:
            output_latex.append(r"\text{Solução ilimitada na Fase 2.} \\")
            return '\n'.join(output_latex)

        if is_done:
            output_latex.append(get_solution(phase2_basic_vars, phase2_tableau, phase2_variables, problem_type, original_objective))
            break

        pivot_row = phase2_basic_vars.index(leaving_var)
        pivot_col = phase2_variables.index(entering_var)
        phase2_basic_vars, phase2_tableau = pivot_tableau(
            phase2_basic_vars, phase2_tableau, pivot_row, pivot_col, phase2_variables
        )
        output_latex.append(f"\\text{{Novo Tableau (Fase 2):}} \\")
        output_latex.append(format_tableau_latex(phase2_basic_vars, phase2_tableau, phase2_variables, is_phase_one=False))

        iteration += 1

    return '\n'.join(output_latex)

def get_solution(basic_vars, tableau, variables, problem_type, original_objective):
    """
    Extract the optimal solution and compute z using the original objective function.
    Returns: LaTeX string with solution and conclusion.
    """
    solution = {var: Fraction(0) for var in variables}
    for i, var in enumerate(basic_vars[:-1]):
        if var in solution:
            solution[var] = tableau[i][-1]

    obj_coeffs = parse_expression(original_objective, variables)
    z_value = Fraction(0)
    for coeff, var in zip(obj_coeffs, variables):
        z_value += coeff * solution[var]

    non_basic_vars = [var for var in variables if var not in basic_vars[:-1]]
    alternate_optima = False
    zero_coeff_vars = []
    for var in non_basic_vars:
        idx = variables.index(var)
        if abs(tableau[-1][idx]) < 1e-10:
            alternate_optima = True
            zero_coeff_vars.append(var)

    solution_values = [solution.get(var, 0) for var in variables]
    solution_latex = [
        f"\\text{{Solução ótima: }} \\(z={float(z_value):.2f}, \\left({','.join(f'{float(v):.2f}' for v in solution_values)}\\right)^T\\). \\",
        r""
    ]
    if alternate_optima:
        solution_latex.append(f"\\text{{Soluções alternativas existem devido a custos zero em: }} \\({', '.join(zero_coeff_vars)}\\). \\")
    else:
        solution_latex.append(r"\text{Solução única.} \\")
    return '\n'.join(solution_latex)

def solve_linear_program(latex_input):
    """
    Main function to process and solve the linear programming problem.
    Returns: LaTeX string with standard form and Simplex solution.
    """
    problem_type, objective, constraints, variables = parse_latex_problem(latex_input)
    output_latex = [r"\text{Problema original:} \\"]
    output_latex.append(r"\begin{align*}")
    output_latex.append(latex_input)
    output_latex.append(r"\end{align*}")

    new_objective, new_constraints, all_variables, slack_vars, artificial_vars, has_geq_constraints = convert_to_standard_form(
        problem_type, objective, constraints, variables
    )

    latex_lines = [r"\text{Forma padrão:} \\"]
    latex_lines.append(r"\begin{align*}")
    obj_prefix = r'\text{Min } -Z' if problem_type == 'Max' else r'\text{Min } Z'
    latex_lines.append(f"{obj_prefix} &= {new_objective} \\\\")
    latex_lines.append(r'\text{sujeito a:}')
    for constraint in new_constraints:
        latex_lines.append(f"&\\quad {constraint} \\\\")
    var_list = ', '.join(all_variables)
    latex_lines.append(f"&\\quad {var_list} \\geq 0")
    latex_lines.append(r"\end{align*}")
    output_latex.append('\n'.join(latex_lines))

    basic_vars, tableau, needs_two_phase, constraint_coeffs, right_hand_sides, artificial_indices = build_initial_tableau(
        problem_type, new_objective, new_constraints, all_variables, slack_vars, artificial_vars, has_geq_constraints
    )

    output_latex.append(r"\text{Tableau inicial:} \\")
    output_latex.append(format_tableau_latex(basic_vars, tableau, all_variables, is_phase_one=False))

    if needs_two_phase:
        output_latex.append(r"\text{Necessária a resolução em duas fases devido a variáveis artificiais ou restrições do tipo } \geq. \\")
        result = two_phase_simplex(
            basic_vars, tableau, all_variables, artificial_vars, 
            parse_expression(new_objective, all_variables), 'Min', 
            artificial_indices, objective, new_constraints
        )
    else:
        output_latex.append(r"\text{Resolvendo com Simplex padrão:} \\")
        initial_tableau = [[coeff for coeff in row] for row in tableau]
        initial_basic_vars = basic_vars.copy()
        basic_vars, tableau = canonicalize_tableau(basic_vars, tableau, all_variables)
        if initial_tableau == tableau and initial_basic_vars == basic_vars:
            output_latex.append(r"\text{O tableau inicial já está na forma canônica.} \\")
        else:
            output_latex.append(r"\text{Tableau após forma canônica:} \\")
            output_latex.append(format_tableau_latex(basic_vars, tableau, all_variables, is_phase_one=False))

        iteration = 1
        while True:
            entering_var, leaving_var, analysis, is_done, is_unbounded = analyze_tableau(
                basic_vars, tableau, all_variables, 'Min' if problem_type == 'Max' else 'Min', is_phase_one=False
            )
            output_latex.append(f"\\text{{Iteração {iteration}:}} \\\\")
            output_latex.append(analysis)

            if is_unbounded:
                output_latex.append(r"\text{Solução ilimitada.} \\")
                break

            if is_done:
                output_latex.append(get_solution(basic_vars, tableau, all_variables, problem_type, objective))
                break

            pivot_row = basic_vars.index(leaving_var)
            pivot_col = all_variables.index(entering_var)
            basic_vars, tableau = pivot_tableau(basic_vars, tableau, pivot_row, pivot_col, all_variables)
            output_latex.append(f"\\text{{Novo Tableau:}} \\")
            output_latex.append(format_tableau_latex(basic_vars, tableau, all_variables, is_phase_one=False))

            iteration += 1

        result = '\n'.join(output_latex)

    return result

# Example usage
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
    print()