import re
from IPython.display import display, Math
from fractions import Fraction


def parse_latex_problem(latex_input):
    """
    Parse a LaTeX-formatted linear programming problem into its components.

    This function extracts the problem type (maximization or minimization), objective
    function, constraints, and variables from a LaTeX string. It is designed for
    problems in standard form with equality constraints, suitable for the Simplex method.

    Args:
        latex_input (str): A LaTeX string containing the linear programming problem,
            including objective function, constraints, and non-negativity conditions.

    Returns:
        tuple: A 4-tuple containing:
            - problem_type (str): 'Max' or 'Min', indicating maximization or minimization.
            - objective (str): The objective function as a LaTeX string (e.g., '-x_1 - 3x_2').
            - constraints (list): List of constraint strings (e.g., ['x_1 - 2x_2 + x_3 = 4']).
            - variables (list): Sorted list of variable names (e.g., ['x_1', 'x_2', 'x_3', 'x_4']).

    Example:
        >>> latex_input = r"\\text{Min -Z} &= -x_1 - 3x_2 \\\\ \\text{sujeito a:}&\\quad x_1 - 2x_2 + x_3 = 4 \\\\ &\\quad x_1, x_2, x_3 \\geq 0"
        >>> parse_latex_problem(latex_input)
        ('Min', '-x_1 - 3x_2', ['x_1 - 2x_2 + x_3 = 4'], ['x_1', 'x_2', 'x_3'])

    Notes:
        - Created by Gabriel S. Delgado with assistance from Grok 3 by xAI.
        - Assumes constraints are equalities (standard form for Simplex).
        - Variables are extracted as 'x_i' where i is the index.
    """
    lines = [line.strip() for line in latex_input.split('\n') if line.strip()]
    problem_type = None
    objective = None
    constraints = []
    variables = set()

    for line in lines:
        if r'\text{Min' in line or r'\text{Max' in line:
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
            continue
        if constraint_section:
            if (r'\leq' in cleaned_line or r'\geq' in cleaned_line or r'=' in cleaned_line) and not (r'\geq 0' in cleaned_line and 'x_' in cleaned_line and '+' not in cleaned_line and '-' not in cleaned_line):
                constraints.append(cleaned_line)
                vars_in_const = re.findall(r'x_(\d+)', cleaned_line)
                variables.update([f'x_{i}' for i in vars_in_const])

    constraints = [re.sub(r'\\+$', '', c).strip() for c in constraints]

    # Debugging prints (commented out for production)
    # print(f"Parsed objective: {objective}")
    # print(f"Parsed constraints: {constraints}")
    # print(f"Parsed variables: {variables}")
    return problem_type, objective, constraints, sorted(list(variables), key=lambda x: int(x.split('_')[1]))


def parse_expression(expr, all_variables):
    """
    Parse a LaTeX expression into coefficients for all variables.

    This function converts a LaTeX mathematical expression (e.g., '-x_1 - 3x_2') into
    a list of coefficients corresponding to the provided variables, ensuring each
    variable has an associated coefficient (0 if not present).

    Args:
        expr (str): A LaTeX string representing a mathematical expression.
        all_variables (list): List of all variable names (e.g., ['x_1', 'x_2', 'x_3']).

    Returns:
        list: List of coefficients in the order of all_variables.

    Example:
        >>> parse_expression('-x_1 - 3x_2', ['x_1', 'x_2', 'x_3'])
        [-1, -3, 0]

    Notes:
        - Created by Gabriel S. Delgado with assistance from Grok 3 by xAI.
        - Handles implicit coefficients (e.g., '-x_1' implies -1).
        - Returns 0 for variables not present in the expression.
    """
    expr = expr.replace(' + ', '+').replace(' - ', '-').replace('+ -', '-')
    if not expr.startswith('-') and not expr.startswith('+'):
        expr = '+' + expr
    terms = re.split(r'(?=[-+])', expr)
    terms = [term.strip() for term in terms if term.strip()]

    coefficients = {var: 0 for var in all_variables}

    for term in terms:
        match = re.match(r'([+-]?)(\d+)?x_(\d+)', term.replace(' ', ''))
        if match:
            sign = match.group(1)
            coeff_str = match.group(2)
            var_index = match.group(3)
            var = f'x_{var_index}'
            coeff = int(coeff_str) if coeff_str else 1
            if sign == '-':
                coeff = -coeff
            if var in coefficients:
                coefficients[var] = coeff

    # Debugging print (commented out for production)
    # print(f"Parsed expression '{expr}': {coefficients}")
    return [coefficients[var] for var in all_variables]


def build_simplex_tableau(problem_type, objective, constraints, variables):
    """
    Build the initial Simplex tableau using existing variables as basic variables.

    This function constructs the Simplex tableau from the objective function and
    constraints, selecting basic variables based on coefficients (preferring +1).

    Args:
        problem_type (str): 'Max' or 'Min', indicating the problem type.
        objective (str): The objective function as a LaTeX string.
        constraints (list): List of constraint strings (equalities).
        variables (list): List of all variable names.

    Returns:
        tuple: A 3-tuple containing:
            - basic_vars (list): List of basic variable names, including '-Z' for the objective.
            - tableau (list): The tableau as a list of lists (matrix).
            - variables (list): List of all variable names (unchanged).

    Example:
        >>> build_simplex_tableau('Min', '-x_1 - 3x_2', ['x_1 - 2x_2 + x_3 = 4'], ['x_1', 'x_2', 'x_3'])
        (['x_3', '-Z'], [[1, -2, 1, 4], [-1, -3, 0, 0]], ['x_1', 'x_2', 'x_3'])

    Notes:
        - Created by Gabriel S. Delgado with assistance from Grok 3 by xAI.
        - Assumes constraints are equalities with sufficient variables to form a basis.
        - Raises ValueError if no basic variable is found for a constraint.
    """
    obj_coeffs = parse_expression(objective, variables)
    if problem_type == 'Max':
        obj_coeffs = [-coeff for coeff in obj_coeffs]

    tableau = []
    basic_vars = []
    constraint_coeffs = []
    right_hand_sides = []

    # Parse constraints
    for constraint in constraints:
        left, right = constraint.split('=')
        left = left.strip()
        right = int(right.strip().split('\\')[0].strip())
        coeffs = parse_expression(left, variables)
        constraint_coeffs.append(coeffs)
        right_hand_sides.append(right)

    num_constraints = len(constraints)
    used_basic_vars = set()

    # Assign basic variables from existing variables
    for i in range(num_constraints):
        basic_var = None
        # Prioritize variables with +1 coefficient
        for j, var in enumerate(variables):
            if constraint_coeffs[i][j] == 1 and var not in used_basic_vars:
                basic_var = var
                used_basic_vars.add(var)
                break
        # If no +1 coefficient, choose any non-zero coefficient
        if not basic_var:
            for j, var in enumerate(variables):
                if constraint_coeffs[i][j] != 0 and var not in used_basic_vars:
                    basic_var = var
                    used_basic_vars.add(var)
                    break
        if basic_var:
            basic_vars.append(basic_var)
        else:
            raise ValueError(f"Não foi possível encontrar uma variável básica para a restrição {i+1}")

        tableau.append(constraint_coeffs[i] + [right_hand_sides[i]])

    # Objective row
    tableau.append(obj_coeffs + [0])
    basic_vars.append('-Z')

    print(f"Basic variables: {basic_vars}")
    print(f"Initial Tableau: {tableau}")
    return basic_vars, tableau, variables


def pivot_tableau(basic_vars, tableau, pivot_row, pivot_col, variables):
    """
    Pivot the Simplex tableau around the specified row and column.

    This function performs Gaussian elimination to update the tableau, making the
    pivot element 1 and setting other elements in the pivot column to 0.

    Args:
        basic_vars (list): List of current basic variables.
        tableau (list): The current Simplex tableau as a list of lists.
        pivot_row (int): Index of the pivot row.
        pivot_col (int): Index of the pivot column.
        variables (list): List of all variable names.

    Returns:
        tuple: A 2-tuple containing:
            - new_basic_vars (list): Updated list of basic variables.
            - new_tableau (list): Updated tableau after pivoting.

    Example:
        >>> pivot_tableau(['x_3', '-Z'], [[1, -2, 1, 4], [-1, -3, 0, 0]], 0, 0, ['x_1', 'x_2', 'x_3'])
        (['x_1', '-Z'], [[1, -2, 1, 4], [0, -5, 1, 4]])

    Notes:
        - Created by Gabriel S. Delgado with assistance from Grok 3 by xAI.
        - Uses Fraction for precise arithmetic to avoid floating-point errors.
    """
    new_tableau = [[Fraction(coeff) for coeff in row] for row in tableau]
    pivot = new_tableau[pivot_row][pivot_col]

    # Normalize pivot row
    for j in range(len(new_tableau[pivot_row])):
        new_tableau[pivot_row][j] /= pivot

    # Eliminate other rows
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

    This function ensures that each basic variable has a coefficient of 1 in its
    row and 0 elsewhere in its column, preparing the tableau for Simplex iterations.

    Args:
        basic_vars (list): List of current basic variables.
        tableau (list): The current Simplex tableau.
        variables (list): List of all variable names.

    Returns:
        tuple: A 2-tuple containing:
            - new_basic_vars (list): Updated list of basic variables.
            - new_tableau (list): Updated tableau in canonical form.

    Example:
        >>> canonicalize_tableau(['x_2', '-Z'], [[3, 1, 1, 9], [-2, -1, 0, 0]], ['x_1', 'x_2', 'x_3'])
        (['x_2', '-Z'], [[3, 1, 1, 9], [-2, -1, 0, 0]])

    Notes:
        - Created by Gabriel S. Delgado with assistance from Grok 3 by xAI.
        - Pivots only if the basic variable’s column is not canonical and the pivot element is non-zero.
    """
    new_tableau = tableau
    new_basic_vars = basic_vars

    for i in range(len(basic_vars) - 1):  # Exclude -Z
        basic_var = basic_vars[i]
        var_idx = variables.index(basic_var)
        is_canonical = True
        for r in range(len(tableau) - 1):
            expected = 1 if r == i else 0
            if tableau[r][var_idx] != expected:
                is_canonical = False
                break
        if not is_canonical and tableau[i][var_idx] != 0:
            new_basic_vars, new_tableau = pivot_tableau(new_basic_vars, new_tableau, i, var_idx, variables)

    return new_basic_vars, new_tableau


def format_tableau_latex(basic_vars, tableau, variables):
    """
    Format the Simplex tableau as a LaTeX string.

    This function generates a LaTeX table representing the Simplex tableau, including
    basic variables, coefficients, and right-hand sides, with fractions formatted properly.

    Args:
        basic_vars (list): List of basic variables, including '-Z' for the objective.
        tableau (list): The Simplex tableau as a list of lists.
        variables (list): List of all variable names.

    Returns:
        str: A LaTeX string representing the tableau.

    Example:
        >>> format_tableau_latex(['x_3', '-Z'], [[1, -2, 1, 4], [-1, -3, 0, 0]], ['x_1', 'x_2', 'x_3'])
        '\\[\\begin{array}{|c|ccc|c|}\\hline\\text{VB} & x_1 & x_2 & x_3 & \\text{b} \\\\ \\hline {x_3} & 1 & -2 & 1 & 4 \\\\ \\hline \\text{-Z} & -1 & -3 & 0 & 0 \\\\ \\hline\\end{array}\\]'

    Notes:
        - Created by Gabriel S. Delgado with assistance from Grok 3 by xAI.
        - Uses \\sfrac for fractions to ensure proper rendering.
    """
    num_vars = len(variables)
    col_spec = '|c|' + 'c' * num_vars + '|c|'

    latex_lines = [r'\[']
    latex_lines.append(r'\begin{array}{' + col_spec + '}')
    latex_lines.append(r'\hline')

    header = [r'\text{VB}'] + [var for var in variables] + [r'\text{b}']
    latex_lines.append(' & '.join(header) + r' \\')
    latex_lines.append(r'\hline')

    for i, row in enumerate(tableau[:-1]):
        row_str = [f'{{{basic_vars[i]}}}' if basic_vars[i].startswith('x_') else basic_vars[i]]
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

    obj_row = [f'\\text{{{basic_vars[-1]}}}']
    for coeff in tableau[-1]:
        if isinstance(coeff, Fraction):
            if coeff.denominator == 1:
                obj_row.append(str(coeff.numerator))
            else:
                obj_row.append(f'\\sfrac{{{coeff.numerator}}}{{{coeff.denominator}}}')
        else:
            obj_row.append(str(coeff))
    latex_lines.append(' & '.join(obj_row) + r' \\')

    latex_lines.append(r'\hline')
    latex_lines.append(r'\end{array}')
    latex_lines.append(r'\]')

    return '\n'.join(latex_lines)


def analyze_tableau(basic_vars, tableau, variables):
    """
    Analyze the Simplex tableau to determine entering and leaving variables.

    This function identifies the entering variable (most negative objective coefficient),
    computes pivot ratios, and selects the leaving variable (minimum positive ratio).
    It also checks for optimality or unboundedness.

    Args:
        basic_vars (list): List of current basic variables.
        tableau (list): The current Simplex tableau.
        variables (list): List of all variable names.

    Returns:
        tuple: A 4-tuple containing:
            - entering_var (str or None): The variable entering the basis, or None if optimal/unbounded.
            - leaving_var (str or None): The variable leaving the basis, or None if unbounded.
            - analysis_latex (str): LaTeX string describing the analysis (entering/leaving variables, ratios).
            - is_optimal (bool): True if the solution is optimal or unbounded, False otherwise.

    Example:
        >>> analyze_tableau(['x_3', '-Z'], [[1, -2, 1, 4], [-1, -3, 0, 0]], ['x_1', 'x_2', 'x_3'])
        ('x_2', 'x_3', '...', False)

    Notes:
        - Created by Gabriel S. Delgado with assistance from Grok 3 by xAI.
        - Highlights the minimum ratio in red in the LaTeX output.
        - Returns a message for unbounded solutions if no positive pivot coefficients exist.
    """
    obj_row = tableau[-1][:-1]
    non_basic_vars = [var for var in variables if var not in basic_vars[:-1]]
    is_optimal = True
    for var in non_basic_vars:
        idx = variables.index(var)
        if obj_row[idx] < 0:
            is_optimal = False
            break
    if is_optimal:
        return None, None, r"Nenhum coeficiente negativo para variáveis não básicas na linha do objetivo. Solução ótima alcançada.", True

    min_coeff = min(obj_row)
    entering_idx = obj_row.index(min_coeff)
    entering_var = variables[entering_idx]

    ratios = []
    ratio_strings = []
    for i, row in enumerate(tableau[:-1]):
        pivot_coeff = row[entering_idx]
        b_value = row[-1]
        if pivot_coeff > 0:
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

    if not ratios:
        return None, None, r"\text{Solução ilimitada: nenhum coeficiente positivo na coluna pivô.}", True

    min_ratio, leaving_idx = min(ratios, key=lambda x: x[0])
    leaving_var = basic_vars[leaving_idx]

    formatted_ratio_strings = []
    for i in range(len(ratio_strings)):
        if i == leaving_idx:
            formatted_ratio_strings.append(f"{{ \\color{{red}}{{{ratio_strings[i]}}} }}")
        else:
            formatted_ratio_strings.append(ratio_strings[i])

    analysis_latex = [
        f"\\text{{Candidato a entrar na base: }} \\({entering_var}\\).\\\\",
        "\\text{Teste da razão para verificar quem sai da base:}\\\\",
        f"\\( {', '.join(formatted_ratio_strings)} \\)\\\\",
        f"\\text{{Candidato a sair da base: }} \\({leaving_var}\\).\\\\"
    ]

    return entering_var, leaving_var, '\n'.join(analysis_latex), False


def get_solution(basic_vars, tableau, variables, problem_type):
    """
    Extract the optimal solution from the Simplex tableau and check for alternate optima.

    This function computes the values of all variables, the objective value, and checks
    if there are alternate optimal solutions (non-basic variables with zero objective coefficients).

    Args:
        basic_vars (list): List of current basic variables.
        tableau (list): The final Simplex tableau.
        variables (list): List of all variable names.
        problem_type (str): 'Max' or 'Min', indicating the problem type.

    Returns:
        str: A LaTeX string describing the optimal solution and any alternate optima.

    Example:
        >>> get_solution(['x_1', 'x_3', '-Z'], [[1, 0, 1, 2], [0, 1, 0, 1], [0, 0, 0, 5]], ['x_1', 'x_2', 'x_3'], 'Min')
        'Como não temos mais valores de custo negativo, a solução atual, \\(z=-5\\), é ótima. A solução \\(\\left(2,0,1\\right)^T\\) é a única partição básica ótima.'

    Notes:
        - Created by Gabriel S. Delgado with assistance from Grok 3 by xAI.
        - Correctly handles objective value for minimization (negates tableau’s -Z).
        - Flags alternate optima if non-basic variables have zero coefficients.
    """
    solution = {var: 0 for var in variables}
    for i, var in enumerate(basic_vars[:-1]):
        if var in solution:
            solution[var] = tableau[i][-1]

    # For minimization, the tableau gives -Z, so negate it to get Z
    z_value = -tableau[-1][-1] if problem_type == 'Max' else -tableau[-1][-1]

    non_basic_vars = [var for var in variables if var not in basic_vars[:-1]]
    alternate_optima = False
    zero_coeff_vars = []
    for var in non_basic_vars:
        idx = variables.index(var)
        if tableau[-1][idx] == 0:
            alternate_optima = True
            zero_coeff_vars.append(var)

    solution_values = [solution.get(var, 0) for var in variables]
    solution_latex = [
        f"Como não temos mais valores de custo negativo, a solução atual,  \\(z={z_value}\\), é ótima. "
    ]
    if alternate_optima:
        solution_latex.append(f"Note que o custo de \\({', '.join(zero_coeff_vars)}\\) é zero, indicando que \\(\\left({','.join(str(v) for v in solution_values)}\\right)^T\\) {{\\color{{red}}{{não}}}} é a única partição básica ótima.")
    else:
        solution_latex.append(f"A solução \\(\\left({','.join(str(v) for v in solution_values)}\\right)^T\\) é a única partição básica ótima.")

    return '\n'.join(solution_latex)


def solve_simplex(latex_input):
    """
    Solve a linear programming problem using the Simplex method.

    This function processes a LaTeX-formatted problem, builds and iterates the Simplex
    tableau, and outputs the steps and optimal solution as a LaTeX string. It assumes
    the problem is in standard form with equality constraints.

    Args:
        latex_input (str): A LaTeX string containing the linear programming problem.

    Returns:
        str: A LaTeX string with all Simplex iterations, tableaus, and the final solution.

    Example:
        >>> latex_input = r"\\text{Min -Z} &= -x_1 - 3x_2 \\\\ \\text{sujeito a:}&\\quad x_1 - 2x_2 + x_3 = 4 \\\\ &\\quad -x_1 + x_2 + x_4 = 3 \\\\ &\\quad x_1, x_2, x_3, x_4 \\geq 0"
        >>> solve_simplex(latex_input)
        '\\text{Tableau inicial:}\\n[...]\\nComo não temos mais valores de custo negativo, a solução atual, \\(z=...\\), é ótima. ...'

    Notes:
        - Created by Gabriel S. Delgado with assistance from Grok 3 by xAI.
        - Outputs detailed LaTeX for each iteration, including tableaus and pivot decisions.
        - Handles trivial solutions and checks for optimality or unboundedness.
    """
    problem_type, objective, constraints, variables = parse_latex_problem(latex_input)
    basic_vars, tableau, all_variables = build_simplex_tableau(problem_type, objective, constraints, variables)

    output_latex = [r"\text{Tableau inicial:}"]
    output_latex.append(format_tableau_latex(basic_vars, tableau, all_variables))

    # Canonicalize the tableau
    basic_vars, tableau = canonicalize_tableau(basic_vars, tableau, all_variables)
    output_latex.append(r"Tableau após pivoteamento inicial para forma canônica:")
    output_latex.append(format_tableau_latex(basic_vars, tableau, all_variables))

    # Check for trivial solution
    right_hand_sides = [row[-1] for row in tableau[:-1]]
    if all(rhs == 0 for rhs in right_hand_sides):
        non_basic_vars = [var for var in all_variables if var not in basic_vars[:-1]]
        obj_row = tableau[-1][:-1]
        is_optimal = True
        for var in non_basic_vars:
            idx = all_variables.index(var)
            if obj_row[idx] < 0:
                is_optimal = False
                break
        if is_optimal:
            output_latex.append(r"Nenhum pivoteamento adicional necessário. Solução trivial é ótima.")
            output_latex.append(get_solution(basic_vars, tableau, all_variables, problem_type))
            return '\n'.join(output_latex)

    iteration = 1
    while True:
        entering_var, leaving_var, analysis, is_done = analyze_tableau(basic_vars, tableau, all_variables)
        output_latex.append(f"\n\\text{{Iteração {iteration}:}}")
        output_latex.append(analysis)

        if is_done:
            if entering_var is None and "ótima" in analysis:
                output_latex.append(get_solution(basic_vars, tableau, all_variables, problem_type))
            break

        pivot_row = basic_vars.index(leaving_var)
        pivot_col = all_variables.index(entering_var)
        basic_vars, tableau = pivot_tableau(basic_vars, tableau, pivot_row, pivot_col, all_variables)
        output_latex.append(f"\n\\text{{Novo Tableau:}}")
        output_latex.append(format_tableau_latex(basic_vars, tableau, all_variables))

        iteration += 1

    return '\n'.join(output_latex)


# Example usage in Google Colab
latex_input = r"""
    \displaystyle \text{Min -Z} &= -x_1 - 3x_2 \\
    \text{sujeito a:} 
    &\quad x_1 - 2x_2 + x_3 = 4 \\
    &\quad -x_1 + x_2 + x_4 = 3 \\
    &\quad x_1, x_2, x_3, x_4 \geq 0
"""

# Solve the Simplex problem
result = solve_simplex(latex_input)

# Display result
display(Math(result))