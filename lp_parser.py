import re
from IPython.display import display, Math


def parse_latex_problem(latex_input):
    """
    Parse a LaTeX-formatted linear programming problem into its components.

    This function extracts the problem type (maximization or minimization), objective
    function, constraints, and variables from a LaTeX string. It supports objective
    functions and constraints with standard mathematical notation, including equalities
    and inequalities.

    Args:
        latex_input (str): A LaTeX string containing the linear programming problem,
            including objective function, constraints, and non-negativity conditions.

    Returns:
        tuple: A 4-tuple containing:
            - problem_type (str): 'Max' or 'Min', indicating maximization or minimization.
            - objective (str): The objective function as a LaTeX string (e.g., '500x_1 + 300x_2').
            - constraints (list): List of constraint strings (e.g., ['15x_1 + 5x_2 \\leq 300']).
            - variables (list): Sorted list of variable names (e.g., ['x_1', 'x_2']).

    Example:
        >>> latex_input = r"\\text{Max Z} &= 500x_1 + 300x_2 \\\\ \\text{sujeito a:}&\\quad 15x_1 + 5x_2 \\leq 300 \\\\ &\\quad x_1, x_2 \\geq 0"
        >>> parse_latex_problem(latex_input)
        ('Max', '500x_1 + 300x_2', ['15x_1 + 5x_2 \\leq 300'], ['x_1', 'x_2'])

    Notes:
        - Created by Gabriel S. Delgado with assistance from Grok 3 by xAI.
        - Supports constraints with \\leq, \\geq, or =, and upper bound constraints (e.g., x_i \\leq u_i).
        - Variables are extracted as 'x_i' where i is the index.
    """
    # Remove extra whitespace and split lines
    lines = [line.strip() for line in latex_input.split('\n') if line.strip()]

    # Initialize variables
    problem_type = None
    objective = None
    constraints = []
    variables = set()

    # Parse objective function
    for line in lines:
        if r'\text{Max' in line or r'\text{Min' in line:
            if r'\text{Max' in line:
                problem_type = 'Max'
            else:
                problem_type = 'Min'
            # Remove & and \quad if present
            line = re.sub(r'&\s*\\quad\s*', '', line)
            obj_match = re.search(r'&=([^\\]+)\\', line)
            if obj_match:
                objective = obj_match.group(1).strip()
                # Find variables in objective
                vars_in_obj = re.findall(r'x_(\d+)', objective)
                variables.update([f'x_{i}' for i in vars_in_obj])
            break

    # Parse constraints
    constraint_section = False
    for line in lines:
        # Clean line by removing & and \quad
        cleaned_line = re.sub(r'&\s*\\quad\s*', '', line).strip()
        if r'\text{sujeito a:}' in line:
            constraint_section = True
            # Check if there's a constraint on the same line
            if cleaned_line != r'\text{sujeito a:}':
                # Extract constraint after \text{sujeito a:}
                constraint_part = cleaned_line.replace(r'\text{sujeito a:}', '').strip()
                if constraint_part and (r'\leq' in constraint_part or r'\geq' in constraint_part or r'=' in constraint_part):
                    constraints.append(constraint_part)
                    vars_in_const = re.findall(r'x_(\d+)', constraint_part)
                    variables.update([f'x_{i}' for i in vars_in_const])
            continue
        if constraint_section:
            # Handle upper bound constraints like 0 \leq x_i \leq u_i or x_i \leq u_i
            upper_bound_match = re.match(r'(0\s*\\leq\s*)?x_(\d+)\s*\\leq\s*(\d+)', cleaned_line)
            if upper_bound_match:
                var_index = upper_bound_match.group(2)
                upper_bound = upper_bound_match.group(3)
                # Add as a regular constraint: x_i \leq u_i
                constraint = f"x_{var_index} \\leq {upper_bound}"
                constraints.append(constraint)
                variables.add(f'x_{var_index}')
            # Handle standard constraints with \leq, \geq, or =
            elif (r'\leq' in cleaned_line or r'\geq' in cleaned_line or r'=' in cleaned_line) and not (r'\geq 0' in cleaned_line and 'x_' in cleaned_line and '+' not in cleaned_line and '-' not in cleaned_line):
                constraints.append(cleaned_line)
                # Find variables in constraint
                vars_in_const = re.findall(r'x_(\d+)', cleaned_line)
                variables.update([f'x_{i}' for i in vars_in_const])

    # Clean constraints by removing trailing \\
    constraints = [re.sub(r'\\+$', '', c).strip() for c in constraints]

    # Debugging prints (commented out for production)
    # print(f"Parsed objective: {objective}")
    # print(f"Parsed constraints: {constraints}")
    # print(f"Parsed variables: {variables}")
    return problem_type, objective, constraints, sorted(list(variables), key=lambda x: int(x.split('_')[1]))


def parse_expression(expr):
    """
    Parse a LaTeX mathematical expression into coefficients and variables.

    This function processes a LaTeX expression (e.g., '-500x_1 + -300x_2') and
    extracts the coefficients and corresponding variable names as tuples.

    Args:
        expr (str): A LaTeX string representing a mathematical expression with
            variables (e.g., '500x_1 - 300x_2').

    Returns:
        list: A list of tuples, each containing:
            - coeff (int): The coefficient of the variable (including sign).
            - var (str): The variable name (e.g., 'x_1').

    Example:
        >>> parse_expression('500x_1 - 300x_2')
        [(500, 'x_1'), (-300, 'x_2')]

    Notes:
        - Created by Gabriel S. Delgado with assistance from Grok 3 by xAI.
        - Handles implicit coefficients (e.g., '+x_1' implies +1, '-x_2' implies -1).
        - Normalizes expressions by handling signs and spaces consistently.
    """
    # Replace spaces around + and - for consistent splitting
    expr = expr.replace(' + ', '+').replace(' - ', '-')
    # Normalize + - to -
    expr = expr.replace('+ -', '-')
    # Add a leading + for first term if no sign
    if not expr.startswith('-') and not expr.startswith('+'):
        expr = '+' + expr
    # Split on + or - (keeping the sign with the term)
    terms = re.split(r'(?=[-+])', expr)
    terms = [term.strip() for term in terms if term.strip()]
    result = []

    for term in terms:
        # Handle coefficient and variable
        # Match patterns like +500x_1, -300x_2, +x_3, -x_4
        match = re.match(r'([+-]?)(\d+)?x_(\d+)', term.replace(' ', ''))
        if match:
            sign = match.group(1)
            coeff_str = match.group(2)
            var_index = match.group(3)
            # Determine coefficient
            coeff = int(coeff_str) if coeff_str else 1
            if sign == '-':
                coeff = -coeff
            var = f'x_{var_index}'
            result.append((coeff, var))

    # Debugging print (commented out for production)
    # print(f"Parsed expression '{expr}': {result}")
    return result


def convert_to_standard_form(problem_type, objective, constraints, variables):
    """
    Convert a linear programming problem to standard Simplex form.

    This function transforms the problem by converting maximization to minimization,
    adding slack/excess/artificial variables for inequalities, and reformatting
    constraints as equalities.

    Args:
        problem_type (str): 'Max' or 'Min', indicating the problem type.
        objective (str): The objective function as a LaTeX string.
        constraints (list): List of constraint strings.
        variables (list): List of variable names.

    Returns:
        tuple: A 3-tuple containing:
            - new_objective (str): The objective function in standard form.
            - new_constraints (list): List of constraints as equalities.
            - all_variables (list): List of all variables, including slack/excess/artificial.

    Example:
        >>> convert_to_standard_form('Max', '500x_1 + 300x_2', ['15x_1 + 5x_2 \\leq 300'], ['x_1', 'x_2'])
        ('-500x_1 - 300x_2', ['15x_1 + 5x_2 + x_3 = 300'], ['x_1', 'x_2', 'x_3'])

    Notes:
        - Created by Gabriel S. Delgado with assistance from Grok 3 by xAI.
        - For maximization, inverts objective coefficients.
        - Adds slack variables for \\leq, excess and artificial variables for \\geq.
        - Equality constraints remain unchanged.
    """
    # Initialize
    new_objective = objective
    new_constraints = []
    slack_var_count = len(variables) + 1  # Start slack variables after original ones
    all_variables = variables.copy()

    # Convert objective for minimization
    if problem_type == 'Max':
        # Invert coefficients for minimization
        terms = parse_expression(objective)
        new_terms = []
        for coeff, var in terms:
            new_coeff = -coeff
            term_str = f"{new_coeff}x_{var.split('_')[1]}" if new_coeff != 1 and new_coeff != -1 else \
                       f"-x_{var.split('_')[1]}" if new_coeff == -1 else f"x_{var.split('_')[1]}"
            new_terms.append(term_str)
        new_objective = ' + '.join(new_terms).replace('+ -', '- ')
    else:
        # For Min, keep objective but clean up + - to -
        terms = parse_expression(objective)
        new_terms = []
        for coeff, var in terms:
            term_str = f"{coeff}x_{var.split('_')[1]}" if coeff != 1 and coeff != -1 else \
                       f"-x_{var.split('_')[1]}" if coeff == -1 else f"x_{var.split('_')[1]}"
            new_terms.append(term_str)
        new_objective = ' + '.join(new_terms).replace('+ -', '- ')

    # Process constraints
    for constraint in constraints:
        # Remove & and \quad for parsing
        constraint = re.sub(r'&\s*\\quad\s*', '', constraint)
        # Check if it's an upper bound constraint (e.g., x_i \leq u_i)
        upper_bound_match = re.match(r'x_(\d+)\s*\\leq\s*(\d+)', constraint)
        if upper_bound_match:
            var_index = upper_bound_match.group(1)
            upper_bound = upper_bound_match.group(2)
            slack_var = f'x_{slack_var_count}'
            all_variables.append(slack_var)
            new_constraint = f"x_{var_index} + {slack_var} = {upper_bound}"
            new_constraints.append(new_constraint)
            slack_var_count += 1
            continue

        # Split into left and right sides for standard constraints
        if r'\leq' in constraint:
            left, right = constraint.split(r'\leq')
            ineq_type = 'leq'
        elif r'\geq' in constraint:
            left, right = constraint.split(r'\geq')
            ineq_type = 'geq'
        else:
            left, right = constraint.split('=')
            ineq_type = 'eq'
        left = left.strip()
        right = right.strip().split('\\')[0].strip()

        # Parse left-hand side
        terms = parse_expression(left)
        new_left = ' + '.join([f"{coeff}x_{var.split('_')[1]}" if coeff != 1 and coeff != -1 else \
                               f"-x_{var.split('_')[1]}" if coeff == -1 else f"x_{var.split('_')[1]}" \
                               for coeff, var in terms])
        new_left = new_left.replace('+ -', '- ')

        # Add slack/excess/artificial variables
        if ineq_type == 'leq':
            slack_var = f'x_{slack_var_count}'
            all_variables.append(slack_var)
            new_constraint = f"{new_left} + {slack_var} = {right}"
            slack_var_count += 1
        elif ineq_type == 'geq':
            excess_var = f'x_{slack_var_count}'
            all_variables.append(excess_var)
            artificial_var = f'x_{slack_var_count + 1}'
            all_variables.append(artificial_var)
            new_constraint = f"{new_left} - {excess_var} + {artificial_var} = {right}"
            slack_var_count += 2
        else:  # Equality
            new_constraint = f"{new_left} = {right}"

        new_constraints.append(new_constraint)

    # Debugging prints (commented out for production)
    # print(f"New constraints: {new_constraints}")
    # print(f"All variables: {all_variables}")
    return new_objective, new_constraints, sorted(all_variables, key=lambda x: int(x.split('_')[1]))


def format_latex_output(problem_type, objective, constraints, variables):
    """
    Format the linear programming problem in standard form as a LaTeX string.

    This function generates a LaTeX representation of the problem, including the
    objective function, constraints, and non-negativity conditions, suitable for display.

    Args:
        problem_type (str): 'Max' or 'Min', indicating the problem type.
        objective (str): The objective function as a LaTeX string.
        constraints (list): List of constraint strings in standard form.
        variables (list): List of all variable names.

    Returns:
        str: A LaTeX string representing the formatted problem.

    Example:
        >>> format_latex_output('Max', '-500x_1 - 300x_2', ['15x_1 + 5x_2 + x_3 = 300'], ['x_1', 'x_2', 'x_3'])
        '\\text{Min -Z} &= -500x_1 - 300x_2 \\\\\n\\text{sujeito a:}\n&\quad 15x_1 + 5x_2 + x_3 = 300 \\\\\n&\quad x_1, x_2, x_3 \\geq 0'

    Notes:
        - Created by Gabriel S. Delgado with assistance from Grok 3 by xAI.
        - Outputs LaTeX suitable for rendering in environments like Jupyter or Overleaf.
    """
    latex_lines = []

    # Objective function
    obj_prefix = r'\text{Min -Z}' if problem_type == 'Max' else r'\text{Min Z}'
    latex_lines.append(f"{obj_prefix} &= {objective} \\\\")

    # Constraints
    latex_lines.append(r'\text{sujeito a:} ')
    for constraint in constraints:
        latex_lines.append(f"&\quad {constraint} \\\\")

    # Non-negativity
    var_list = ', '.join(variables)
    latex_lines.append(f"&\quad {var_list} \\geq 0")

    return '\n'.join(latex_lines)


def process_linear_program(latex_input):
    """
    Process a LaTeX linear programming problem and convert it to standard form.

    This function serves as the main entry point, parsing the input, converting it
    to standard form, and formatting the output as LaTeX.

    Args:
        latex_input (str): A LaTeX string containing the linear programming problem.

    Returns:
        str: A LaTeX string representing the problem in standard form.

    Example:
        >>> latex_input = r"\\text{Max Z} &= 500x_1 + 300x_2 \\\\ \\text{sujeito a:}&\\quad 15x_1 + 5x_2 \\leq 300 \\\\ &\\quad x_1, x_2 \\geq 0"
        >>> process_linear_program(latex_input)
        '\\text{Min -Z} &= -500x_1 - 300x_2 \\\\\n\\text{sujeito a:}\n&\quad 15x_1 + 5x_2 + x_3 = 300 \\\\\n&\quad x_1, x_2, x_3 \\geq 0'

    Notes:
        - Created by Gabriel S. Delgado with assistance from Grok 3 by xAI.
        - Combines parsing, conversion, and formatting for Simplex preparation.
    """
    # Parse the input
    problem_type, objective, constraints, variables = parse_latex_problem(latex_input)

    # Convert to standard form
    new_objective, new_constraints, all_variables = convert_to_standard_form(
        problem_type, objective, constraints, variables
    )

    # Format output
    latex_output = format_latex_output(problem_type, new_objective, new_constraints, all_variables)

    return latex_output


# Example usage in Google Colab
from IPython.display import display, Math

latex_inputs = [
    r"""
    \text{Max Z} &= 500x_1 + 300x_2 \\
    \text{sujeito a:}&\quad15x_1 + 5x_2 \leq 300 \\
    &\quad 10x_1 + 6x_2 \leq 240 \\
    &\quad 8x_1 + 12x_2 \leq 450 \\
    &\quad x_1, x_2 \geq 0
    """
    # Additional example (commented out)
    # r"""
    # \text{Min Z} &= -2x_1 - x_2 \\
    # \text{sujeito a:}&\quad 3x_1 + x_2 \leq 9 \\
    # &\quad 2x_1 - 2x_2 \leq 3 \\
    # &\quad 0 \leq x_1 \leq 1 \\
    # &\quad 0 \leq x_2 \leq 8
    # """,
]

# Process each problem and display results
for i, latex_input in enumerate(latex_inputs):
    # print(f"\nProcessing Problem {i+1}")
    result = process_linear_program(latex_input)
    display(Math(f'\\text{{Problem }} {i+1}:'))
    display(Math(result))