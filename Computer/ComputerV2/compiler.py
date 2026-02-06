import re

def is_number(num):
    return num.lstrip('-+').isdigit()

def lex(code):
    patterns = [
        ('NEWLINE', r'\n+'),
        ('COMMENT', r'#'),
        ('OP',      r'\+|\-|=='),
        ('EQ',      r'='),
        ('DEF',     r':'),
        ('NUM',     r'\d+'),
        ('KEYWORD', r'if'),
        ('BOOLEAN', r'true|false'),
        ('P_LEFT',  r'\('),
        ('P_RIGHT', r'\)'),
        ('B_LEFT',  r'\{'),
        ('B_RIGHT', r'\}'),
        ('ACC',     r'@'),
        ('ID',      r'[a-zA-Z_][a-zA-Z0-9_]*'),
        ('SKIP',    r'\s+'),
    ]
    tokens = []
    line_num = 1

    for match in re.finditer('|'.join(f'(?P<{pair[0]}>{pair[1]})' for pair in patterns), code):
        kind = match.lastgroup
        value = match.group()
        
        if kind == 'NEWLINE':
            tokens.append(('NEWLINE', value, line_num))
            line_num += len(value)
        elif kind == 'SKIP':
            continue
        else:
            tokens.append((kind, value, line_num))

    return tokens

class BinOpNode:
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right
    def view(self, level=0):
        return "  " * level + f"{self.op}\n{self.view_child(self.left, level+1)}\n{self.view_child(self.right, level+1)}"
    def view_child(self, child, level):
        if hasattr(child, 'view'): return child.view(level)
        return ("  " * level) + str(child)
    def get_weight(self):
        left_w = 1 + self.left.get_weight() if isinstance(self.left, BinOpNode) else 1
        right_w = 1 + self.right.get_weight() if isinstance(self.right, BinOpNode) else 1
        
        # if left_w == right_w:
        #     return left_w + 1
        return max(left_w, right_w)

class DeclNode:
    def __init__(self, name, dtype, value):
        self.name = name
        self.dtype = dtype
        self.value = value
    def view(self, level=0):
        val_str = self.value.view(level + 1) if hasattr(self.value, 'view') else ("  " * (level+1)) + str(self.value)
        return "  " * level + f"DECL {self.name}:{self.dtype} =\n{val_str}"

class AssignNode:
    def __init__(self, name, dtype, value):
        self.name = name
        self.dtype = dtype
        self.value = value
    def view(self, level=0):
        val_str = self.value.view(level + 1) if hasattr(self.value, 'view') else ("  " * (level+1)) + str(self.value)
        return "  " * level + f"ASSIGN {self.name}:{self.dtype} =\n{val_str}"

class IfNode:
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body
    def view(self, level=0):
        indent = "  " * level
        out = f"{indent}IF:\n"
        condition = self.condition.view(level + 2) if hasattr(self.condition, 'view') else self.condition
        out += f"{indent}  COND:\n{condition}\n"
        out += f"{indent}  BODY:\n"
        for statement in self.body.statements:
            out += statement.view(level + 2) + "\n"
        return out.rstrip()

class BlockNode:
    def __init__(self, statements):
        self.statements = statements

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

        self.variables = {}

    def consume(self, expected_type):
        token = self.peek()
        print("consuming", token, expected_type)
        token_type = token[0]
        if token_type != expected_type:
            raise SyntaxError(f"Expected {expected_type}, got {token_type} in line {token[2]}")

        self.pos += 1
        return token

    def peek(self, offset=0):
        pos = self.pos + offset
        if pos < len(self.tokens):
            return self.tokens[pos if pos else self.pos]
        else: 
            return None

    def consume_val(self):
        val_type = self.peek()[0]
        if val_type == 'NUM':
            return self.consume('NUM')[1]
        elif val_type == 'ACC':
            return self.consume('ACC')[1]
        elif val_type == 'ID':
            return self.consume('ID')[1]
        elif val_type == 'BOOLEAN':
            return self.consume('BOOLEAN')[1]
        else:
            print("INVALID VAL ")
            return None

    def is_val(self, offset=0):
        val_type = self.peek(offset)[0]
        return val_type == 'NUM' or val_type == 'ID' or val_type == 'BOOLEAN'
        
    def parse_expression(self, left=None):
        print("PARSING EXPRESSION", "left", left, self.peek())
        current_token = self.peek()[0]
        if current_token and current_token == "P_LEFT":
            self.consume('P_LEFT')
            print("WHAT THE HELL", self.peek(), self.peek(1))
            expression = self.parse_expression()
            self.consume('P_RIGHT')
            
            # if left is True leading minus
            if left:
                expression = BinOpNode("0", "-", expression)
                
            return self.parse_expression(expression)
            
        if not left:
            # check if leading minus
            if current_token == "OP":
                op = self.peek()[1]
                # only works if val is direct number
                if op == "-":
                    op = self.consume("OP")[1]
                    next_token = self.peek()
                    if next_token and next_token[0] == "NUM":
                        num = self.consume("NUM")[1]
                        return self.parse_expression(op + num)
                    return self.parse_expression(True)
                elif op == "+":
                    self.consume("OP")
                    return self.parse_expression()
                else:
                    raise SyntaxError("Operator before argument", op)
                    
            left = self.consume_val()

        potential_op = self.peek()[0]
        if potential_op and potential_op == 'OP':
            op = self.consume('OP')[1]

            if self.is_val():
                right = self.consume_val()
                combined = BinOpNode(left, op, right)

                return self.parse_expression(combined)
            else:
                current_token = self.peek()[0]
                if current_token and current_token == "P_LEFT":
                    self.consume('P_LEFT')
                    expression = self.parse_expression()
                    self.consume('P_RIGHT')
                    combined = BinOpNode(left, op, expression)
                    return self.parse_expression(combined)
        else:
            return left

    def parse_statement(self):
        token = self.peek()
        token_type = token[0]
        print('parsing', token)

        if token_type == 'KEYWORD':
            keyword = self.consume('KEYWORD')
            
            # if statement
            if keyword[1] == 'if':
                self.consume('P_LEFT')
                cond = self.parse_expression()
                self.consume('P_RIGHT')

                block = self.parse_block()

                return IfNode(cond, block)
        elif token_type == 'ID' or token_type == 'ACC':
            name = self.consume('ID')[1] if token_type == 'ID' else self.consume('ACC')[1]

            # variable declaration
            if self.peek()[0] == 'DEF':
                self.consume('DEF')
                var_type = self.consume('ID')[1]
                self.variables[name] = var_type
                print("ASSIGNED", self.variables)

                # just declare, no init
                if self.peek() and self.peek()[0] == "NEWLINE":
                    return DeclNode(name, var_type, None)

                self.consume('EQ')
                exp = self.parse_expression()

                print("DECLARED EXPRESSION", exp)

                return DeclNode(name, var_type, exp)
            # varible assignment
            elif self.peek()[0] == "EQ":
                self.consume('EQ')
                print("ASSIGNING ACC", name, "to", self.peek())
                exp = self.parse_expression()

                if name not in self.variables:
                    raise SyntaxError(f"Variable '{name}' used before declaration")

                return AssignNode(name, self.variables[name], exp)
        else:
            raise SyntaxError(f"Unknown statement {token}")

        print(token)

    def parse_block(self):
        self.consume('B_LEFT')

        statements = []
        while self.peek() and self.peek()[0] != 'B_RIGHT':
            if self.peek()[0] == 'NEWLINE':
                self.pos += 1
                continue
            if self.peek()[0] == 'COMMENT':
                while self.peek() and self.peek(0)[0] != 'NEWLINE':
                    self.pos += 1
                continue

            statements.append(self.parse_statement())

        self.consume('B_RIGHT')
        print('BLOCK DONE')

        return BlockNode(statements)

    def parse_program(self):
        code = []
        while self.pos < len(tokens):
            if self.peek()[0] == 'NEWLINE':
                self.pos += 1
                continue
            if self.peek()[0] == 'COMMENT':
                while self.peek() and self.peek(0)[0] != 'NEWLINE':
                    self.pos += 1
                continue

            code.append(self.parse_statement())
        return code


def to_assembly(code):
    asm = ["NOP"]

    for node in program:
        if isinstance(node, DeclNode) or isinstance(node, AssignNode):
            print("decl node")
            if node.dtype == 'int8':
                low_var_adr, high_var_adr = node.name + "@LB", node.name + "@HB"

                is_decl_node = isinstance(node, DeclNode)
                if is_decl_node and node.name != '@':
                    asm.append(f"{low_var_adr} = *\n{high_var_adr} = *\n")

                if isinstance(node.value, BinOpNode):
                    # non zero left side nodes
                    bad_nodes = deconstruct_binary_node(node.value)
                    # print("bad nodes", bad_nodes)
                    for bad_node in bad_nodes:
                        # print(f"bad node @{bad_node[0]}", bad_node[1].view())
                        asm += trivial_tree_to_asm(bad_node[1], bad_node[0])
                    asm += trivial_tree_to_asm(node.value)
                elif is_number(node.value):
                    asm.append(f"LDV_A {node.value}")
                elif node.value == '@':
                    print("SUCCC")
                else:
                    print("NOT SUCC", node, node.value)
                    low_dcl_adr, high_dcl_adr = node.value + "@LB", node.value + "@HB"
                    asm.append(f"LDA_A_FB {low_dcl_adr} {high_dcl_adr}")
                    print("not digit")

                if node.name == '@':
                    asm.append("")
                else:
                    asm.append(f"STA_A_FB {low_var_adr} {high_var_adr}\n")

    return asm

def trivial_tree_to_asm(node, temp=None):
    # print("trivial function call", node)
    # arrived at last node
    if not isinstance(node, BinOpNode):
        if is_number(node):
            return [f"LDV_A {node}"]
        if node == '@':
            return []
        return [f"LDA_A {node}@HB {node}@LB"]

    # print("node to assembly", "@" + str(temp), ":", node.view(), "\n")

    code = []
    # only add temp if does not already exist, technically not a problem since will be ignored
    if temp:
        low_var_adr, high_var_adr = "temp_" + str(temp) + "@LB", "temp_" + str(temp) + "@HB"
        code = [f"{low_var_adr} = *\n{high_var_adr} = *"]

    code += trivial_tree_to_asm(node.left)

    node_op = node.op
    node_value = node.right
    if is_number(node_value):
        if node_op == "+":
            code += [f"ADV_A {node_value}"]
        elif node_op == "-":
            code += [f"SUV_A {node_value}"]
        else:
            raise SyntaxError("Invalid operator", node)
    else:
        if node_op == "+":
            code += [f"ADA_A {node_value}@HB {node_value}@LB"]
        elif node_op == "-":
            code += [f"SUA_A {node_value}@HB {node_value}@LB"]
        else:
            raise SyntaxError("Invalid operator", node)
    
    if temp:
        return code + [f"STA_A_FB {temp}@LB {temp}@HB\n"]
    else:
        return code

def deconstruct_binary_node(node, to_acc=False):
    if not isinstance(node, BinOpNode):
        return []

    if node.left == '@':
        to_acc = True

    if node.right == '@':
        raise SyntaxError(f"Cannot have @ on the right of op {node.view()}")

    print("deconstruct binary node", node.left, node.op, node.right)
    left_weight = node.left.get_weight() if isinstance(node.left, BinOpNode) else 0
    right_weight = node.right.get_weight() if isinstance(node.right, BinOpNode) else 0

    # needs @ protection
    # distribute if op is + or -
    if right_weight != 0 and False:
        print("try to reorder right node of plus op")
        current_left_node = node.left
        current_right_node = node.right

        if current_right_node.op == "+" or current_right_node.op == "-":
            print("HHH")
            right_left_node = current_right_node.left
            right_right_node = current_right_node.right

            print("Current right node", current_right_node.view())

            print("before reorder")
            print(node.view(), "\n")

            if node.op == "+":
                node.left = BinOpNode(current_left_node, current_right_node.op, right_right_node)
            else:
                node.left = BinOpNode(current_left_node, "+" if current_right_node.op == "-" else "-", right_right_node)

            node.right = right_left_node

            print("reordered nodes")
            print(node.view(), "\n")

            # same function call with reordered nodes
            return deconstruct_binary_node(node, to_acc)

    # try to reorder tree
    if node.op == '+' and node.left != '@':  # + or *; could be for any op if any register select for ALU
        if right_weight > left_weight:
            old_right_weight = right_weight
            old_right_node = node.right

            node.right = node.left
            node.left = old_right_node

            print("WE REORDEREd", node.view())

            # same function call with reordered nodes
            return deconstruct_binary_node(node, to_acc)

    if right_weight == 0:
        if left_weight == 0:
            return []

        return deconstruct_binary_node(node.left, to_acc)
    else:
        if to_acc:
            raise SyntaxError("Cannot have non trivial operations after @")

        right_temp = deconstruct_binary_node(node.right, to_acc)
        bad_node = [(max(left_weight, right_weight), node.right)]
        node.right = "temp_" + str(max(left_weight, right_weight))
        return right_temp + bad_node + deconstruct_binary_node(node.left, to_acc)

    print("LW RW", left_weight, right_weight)

file_name = 'program.j'
with open(file_name, 'r') as f:
    tokens = lex(f.read())
print(tokens)

parser = Parser(tokens)
program = parser.parse_program()

print("tree before reorder")
for node in program:
    if hasattr(node, 'view'):
        print(node.view())

assembly = to_assembly(program)
print("assembly", assembly)

print("tree after reorder")
for node in program:
    if hasattr(node, 'view'):
        print(node.view())

with open("compiled.a", "w") as f:
    f.write("\n".join(assembly))