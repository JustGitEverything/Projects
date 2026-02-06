RECORD_TYPE = 0x00
BYTE_COUNT = 32
EOF = ":00000001FF"

def calculate_checksum(byte_count, address, data):
    total = byte_count + (address >> 8) + (address & 0xFF) + RECORD_TYPE + sum(data)
    return (-total) & 0xFF

def create_hex_file(data):
    lines = ""

    for i in range(int(((len(data) + BYTE_COUNT - 1) // BYTE_COUNT))):
        row_data = data[BYTE_COUNT * i:BYTE_COUNT * i + BYTE_COUNT]
        row_byte_count = len(row_data)
        row_address = i * BYTE_COUNT
        checksum = calculate_checksum(row_byte_count, row_address, row_data)
        lines += f":{row_byte_count:02x}{row_address:04x}{RECORD_TYPE:02x}" + "".join(f"{d:02x}" for d in row_data) + f"{checksum:02x}" + "\n"

    with open("EEPROM_PROGRAM.hex", "w") as f:
        f.write(lines + EOF)

def to_num(num, parts):
    if type(num) == int:
        return num

    if num in variables:
        num = str(to_num(variables[num], parts))

    num = int(num, 0)
    
    if num < 256:
        return num

    raise Exception("Invalid number value", num, parts)

# instruction name, index, expected arguments
instructions = {
    "NOP": [0, 0],
    "LDV_A": [1, 1],
    "LDA_A_LB": [2, 1],
    "LDA_A_FB": [3, 2],
    "STA_A_FB": [4, 2],
    "JMP_LB": [5, 1],
    "JMP_FB": [6, 2],
    "ADV_A": [7, 1],
    "ADA_A_FB": [8, 2],
    "SUV_A": [9, 1],
    "SUA_A_FB": [10, 2],
    "JUN_LB": [11, 1],
    "JUN_FB": [12, 2],
}

file_name = 'program.a'
program = []

variables = {}

with open(file_name, 'r') as f:
    for line, c in enumerate(f):
        parts = c.strip().split()
        print("PARTS", parts)

        if len(parts) == 0 or parts[0][0] == "#":
            continue
        elif parts[0] in instructions:
            instruction, num_args = instructions[parts[0]]

            if len(parts) - 1 != num_args:
                raise Exception("Invalid arguments in line", line, parts)

            print("valid instruction", instruction, num_args)

            program.append(instruction)

            program += [part for part in parts[1:]]
        elif parts[0] == "CONST":
            if len(parts) != 4:
                raise Exception("Invalid assignment in line", line, parts)

            variables[parts[1]] = len(program)
            program.append(parts[3])
        elif "=" in parts:
            if len(parts) != 3:
                raise Exception("Invalid assignment in line", line, parts)
                
            if parts[2] != "*":
                variables[parts[0]] = parts[2]
            else:
                variables[parts[0]] = "*"
            
        elif parts[0][-1] == ":":
            if len(parts) != 1:
                raise Exception("Invalid location in line", line, parts)
        
            variables[parts[0][:-1]] = len(program)

        else:
            raise Exception("Invalid instruction in line", line, parts)

print("variables pre", variables)

# replace * variables with correct addresses
program_length = len(program)
for var in variables:
    if variables[var] == "*":
        variables[var] = program_length
        program_length += 1

print("variables", variables)
print("program", program)

program = [to_num(byte, parts) for byte in program]
print("finished byte code", program)

create_hex_file(program)
