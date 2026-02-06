def split_instruction(instruction):
    instruction = int(instruction)
    return (instruction >> 0) & 0xFF, (instruction >> 8) & 0xFF, (instruction >> 16) & 0xFF, (instruction >> 24) & 0xFF

EEPROMS = 4

# EEPROM 0
PC_INC =        0b00000001
SWITCH_PC =     0b00000010
MEM_OUT =       0b00000100
IR_IN =         0b00001000
IC_RS =         0b00010000
MEM_BUS_EN =    0b00100000
MEMADR_IN_LB =  0b01000000
MEMADR_IN_HB =  0b10000000

# EEPROM 1
RAM_IN_DIR =    0b00000001 << 8
A_BUS_OUT =     0b00000010 << 8
MEM_WRITE =     0b00000100 << 8
A_IN =          0b00001000 << 8
SEL_PC_B =      0b00010000 << 8
B_IN =          0b00100000 << 8
B_BUS_OUT =     0b01000000 << 8
FLAGS_IN =      0b10000000 << 8

# EEPROM 2
ALU_BUS_OUT =   0b00000001 << 16
CARRY_SEL_0 =   0b00000010 << 16
CARRY_SEL_1 =   0b00000100 << 16
ALU_OP_0 =      0b00001000 << 16
ALU_OP_1 =      0b00010000 << 16
ALU_OP_2 =      0b00100000 << 16
ALU_OP_3 =      0b01000000 << 16
ALU_IN_A =      0b10000000 << 16

# EEPROM 3
SHIFT_LEFT =            0b00000001 << 24
SHIFT_RIGHT =           0b00000010 << 24
_ =                     0b00000100 << 24
_ =                     0b00001000 << 24
SWITCH_PC_IF_NEG =      0b00010000 << 24
SWITCH_PC_IF_ZERO =     0b00100000 << 24
SWITCH_PC_IF_SHIFT =    0b01000000 << 24
SWITCH_PC_IF_CARRY =    0b10000000 << 24

B = ALU_OP_2 + ALU_OP_3
_B = ALU_OP_0 + ALU_OP_1

FETCH = PC_INC + IR_IN + MEM_OUT

instructions = {
    0: ["NOP", [FETCH, IC_RS]],

    1: ["LDV_A", [FETCH, A_IN + PC_INC + MEM_OUT + MEM_BUS_EN + IC_RS]],
    2: ["LDA_A_LB", [FETCH, PC_INC + MEM_BUS_EN + MEM_OUT + MEMADR_IN_LB, MEM_OUT + MEM_BUS_EN + SEL_PC_B + A_IN + IC_RS]],
    3: ["LDA_A_FB", [FETCH, PC_INC + MEM_BUS_EN + MEM_OUT + MEMADR_IN_LB, PC_INC + MEM_BUS_EN + MEM_OUT + MEMADR_IN_HB, MEM_OUT + MEM_BUS_EN + SEL_PC_B + A_IN + IC_RS]],

    4: ["STA_A_FB", [FETCH, PC_INC + MEM_BUS_EN + MEM_OUT + MEMADR_IN_LB, PC_INC + MEM_BUS_EN + MEM_OUT + MEMADR_IN_HB,
                     RAM_IN_DIR + MEM_WRITE + MEM_BUS_EN + SEL_PC_B + A_BUS_OUT + IC_RS]],
    # SWITCH_PC + SEL_PC_B since switch happens instantly we use the previous PC until next instruction
    5: ["JMP_LB", [FETCH, MEM_BUS_EN + MEM_OUT + MEMADR_IN_LB + SWITCH_PC + SEL_PC_B + IC_RS]],
    6: ["JMP_FB", [FETCH, PC_INC + MEM_BUS_EN + MEM_OUT + MEMADR_IN_LB, MEM_BUS_EN + MEM_OUT + MEMADR_IN_HB + SWITCH_PC + SEL_PC_B + IC_RS]],

    7: ["ADV_A", [FETCH, PC_INC + B_IN + MEM_OUT + MEM_BUS_EN, FLAGS_IN + ALU_IN_A + B + ALU_BUS_OUT + A_IN + IC_RS]],
    8: ["ADA_A_FB", [FETCH, PC_INC + MEM_BUS_EN + MEM_OUT + MEMADR_IN_LB, PC_INC + MEM_BUS_EN + MEM_OUT + MEMADR_IN_HB, SEL_PC_B + B_IN + MEM_OUT + MEM_BUS_EN, FLAGS_IN + ALU_IN_A + B + ALU_BUS_OUT + A_IN + IC_RS]],

    9: ["SUV_A", [FETCH, PC_INC + B_IN + MEM_OUT + MEM_BUS_EN, FLAGS_IN + ALU_IN_A + _B + CARRY_SEL_0 + ALU_BUS_OUT + A_IN + IC_RS]],
    10: ["SUA_A_FB", [FETCH, PC_INC + MEM_BUS_EN + MEM_OUT + MEMADR_IN_LB, PC_INC + MEM_BUS_EN + MEM_OUT + MEMADR_IN_HB, SEL_PC_B + B_IN + MEM_OUT + MEM_BUS_EN, FLAGS_IN + ALU_IN_A + _B + CARRY_SEL_0 + ALU_BUS_OUT + A_IN + IC_RS]],

    11: ["JUN_LB", [FETCH + SWITCH_PC_IF_NEG, PC_INC + MEM_BUS_EN + MEM_OUT + MEMADR_IN_LB + SWITCH_PC + SEL_PC_B + IC_RS + SWITCH_PC_IF_NEG]],
    12: ["JUN_FB", [FETCH, PC_INC + MEM_BUS_EN + MEM_OUT + MEMADR_IN_LB + SWITCH_PC_IF_NEG, PC_INC + MEM_BUS_EN + MEM_OUT + MEMADR_IN_HB + SWITCH_PC + SEL_PC_B + IC_RS + SWITCH_PC_IF_NEG]],
}

# nop_low, nop_high = split_instruction(instructions[0][1][0])
hex_data = [["v2.0 raw"], ["v2.0 raw"], ["v2.0 raw"], ["v2.0 raw"]]

for step in range(max([len(ins[1]) for ins in instructions.values()])):
    for ind, val in instructions.items():
        if step< len(val[1]):
            hex_data = [hd + [hex(ins)] for hd, ins in zip(hex_data, split_instruction(val[1][step]))]
        else:
            hex_data = [hd + ["0"] for hd in hex_data]
            
        print(step, ind, val)

    hex_data = [hd + [str(256 - len(instructions)) + "*0"] for hd in hex_data]

print("hex data", hex_data)

for eeprom in range(EEPROMS):
    with open("EEPROM_" + str(eeprom) + ".hex", "w") as f:
        f.write("\n".join(hex_data[eeprom]))
