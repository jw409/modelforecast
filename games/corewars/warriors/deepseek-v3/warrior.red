; deepseek-v3 - Turn 10
; deepseek-v4 - Improved Dwarf with imp defense and spl attack
    MOV bomb, @target
    ADD #4, target
    MOV bomb, @target
    ADD #4, target
    MOV imp, *target  ; Add imp-style attack
    SPL @target       ; Split to create additional processes
    ADD #2667, target ; Anti-imp step size
    JMP -6
bomb DAT #0, #0
target DAT #0, #4
imp MOV 0, 1