; Dwarf - Classic bomber
; Drops DAT bombs across memory
        bomb    DAT #0
                ADD #4, bomb
                MOV bomb, @bomb
                JMP -2
