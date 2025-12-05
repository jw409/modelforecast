; gpt4o-mini - Turn 10
; gpt4o-mini - Turn 10
    SPL 0          ; Split to create new processes
    MOV 0, 1       ; Move current process to next address
    SPL 1          ; Create another split for more coverage
    MOV 1, 2       ; Move created process
    SPL 2          ; Split again for additional coverage
    MOV 2, 3       ; Move next
    JMP <3         ; Redirect to original processes
    DJN <0, -1     ; Loop until zero
    DAT 0          ; Protects against direct hits
    DAT 0          ; Clutter
    DAT 0          ; More clutter