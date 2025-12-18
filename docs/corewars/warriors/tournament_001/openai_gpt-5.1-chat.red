;redcode
;name StoneGate
;author ChatGPT
;strategy Mod-4 bomber with self-protecting imp-gate

step    EQU     4

gate    DAT.F   #0, #0       ; imp-gate: enemy imps hitting here die
bomb    DAT.F   #0, #0

start   ADD.AB  #step, ptr   ; increment pointer by 4 (good vs imps)
ptr     MOV.I   bomb, @ptr   ; drop DAT bomb
        JMZ.F   start, *ptr  ; if location was zero, continue scanning
loop    JMP.A   start         ; continue bomb loop

        END start