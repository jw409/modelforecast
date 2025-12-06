;redcode
;name Ironclad
;author LlamaWarrior
;strategy Bomber + Gate: Drop DAT bombs every 5 cells, then build decrementing
;         protection gate. Survives Imps, kills Dwarfs and scanners.

; ----- Core Clear (Bomber) -----
; Drop DAT bombs spaced by 5 to avoid Dwarf's pattern.
; When we hit something (non-zero), jump to start the gate.
add_ptr ADD.AB  #5,     scan_ptr   ; Move scan pointer forward by 5
scan_ptr JMZ.F   add_ptr, bomb_val  ; If location is zero, keep bombing
                                   ; If non-zero (likely enemy), start gate
        JMP.A   gate_start           ; Jump to gate building phase

; DAT bomb value used by scanner to overwrite enemy code.
bomb_val DAT.F   #0,     #0         ; DAT bomb dropped at scan_ptr

; ----- Protection Gate -----
; Build a multi-layer gate to slow down enemy processes.
; Gate consists of:
; 1. A decrementing DJN loop that catches fast replicators (Imps).
; 2. A SPL 0 barrier that creates many processes, slowing everything.
; 3. A core-clear to overwrite enemy code behind the gate.
gate_start SPL.B   1                ; Split to create multiple processes
        SPL.B   1                ; Further multiply processes
        MOV.I   gate_instr, gate_loc  ; Place first gate instruction
        ADD.AB  #1,     gate_loc      ; Move gate location forward
        DJN.B   gate_start, #100      ; Decrement counter, loop until 0
        JMP.A   clear_start           ; After gate, start core clear

; Gate instruction: DJN loop targeting itself.
; Any process hitting this will decrement the B-field and jump back,
; creating an infinite loop that slows them to a crawl.
gate_instr DJN.F   gate_loc, #0
gate_loc DAT.F   #0,     #0         ; Location where gate instructions are placed

; ----- Core Clear -----
; Overwrite enemy code with DAT bombs to kill them.
clear_start MOV.I   clear_bomb, clear_ptr  ; Drop DAT at clear_ptr
clear_ptr DAT.F   #0,     #0             ; Location being cleared
        SUB.AB  #1,     clear_ptr        ; Move pointer backward
        JMP.A   clear_start              ; Repeat until all memory cleared
clear_bomb DAT.F   #0,     #0             ; DAT bomb for core clear

        END scan_ptr