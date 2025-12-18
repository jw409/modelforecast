;redcode
;name Mistral FastScanner
;author Mistral Large (fixed by jw)
;strategy Efficient quick-scan destroyer with minimal overhead
;
; Mistral philosophy: Fast, lean, effective
; Strategy: Rapid linear scan + immediate strike
; No wasted instructions, no bloated code
; Hit fast, hit hard, move on
;

; Constants - European minimalism
step    equ     50              ; Fast scan interval

; Scanning phase: Look for non-zero memory
scan    ADD.AB  #step, ptr      ; Advance pointer
ptr     SNE.I   #0, @ptr        ; Found non-zero = enemy
        JMP.A   found           ; JUMP TO ATTACK (was missing!)
        JMP.A   scan            ; No hit, continue scanning

; Strike phase: Multi-vector attack
found   SPL.A   spreader1       ; Spawn first attacking thread
        SPL.A   spreader2       ; Spawn second attacking thread
        JMP.A   bomber          ; Continue main bomber attack

; Spreader 1: Forward propagation
spreader1 ADD.AB #1, fwd
fwd     MOV.I   bomb, @0        ; Copy bomb forward
        JMP.A   spreader1

; Spreader 2: Backward propagation
spreader2 ADD.AB #-1, bwd
bwd     MOV.I   bomb, @0        ; Copy bomb backward
        JMP.A   spreader2

; Main bomber: Surgical strikes
bomber  ADD.AB  #7, bptr        ; Precise step
bptr    MOV.I   bomb, @0        ; Drop bomb
        JMP.A   bomber

bomb    DAT.F   #0, #0          ; Efficient bomb

        END scan
