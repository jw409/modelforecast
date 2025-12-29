;redcode
;name neon_vampire.red v4
;author Metacognitive Loop (Generation 3)
;strategy Adaptive replicator with dynamic scanning and multi-phase attack

        org start

step    equ 4
gap     equ 8
scan    equ 50    ; increased scanning range for better coverage
phase   equ 1000  ; phase shift interval

start   mov bomb, @targ      ; primary replication
        add #step, targ      ; standard step
        mov core, @core      ; core protection
        jmz scan_start, @core ; switch to scanning when safe

scan_start
        mov scan_bomb, <scan_ptr ; spiral bombing
        add #step+1, scan_ptr    ; dynamic stepping (+1 breaks patterns)
        mod #phase, scan_ptr     ; phase shifting
        slt #core+8000, scan_ptr ; boundary check
        jmp clear_core           ; switch to core clearing

clear_core
        mov bomb, <scan_ptr      ; aggressive clearing
        djn clear_core, #300     ; extended attack duration
        jmp diversify            ; new contingency strategy

diversify                        ; anti-stalemate measures
        mov split_bomb, @split_ptr
        add #step*5, split_ptr   ; irregular pattern
        djn diversify, #100
        jmp scan_start

targ    dat #0, #gap             ; replication target
bomb    dat #0, #step            ; primary bomb
scan_bomb dat #0, #step*2        ; scanning bomb
split_bomb dat #0, #step*3       ; pattern-breaking bomb
split_ptr dat #0, #core+2000     ; secondary attack vector
core    dat #0, #0               ; protected core

attack  mov bomb, @targ          ; secondary attack
        add #step*3, targ        ; aggressive step
        djn attack, #200         ; longer attack cycle
        jmp scan_start           ; transition to scanning

end