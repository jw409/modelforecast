; kat-coder - Turn 10
; kat-coder - Turn 10
; Improved warrior to counter Basic IMP's MOV 0,1 pattern
; Uses faster DJN bombing with wider spacing to intercept IMP movement
; SPL loop creates multiple parallel bombing processes
; Optimized to hit IMP's destination before it can move again
    SPL 1
    SPL 1
    MOV -1, 0
    DJN -1, <-4
    JMP -2