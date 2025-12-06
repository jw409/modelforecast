;redcode
;name Granite
;author Claude
;strategy Rolling stone bomber with wide bombing pattern

stone   MOV.I   <-100,  >200    ; Decrement behind, increment ahead (IMP gate effect)
        ADD.AB  #653,   stone   ; Change bombing distance (step of 653 is coprime to 8000)
        JMP.A   stone           ; Loop
        END     stone