;redcode
;name Hybrid Hunter
;author jw experimental
;strategy Scanner + Bomber coordination

start   SPL     scanner
        JMP     bomber

scanner ADD.AB  #17,    s_ptr
s_ptr   SNE.I   #0,     500
        JMP.A   scanner
        MOV.I   bomb,   @s_ptr
        MOV.I   bomb,   >s_ptr
        MOV.I   bomb,   <s_ptr
        JMP.A   scanner

bomber  MOV.I   bomb,   @b_ptr
        ADD.AB  #11,    b_ptr
        MOV.I   bomb,   @b_ptr
        MOV.I   bomb,   <b_ptr
        JMP.A   bomber

b_ptr   DAT.F   #3500,  #0
bomb    DAT.F   #0,     #0

        END start
