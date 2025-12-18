;redcode
;name SPL Vampire
;author jw experimental
;strategy Pure vampire with exponential SPL replication

start   SPL     vamp2
        SPL     vamp3
        JMP     vamp1

vamp1   MOV.I   spl_code, @v1ptr
        SPL.B   @v1ptr
        ADD.AB  #350,   v1ptr
        JMP.A   vamp1
v1ptr   DAT.F   #700,   #0

vamp2   MOV.I   spl_code, @v2ptr
        SPL.B   @v2ptr
        ADD.AB  #1100,  v2ptr
        JMP.A   vamp2
v2ptr   DAT.F   #2200,  #0

vamp3   MOV.I   spl_code, @v3ptr
        SPL.B   @v3ptr
        ADD.AB  #1900,  v3ptr
        JMP.A   vamp3
v3ptr   DAT.F   #3800,  #0

spl_code SPL.A  0,      #0

        END start
