;redcode
;name Pure Scanner
;author jw experimental
;strategy Ultra-aggressive 4-process scanner with saturation bombing

start   SPL     scan2
        SPL     scan3
        SPL     scan4

scan1   ADD.AB  #23,    ptr1
ptr1    SNE.I   #0,     @ptr1
        JMP.A   scan1
        MOV.I   bomb,   @ptr1
        MOV.I   bomb,   >ptr1
        MOV.I   bomb,   <ptr1
        JMP.A   scan1

scan2   ADD.AB  #23,    ptr2
ptr2    SNE.I   #0,     400
        JMP.A   scan2
        MOV.I   bomb,   @ptr2
        MOV.I   bomb,   >ptr2
        MOV.I   bomb,   <ptr2
        JMP.A   scan2

scan3   ADD.AB  #29,    ptr3
ptr3    SNE.I   #0,     800
        JMP.A   scan3
        MOV.I   bomb,   @ptr3
        MOV.I   bomb,   >ptr3
        MOV.I   bomb,   <ptr3
        JMP.A   scan3

scan4   SUB.AB  #17,    ptr4
ptr4    SNE.I   #0,     1200
        JMP.A   scan4
        MOV.I   bomb,   @ptr4
        MOV.I   bomb,   >ptr4
        MOV.I   bomb,   <ptr4
        JMP.A   scan4

bomb    DAT.F   #0,     #0

        END start
