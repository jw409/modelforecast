;redcode
;name Silk
;author Claude
;strategy Multi-pronged attack: fast scanner + bomber + self-replicator
;strategy Scans aggressively for enemies, bombs them, while creating defensive copies
;strategy The scanner looks for non-zero cells and bombs them immediately
;strategy Meanwhile, SPL creates parallel processes for redundancy
;strategy Combines scanning (beats static warriors) with replication (survives bombing)

;=======================================================================
; INITIALIZATION - Create multiple attack processes
;=======================================================================

start   SPL     bomber          ; Split off bomber process
        SPL     scanner         ; Split off scanner process
        SPL     replicate       ; Split off replication process

;=======================================================================
; SCANNER - Aggressively hunt for enemy code
;=======================================================================

scanner ADD.AB  #15,    sptr    ; Increment scan pointer by 15
sptr    JMZ.F   scanner, 300    ; If location is zero, keep scanning
        
        ; Found something! Bomb it and surrounding area
        MOV.AB  sbomb,  @sptr   ; Drop bomb at target
        MOV.AB  sbomb,  <sptr   ; Bomb one behind
        MOV.AB  sbomb,  @sptr   ; Bomb again
        DJN.F   scanner, sptr   ; Continue scanning

sbomb   DAT.F   #0,     #0      ; Scanner bomb

;=======================================================================
; BOMBER - Carpet bomb at regular intervals
;=======================================================================

bomber  MOV.I   bbomb,  @bptr   ; Drop bomb at pointer location
        ADD.AB  #7,     bptr    ; Move pointer by 7 (coprime with 15)
        JMP.A   bomber          ; Loop forever

bptr    DAT.F   #3500,  #0      ; Bomb pointer (start far away)
bbomb   DAT.F   #0,     #0      ; Bomber bomb

;=======================================================================
; REPLICATOR - Create defensive copies to survive attacks
;=======================================================================

replicate MOV.I  #0,     rptr   ; Copy core at offset
          ADD.AB #550,   rptr   ; Jump 550 cells (spread out)
          MOV.I  start,  @rptr  ; Copy start instruction
          MOV.I  start+1, >rptr ; Copy next instruction
          MOV.I  start+2, >rptr ; Copy next instruction
          SPL.B  @rptr          ; Execute the copy
          JMP.A  replicate      ; Keep replicating

rptr      DAT.F  #0,     #0     ; Replication pointer

;=======================================================================

        END start