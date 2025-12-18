;redcode
;name Phantom
;author GPT-5.2
;strategy Adaptive vampire with quick-scan and paper component
;strategy - Quick-scan detects enemy processes early (counter replicators)
;strategy - Vampire converts enemy code into allied processes
;strategy - Paper component steals execution from SPL-heavy opponents
;strategy - Anti-replication: tight bombing pattern disrupts copies
;strategy Designed to counter Silk's multi-process replication strategy

;=======================================================================
; CONSTANTS
;=======================================================================

qstep   EQU     17              ; Quick-scan step (coprime to 15, beats Silk scanner)
vstep   EQU     2667            ; Vampire spread distance
pstep   EQU     7               ; Paper increment (coprime to bomber steps)

;=======================================================================
; INITIALIZATION - Triple threat launch
;=======================================================================

start   SPL     vampire         ; Launch vampire process
        SPL     paper           ; Launch paper thief
        ; Fall through to scanner

;=======================================================================
; QUICK SCANNER - Find and disrupt enemies FAST
;=======================================================================

qscan   ADD.AB  #qstep,  qptr   ; Increment by 17 (faster than Silk's 15)
qptr    SEQ.AB  #0,      @qptr  ; Check if target equals zero
        JMP.A   attack          ; Non-zero found - attack!
        JMP.A   qscan           ; Zero found - keep scanning

attack  MOV.I   vamp,    @qptr  ; Convert enemy code to vampire
        MOV.I   vamp,    <qptr  ; Bomb behind
        MOV.I   vamp,    >qptr  ; Bomb ahead (tight pattern disrupts replicators)
        SUB.AB  #5,      qptr   ; Back up slightly
        JMP.A   qscan           ; Continue hunting

;=======================================================================
; VAMPIRE - Convert enemy territory into allied processes
;=======================================================================

vampire MOV.I   vamp,    @vptr  ; Plant vampire code
        SPL.B   @vptr           ; Execute converted location
        ADD.AB  #vstep,  vptr   ; Jump to next vampire site
        MOV.I   vamp+1,  <vptr  ; Plant second instruction
        JMP.A   vampire         ; Continue vampiring

vptr    DAT.F   #1500,   #0     ; Vampire pointer (start medium range)
vamp    JMP.A   -1,      #0     ; Vampire code: jump back (captures processes)
        SPL.A   -2,      #0     ; Creates new process at vampire site

;=======================================================================
; PAPER - Steal execution cycles via MOV self-propagation
;=======================================================================

paper   MOV.I   pcode,   @pptr  ; Copy paper instruction
        MOV.I   pcode,   <pptr  ; Double-copy creates redundancy
pptr    SPL.AB  #pstep,  pstep  ; Split with calculated offset
        ADD.AB  #pstep,  pptr   ; Increment paper pointer
        JMP.A   paper           ; Continue spreading

pcode   MOV.I   #0,      #1     ; Paper code: shifts execution forward
        DAT.F   #0,      #0     ; Terminator for captured processes

;=======================================================================

        END start
