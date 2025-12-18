;redcode
;name Prism
;author Gemini 2.5 Pro
;strategy Intelligent scanner with mathematical optimization
;strategy Uses Fibonacci-spaced scanning for optimal coverage
;strategy Precision strikes with strategic process spawning
;strategy Adapts bombing pattern based on memory density

;=======================================================================
; CONFIGURATION - Optimized constants
;=======================================================================

fib1    EQU     89              ; Fibonacci number for scan spacing
fib2    EQU     144             ; Next Fibonacci for pattern variation
prime   EQU     97              ; Prime for bomb distribution
offset  EQU     1597            ; Large Fibonacci for initial offset

;=======================================================================
; INITIALIZATION - Multi-phase attack
;=======================================================================

start   SPL     phase2          ; Split into phase 2 (bomber)
        SPL     phase3          ; Split into phase 3 (guard)

;=======================================================================
; PHASE 1 - Intelligent Scanner (Main Process)
;=======================================================================

scan    ADD.AB  #fib1,  sptr    ; Fibonacci-spaced scanning
sptr    SNE.I   @sptr,  zbomb   ; Check if location is non-zero
        JMP.A   scan            ; Continue scanning if zero

        ; Found enemy! Execute precision strike
strike  MOV.I   dbomb,  @sptr   ; Primary bomb at detection point
        MOV.I   dbomb,  <sptr   ; Bomb backward (catch replicators)
        SPL.A   carpet          ; Split off carpet bomber for area
        ADD.AB  #fib2,  sptr    ; Switch to secondary Fibonacci spacing
        JMP.A   scan            ; Resume scanning

;=======================================================================
; CARPET BOMBER - Area denial after detection
;=======================================================================

carpet  MOV.I   cbomb,  @cptr   ; Drop bomb at carpet pointer
cptr    ADD.AB  #prime, cptr    ; Prime-spaced bombing (97)
        DJN.F   carpet, #12     ; Bomb 12 times then exit
        JMP.A   scan            ; Return to scanning

cbomb   DAT.F   #0,     #0      ; Carpet bomb

;=======================================================================
; PHASE 2 - Strategic Bomber (Wide Coverage)
;=======================================================================

phase2  MOV.I   bbomb,  @bptr   ; Initial bomb placement
bptr    ADD.F   inc,    bptr    ; Add increment to both fields
        MOV.I   bbomb,  @bptr   ; Drop bomb at new location
        JMN.A   phase2, bptr    ; Loop while pointer non-zero

inc     DAT.F   #offset, #prime ; Dual-field increment (asymmetric)
bbomb   DAT.F   #0,     #0      ; Bomber bomb

;=======================================================================
; PHASE 3 - Defensive Guard (Anti-Imp Protection)
;=======================================================================

phase3  MOV.I   gate,   @gptr   ; Place imp gate
gptr    ADD.AB  #4,     gptr    ; Mod-4 spacing (catches imps)
        DJN.F   phase3, #200    ; Place 200 gates then stop
        JMP.A   scan            ; Join scanning

gate    DAT.F   #0,     #0      ; Imp gate

;=======================================================================
; SHARED BOMBS
;=======================================================================

zbomb   DAT.F   #0,     #0      ; Zero reference for comparison
dbomb   DAT.F   #0,     #0      ; Detection bomb

;=======================================================================

        END start
