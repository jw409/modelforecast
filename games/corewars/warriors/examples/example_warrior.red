;name Example Warrior
;author AI Warrior System
; This is a simple example warrior demonstrating Redcode syntax
; Strategy: Basic bomber with scanning

; Constants and data
bomb     DAT.F #0, #0          ; The bomb payload
step     DAT.F #4, #0          ; Bombing increment
target   DAT.F #10, #0         ; Initial target offset

; Main bombing loop
start    ADD.AB step, target   ; Increment target by step
         MOV.I bomb, @target   ; Drop bomb at target location
         JMP.B start           ; Loop back

; Alternative syntax examples (commented out):

; Label with colon
; loop:   MOV 0, 1
;         JMP loop

; Different addressing modes
; imm     MOV.I #100, $5        ; Immediate to direct
; indir   MOV.I @2, @3          ; Indirect to indirect
; predec  MOV.AB <-1, $0        ; Pre-decrement
; postinc MOV.B $0, >1          ; Post-increment

; Process splitting (multi-threading)
; spawn   SPL.B $0              ; Create new process at current location
;         JMP.B attacker        ; Original process attacks
;         JMP.B defender        ; New process defends
