; Imp Gate - Classic defensive warrior
; Creates a wall of DAT instructions, spawns imps through gates
        gate    DAT #0
                ADD #4, gate
                MOV #0, @gate
                JMP -2
