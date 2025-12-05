; Mice - Self-replicating warrior
; Copies itself across memory
        ptr     DAT #0
        start   MOV #12, count
        loop    MOV @ptr, <dest
                DJN loop, count
                SPL @dest
                ADD #653, ptr
                JMZ start, ptr
        count   DAT #0
        dest    DAT #833
