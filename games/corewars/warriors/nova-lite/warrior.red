; nova-lite - Turn 7
; nova-lite - Turn 7
; Aggressive bomber spreading DATs
    DAT #0       ; initial bomb position
bomb:
    DAT #0       ; plant DAT here
    ADD #4, bomb ; move to next position
    JMP -1       ; loop back to plant next DAT