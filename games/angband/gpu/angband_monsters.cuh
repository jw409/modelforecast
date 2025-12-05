#ifndef ANGBAND_MONSTERS_CUH
#define ANGBAND_MONSTERS_CUH

#include <cuda_runtime.h>
#include <stdint.h>

// Maximum monsters and blows
#define MAX_MONSTER_RACES 623
#define MAX_BLOWS_PER_MONSTER 4

// Attack effects
#define BLOW_EFFECT_ACID 1
#define BLOW_EFFECT_BLACK_BREATH 2
#define BLOW_EFFECT_BLIND 3
#define BLOW_EFFECT_COLD 4
#define BLOW_EFFECT_CONFUSE 5
#define BLOW_EFFECT_DISENCHANT 6
#define BLOW_EFFECT_DRAIN_CHARGES 7
#define BLOW_EFFECT_EAT_FOOD 8
#define BLOW_EFFECT_EAT_GOLD 9
#define BLOW_EFFECT_EAT_ITEM 10
#define BLOW_EFFECT_EAT_LIGHT 11
#define BLOW_EFFECT_ELEC 12
#define BLOW_EFFECT_EXP_10 13
#define BLOW_EFFECT_EXP_20 14
#define BLOW_EFFECT_EXP_40 15
#define BLOW_EFFECT_EXP_80 16
#define BLOW_EFFECT_FIRE 17
#define BLOW_EFFECT_HALLU 18
#define BLOW_EFFECT_HURT 19
#define BLOW_EFFECT_LOSE_ALL 20
#define BLOW_EFFECT_LOSE_CON 21
#define BLOW_EFFECT_LOSE_DEX 22
#define BLOW_EFFECT_LOSE_INT 23
#define BLOW_EFFECT_LOSE_STR 24
#define BLOW_EFFECT_LOSE_WIS 25
#define BLOW_EFFECT_PARALYZE 26
#define BLOW_EFFECT_POISON 27
#define BLOW_EFFECT_SHATTER 28
#define BLOW_EFFECT_TERRIFY 29

// Monster flags
#define MFLAG_UNIQUE     0x01
#define MFLAG_EVIL       0x02
#define MFLAG_UNDEAD     0x04
#define MFLAG_DRAGON     0x08
#define MFLAG_DEMON      0x10
#define MFLAG_ANIMAL     0x20
#define MFLAG_SMART      0x40
#define MFLAG_REGENERATE 0x80

// Monster blow (attack)
struct MonsterBlow {
    uint8_t effect;    // BLOW_EFFECT_*
    uint8_t dd;        // damage dice
    uint8_t ds;        // damage sides
    uint8_t pad;       // padding for alignment
};

// Monster race definition
struct MonsterRace {
    int16_t speed;      // 110 = normal, higher = faster
    int16_t hp;         // average hit points (will overflow for Morgoth, but scaled in practice)
    int16_t ac;         // armor class
    int16_t depth;      // native depth (1-100)
    int16_t rarity;     // inverse spawn probability
    int32_t exp;        // experience value (needs 32-bit for high-level monsters)
    uint8_t num_blows;  // number of attacks
    uint8_t flags;      // packed flags (UNIQUE, EVIL, etc)
    MonsterBlow blows[MAX_BLOWS_PER_MONSTER];
};

// Monster data in global memory (too large for constant memory)
__device__ MonsterRace MONSTER_RACES[MAX_MONSTER_RACES] = {
    // 0: filthy street urchin
    {
        110, // speed
        3, // hp
        1, // ac
        0, // depth
        2, // rarity
        0, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {0, 0, 0, 0},
            {9, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 1: scrawny cat
    {
        110, // speed
        2, // hp
        1, // ac
        0, // depth
        3, // rarity
        0, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 1, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 2: scruffy little dog
    {
        110, // speed
        2, // hp
        1, // ac
        0, // depth
        3, // rarity
        0, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 1, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 3: Farmer Maggot
    {
        110, // speed
        350, // hp
        12, // ac
        0, // depth
        4, // rarity
        0, // exp
        2, // num_blows
        0x01, // flags
        { // blows
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 4: blubbering idiot
    {
        110, // speed
        2, // hp
        1, // ac
        0, // depth
        1, // rarity
        0, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 5: boil-covered wretch
    {
        110, // speed
        2, // hp
        1, // ac
        0, // depth
        1, // rarity
        0, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 6: village idiot
    {
        120, // speed
        10, // hp
        1, // ac
        0, // depth
        1, // rarity
        0, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 7: pitiful-looking beggar
    {
        110, // speed
        3, // hp
        1, // ac
        0, // depth
        1, // rarity
        0, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 8: mangy-looking leper
    {
        110, // speed
        1, // hp
        1, // ac
        0, // depth
        1, // rarity
        0, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 9: squint-eyed rogue
    {
        110, // speed
        9, // hp
        9, // ac
        0, // depth
        1, // rarity
        0, // exp
        3, // num_blows
        0x02, // flags
        { // blows
            {19, 1, 6, 0},
            {10, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 10: singing, happy drunk
    {
        110, // speed
        4, // hp
        1, // ac
        0, // depth
        1, // rarity
        0, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 11: aimless-looking merchant
    {
        110, // speed
        6, // hp
        1, // ac
        0, // depth
        1, // rarity
        0, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 3, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 12: mean-looking mercenary
    {
        110, // speed
        23, // hp
        24, // ac
        0, // depth
        1, // rarity
        0, // exp
        1, // num_blows
        0x02, // flags
        { // blows
            {19, 1, 10, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 13: battle-scarred veteran
    {
        110, // speed
        32, // hp
        36, // ac
        0, // depth
        1, // rarity
        0, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 14: red-hatted elf
    {
        110, // speed
        100, // hp
        12, // ac
        0, // depth
        1, // rarity
        0, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 1, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 15: Father Christmas
    {
        100, // speed
        1000, // hp
        12, // ac
        0, // depth
        1, // rarity
        0, // exp
        0, // num_blows
        0x01, // flags
        { // blows
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 16: grey mold
    {
        110, // speed
        2, // hp
        1, // ac
        1, // depth
        1, // rarity
        3, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 4, 0},
            {19, 1, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 17: grey mushroom patch
    {
        110, // speed
        2, // hp
        1, // ac
        1, // depth
        1, // rarity
        1, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {5, 1, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 18: giant yellow centipede
    {
        110, // speed
        7, // hp
        14, // ac
        1, // depth
        1, // rarity
        2, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 3, 0},
            {19, 1, 3, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 19: giant white centipede
    {
        110, // speed
        9, // hp
        12, // ac
        1, // depth
        1, // rarity
        2, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 2, 0},
            {4, 1, 2, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 20: white icky thing
    {
        110, // speed
        6, // hp
        8, // ac
        1, // depth
        1, // rarity
        1, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 2, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 21: clear icky thing
    {
        110, // speed
        6, // hp
        7, // ac
        1, // depth
        1, // rarity
        2, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 2, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 22: giant white mouse
    {
        110, // speed
        2, // hp
        4, // ac
        1, // depth
        1, // rarity
        1, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 2, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 23: large white snake
    {
        100, // speed
        11, // hp
        36, // ac
        1, // depth
        1, // rarity
        2, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 1, 0},
            {19, 1, 1, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 24: small kobold
    {
        110, // speed
        8, // hp
        24, // ac
        1, // depth
        1, // rarity
        5, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 5, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 25: white worm mass
    {
        100, // speed
        10, // hp
        1, // ac
        1, // depth
        1, // rarity
        2, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {27, 1, 2, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 26: floating eye
    {
        110, // speed
        11, // hp
        7, // ac
        1, // depth
        1, // rarity
        1, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {26, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 27: rock lizard
    {
        110, // speed
        8, // hp
        4, // ac
        1, // depth
        1, // rarity
        2, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 1, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 28: wild dog
    {
        110, // speed
        3, // hp
        3, // ac
        1, // depth
        1, // rarity
        1, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 1, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 29: soldier ant
    {
        110, // speed
        6, // hp
        4, // ac
        1, // depth
        1, // rarity
        3, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 2, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 30: fruit bat
    {
        120, // speed
        4, // hp
        3, // ac
        1, // depth
        1, // rarity
        1, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 1, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 31: kobold
    {
        110, // speed
        12, // hp
        24, // ac
        2, // depth
        1, // rarity
        5, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 8, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 32: shrieker mushroom patch
    {
        110, // speed
        1, // hp
        1, // ac
        2, // depth
        1, // rarity
        1, // exp
        0, // num_blows
        0x00, // flags
        { // blows
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 33: blubbering icky thing
    {
        110, // speed
        18, // hp
        4, // ac
        2, // depth
        1, // rarity
        8, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {27, 1, 4, 0},
            {8, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 34: metallic green centipede
    {
        120, // speed
        10, // hp
        4, // ac
        2, // depth
        1, // rarity
        3, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 1, 0},
            {1, 1, 1, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 35: soldier
    {
        110, // speed
        23, // hp
        24, // ac
        2, // depth
        1, // rarity
        6, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 7, 0},
            {19, 1, 7, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 36: cutpurse
    {
        110, // speed
        20, // hp
        18, // ac
        2, // depth
        1, // rarity
        6, // exp
        2, // num_blows
        0x02, // flags
        { // blows
            {19, 1, 6, 0},
            {9, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 37: acolyte
    {
        110, // speed
        18, // hp
        15, // ac
        2, // depth
        1, // rarity
        6, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 5, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 38: apprentice
    {
        110, // speed
        15, // hp
        9, // ac
        2, // depth
        1, // rarity
        6, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 39: yellow mushroom patch
    {
        110, // speed
        1, // hp
        1, // ac
        2, // depth
        1, // rarity
        2, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {29, 1, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 40: white jelly
    {
        120, // speed
        36, // hp
        1, // ac
        2, // depth
        1, // rarity
        10, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {27, 1, 2, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 41: giant green frog
    {
        110, // speed
        9, // hp
        9, // ac
        2, // depth
        1, // rarity
        6, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 3, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 42: giant black ant
    {
        110, // speed
        11, // hp
        24, // ac
        2, // depth
        1, // rarity
        8, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 43: salamander
    {
        110, // speed
        14, // hp
        24, // ac
        2, // depth
        1, // rarity
        10, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {17, 1, 3, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 44: white harpy
    {
        110, // speed
        6, // hp
        20, // ac
        2, // depth
        1, // rarity
        5, // exp
        3, // num_blows
        0x22, // flags
        { // blows
            {19, 1, 1, 0},
            {19, 1, 1, 0},
            {19, 1, 2, 0},
            {0, 0, 0, 0}
        }
    },
    // 45: blue yeek
    {
        110, // speed
        7, // hp
        16, // ac
        2, // depth
        1, // rarity
        4, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 5, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 46: Grip, Farmer Maggot's Dog
    {
        120, // speed
        25, // hp
        36, // ac
        2, // depth
        1, // rarity
        30, // exp
        1, // num_blows
        0x01, // flags
        { // blows
            {19, 1, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 47: Fang, Farmer Maggot's Dog
    {
        120, // speed
        25, // hp
        36, // ac
        2, // depth
        1, // rarity
        30, // exp
        1, // num_blows
        0x01, // flags
        { // blows
            {19, 1, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 48: green worm mass
    {
        100, // speed
        15, // hp
        3, // ac
        2, // depth
        1, // rarity
        3, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {1, 1, 3, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 49: cave spider
    {
        120, // speed
        7, // hp
        19, // ac
        2, // depth
        1, // rarity
        7, // exp
        1, // num_blows
        0x20, // flags
        { // blows
            {19, 1, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 50: wild cat
    {
        120, // speed
        9, // hp
        14, // ac
        2, // depth
        2, // rarity
        8, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 3, 0},
            {19, 1, 3, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 51: crow
    {
        120, // speed
        9, // hp
        14, // ac
        2, // depth
        2, // rarity
        8, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 3, 0},
            {19, 1, 3, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 52: Sméagol
    {
        130, // speed
        400, // hp
        14, // ac
        3, // depth
        2, // rarity
        50, // exp
        1, // num_blows
        0x03, // flags
        { // blows
            {9, 1, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 53: green ooze
    {
        120, // speed
        8, // hp
        19, // ac
        3, // depth
        2, // rarity
        4, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {1, 1, 3, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 54: poltergeist
    {
        130, // speed
        6, // hp
        18, // ac
        3, // depth
        1, // rarity
        8, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {29, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 55: metallic blue centipede
    {
        120, // speed
        12, // hp
        7, // ac
        3, // depth
        1, // rarity
        7, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 2, 0},
            {12, 1, 2, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 56: giant white louse
    {
        120, // speed
        1, // hp
        6, // ac
        3, // depth
        1, // rarity
        1, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 1, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 57: black naga
    {
        110, // speed
        27, // hp
        60, // ac
        3, // depth
        1, // rarity
        20, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 8, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 58: spotted mushroom patch
    {
        110, // speed
        1, // hp
        1, // ac
        3, // depth
        1, // rarity
        3, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {27, 2, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 59: silver jelly
    {
        120, // speed
        45, // hp
        1, // ac
        3, // depth
        2, // rarity
        12, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {11, 1, 3, 0},
            {11, 1, 3, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 60: yellow jelly
    {
        120, // speed
        45, // hp
        1, // ac
        3, // depth
        1, // rarity
        12, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {27, 1, 3, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 61: scruffy looking hobbit
    {
        110, // speed
        9, // hp
        9, // ac
        3, // depth
        1, // rarity
        4, // exp
        2, // num_blows
        0x02, // flags
        { // blows
            {19, 1, 4, 0},
            {9, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 62: giant white ant
    {
        110, // speed
        11, // hp
        19, // ac
        3, // depth
        1, // rarity
        7, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 63: yellow mold
    {
        110, // speed
        36, // hp
        12, // ac
        3, // depth
        1, // rarity
        9, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 64: metallic red centipede
    {
        120, // speed
        18, // hp
        10, // ac
        3, // depth
        1, // rarity
        12, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 2, 0},
            {17, 1, 2, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 65: yellow worm mass
    {
        100, // speed
        18, // hp
        4, // ac
        3, // depth
        2, // rarity
        4, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {22, 1, 3, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 66: clear worm mass
    {
        100, // speed
        10, // hp
        1, // ac
        3, // depth
        2, // rarity
        4, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {27, 1, 2, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 67: radiation eye
    {
        110, // speed
        11, // hp
        7, // ac
        3, // depth
        1, // rarity
        6, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {24, 1, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 68: kobold shaman
    {
        110, // speed
        11, // hp
        24, // ac
        3, // depth
        1, // rarity
        15, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 8, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 69: cave lizard
    {
        110, // speed
        11, // hp
        19, // ac
        4, // depth
        1, // rarity
        8, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 5, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 70: scout
    {
        110, // speed
        27, // hp
        12, // ac
        4, // depth
        1, // rarity
        18, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 5, 0},
            {19, 1, 5, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 71: gallant
    {
        110, // speed
        27, // hp
        24, // ac
        4, // depth
        1, // rarity
        18, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 7, 0},
            {19, 1, 7, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 72: tamer
    {
        110, // speed
        20, // hp
        20, // ac
        4, // depth
        1, // rarity
        13, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 2, 0},
            {19, 1, 2, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 73: blue jelly
    {
        110, // speed
        54, // hp
        1, // ac
        4, // depth
        1, // rarity
        14, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {4, 1, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 74: creeping copper coins
    {
        100, // speed
        32, // hp
        28, // ac
        4, // depth
        3, // rarity
        9, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 4, 0},
            {27, 2, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 75: giant white rat
    {
        110, // speed
        3, // hp
        8, // ac
        4, // depth
        1, // rarity
        1, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {27, 1, 3, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 76: blue worm mass
    {
        100, // speed
        23, // hp
        14, // ac
        4, // depth
        1, // rarity
        5, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {4, 1, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 77: large grey snake
    {
        100, // speed
        27, // hp
        61, // ac
        4, // depth
        1, // rarity
        14, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 5, 0},
            {19, 1, 8, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 78: kobold archer
    {
        110, // speed
        24, // hp
        24, // ac
        4, // depth
        1, // rarity
        20, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 9, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 79: silver mouse
    {
        110, // speed
        2, // hp
        4, // ac
        4, // depth
        1, // rarity
        1, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {11, 1, 2, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 80: Bullroarer the Hobbit
    {
        120, // speed
        60, // hp
        12, // ac
        5, // depth
        3, // rarity
        90, // exp
        2, // num_blows
        0x01, // flags
        { // blows
            {19, 1, 8, 0},
            {19, 1, 8, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 81: green naga
    {
        110, // speed
        41, // hp
        48, // ac
        5, // depth
        1, // rarity
        30, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 8, 0},
            {1, 2, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 82: blue ooze
    {
        110, // speed
        8, // hp
        19, // ac
        5, // depth
        1, // rarity
        7, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {4, 1, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 83: green glutton ghost
    {
        130, // speed
        8, // hp
        24, // ac
        5, // depth
        1, // rarity
        15, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {8, 1, 1, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 84: green jelly
    {
        120, // speed
        99, // hp
        1, // ac
        5, // depth
        1, // rarity
        18, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {1, 1, 2, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 85: large kobold
    {
        110, // speed
        65, // hp
        48, // ac
        5, // depth
        1, // rarity
        25, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 10, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 86: skeleton kobold
    {
        110, // speed
        23, // hp
        39, // ac
        5, // depth
        1, // rarity
        12, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 87: grey icky thing
    {
        110, // speed
        18, // hp
        14, // ac
        5, // depth
        1, // rarity
        10, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 5, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 88: disenchanter eye
    {
        100, // speed
        32, // hp
        7, // ac
        5, // depth
        2, // rarity
        20, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {6, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 89: red worm mass
    {
        100, // speed
        23, // hp
        14, // ac
        5, // depth
        1, // rarity
        6, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {17, 1, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 90: copperhead snake
    {
        110, // speed
        14, // hp
        30, // ac
        5, // depth
        1, // rarity
        15, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {27, 2, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 91: giant white dragon fly
    {
        110, // speed
        14, // hp
        30, // ac
        5, // depth
        2, // rarity
        15, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {4, 1, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 92: giant green dragon fly
    {
        110, // speed
        14, // hp
        30, // ac
        5, // depth
        2, // rarity
        16, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {27, 1, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 93: rot jelly
    {
        120, // speed
        90, // hp
        36, // ac
        5, // depth
        1, // rarity
        15, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {8, 2, 3, 0},
            {25, 2, 3, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 94: purple mushroom patch
    {
        110, // speed
        1, // hp
        1, // ac
        6, // depth
        2, // rarity
        15, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {21, 1, 2, 0},
            {21, 1, 2, 0},
            {21, 1, 2, 0},
            {0, 0, 0, 0}
        }
    },
    // 95: brown mold
    {
        110, // speed
        68, // hp
        14, // ac
        6, // depth
        1, // rarity
        20, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {5, 1, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 96: giant brown bat
    {
        130, // speed
        14, // hp
        18, // ac
        6, // depth
        1, // rarity
        10, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 3, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 97: creeping silver coins
    {
        100, // speed
        54, // hp
        36, // ac
        6, // depth
        3, // rarity
        18, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 6, 0},
            {27, 2, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 98: snaga
    {
        110, // speed
        36, // hp
        48, // ac
        6, // depth
        1, // rarity
        15, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 8, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 99: rattlesnake
    {
        110, // speed
        24, // hp
        36, // ac
        6, // depth
        1, // rarity
        20, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {27, 2, 5, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 100: giant black dragon fly
    {
        120, // speed
        14, // hp
        30, // ac
        6, // depth
        2, // rarity
        18, // exp
        0, // num_blows
        0x00, // flags
        { // blows
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 101: giant gold dragon fly
    {
        120, // speed
        14, // hp
        30, // ac
        6, // depth
        2, // rarity
        18, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 3, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 102: crow of Durthang
    {
        120, // speed
        12, // hp
        14, // ac
        7, // depth
        3, // rarity
        10, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 3, 0},
            {19, 1, 2, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 103: cave orc
    {
        110, // speed
        55, // hp
        48, // ac
        7, // depth
        1, // rarity
        20, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 8, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 104: wood spider
    {
        120, // speed
        11, // hp
        19, // ac
        7, // depth
        3, // rarity
        15, // exp
        2, // num_blows
        0x20, // flags
        { // blows
            {19, 1, 3, 0},
            {27, 1, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 105: bloodshot eye
    {
        110, // speed
        45, // hp
        7, // ac
        7, // depth
        3, // rarity
        15, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {3, 2, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 106: red naga
    {
        110, // speed
        50, // hp
        48, // ac
        7, // depth
        2, // rarity
        40, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 10, 0},
            {24, 1, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 107: red jelly
    {
        110, // speed
        117, // hp
        1, // ac
        7, // depth
        1, // rarity
        26, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {24, 1, 5, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 108: giant red frog
    {
        110, // speed
        23, // hp
        19, // ac
        7, // depth
        1, // rarity
        16, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {24, 2, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 109: green icky thing
    {
        110, // speed
        23, // hp
        14, // ac
        7, // depth
        2, // rarity
        18, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {1, 2, 5, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 110: zombified kobold
    {
        110, // speed
        27, // hp
        21, // ac
        7, // depth
        1, // rarity
        14, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 2, 0},
            {19, 1, 2, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 111: lost soul
    {
        110, // speed
        9, // hp
        12, // ac
        7, // depth
        2, // rarity
        18, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 2, 0},
            {25, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 112: eastern dwarf
    {
        110, // speed
        39, // hp
        24, // ac
        7, // depth
        2, // rarity
        25, // exp
        2, // num_blows
        0x02, // flags
        { // blows
            {19, 1, 6, 0},
            {19, 1, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 113: night lizard
    {
        110, // speed
        18, // hp
        19, // ac
        7, // depth
        2, // rarity
        35, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 6, 0},
            {19, 1, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 114: Mughash the Kobold Lord
    {
        110, // speed
        150, // hp
        30, // ac
        7, // depth
        3, // rarity
        100, // exp
        3, // num_blows
        0x01, // flags
        { // blows
            {19, 1, 10, 0},
            {19, 1, 10, 0},
            {19, 1, 10, 0},
            {0, 0, 0, 0}
        }
    },
    // 115: Wormtongue, Agent of Saruman
    {
        110, // speed
        250, // hp
        45, // ac
        8, // depth
        1, // rarity
        150, // exp
        4, // num_blows
        0x03, // flags
        { // blows
            {19, 1, 5, 0},
            {19, 1, 5, 0},
            {9, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 116: Lagduf, the Snaga
    {
        110, // speed
        190, // hp
        48, // ac
        8, // depth
        2, // rarity
        80, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {19, 1, 10, 0},
            {19, 1, 10, 0},
            {19, 1, 9, 0},
            {19, 1, 9, 0}
        }
    },
    // 117: terrified yeek
    {
        110, // speed
        18, // hp
        21, // ac
        8, // depth
        1, // rarity
        11, // exp
        0, // num_blows
        0x00, // flags
        { // blows
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 118: giant salamander
    {
        110, // speed
        24, // hp
        60, // ac
        8, // depth
        1, // rarity
        50, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {17, 3, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 119: green mold
    {
        110, // speed
        95, // hp
        16, // ac
        8, // depth
        1, // rarity
        28, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {29, 1, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 120: skeleton orc
    {
        110, // speed
        45, // hp
        54, // ac
        8, // depth
        1, // rarity
        26, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 5, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 121: lemure
    {
        110, // speed
        65, // hp
        48, // ac
        8, // depth
        3, // rarity
        16, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 8, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 122: orc tracker
    {
        110, // speed
        55, // hp
        40, // ac
        8, // depth
        1, // rarity
        25, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 8, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 123: ruffian
    {
        110, // speed
        36, // hp
        28, // ac
        8, // depth
        2, // rarity
        26, // exp
        2, // num_blows
        0x02, // flags
        { // blows
            {19, 2, 4, 0},
            {19, 2, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 124: giant black louse
    {
        120, // speed
        2, // hp
        8, // ac
        9, // depth
        1, // rarity
        3, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 2, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 125: yeti
    {
        110, // speed
        55, // hp
        36, // ac
        9, // depth
        3, // rarity
        30, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 3, 0},
            {19, 1, 3, 0},
            {19, 1, 4, 0},
            {0, 0, 0, 0}
        }
    },
    // 126: bloodshot icky thing
    {
        110, // speed
        32, // hp
        21, // ac
        9, // depth
        3, // rarity
        24, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 4, 0},
            {1, 2, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 127: giant grey rat
    {
        110, // speed
        4, // hp
        14, // ac
        9, // depth
        1, // rarity
        2, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {27, 1, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 128: black harpy
    {
        120, // speed
        14, // hp
        26, // ac
        9, // depth
        1, // rarity
        19, // exp
        3, // num_blows
        0x22, // flags
        { // blows
            {19, 1, 2, 0},
            {19, 1, 2, 0},
            {19, 1, 3, 0},
            {0, 0, 0, 0}
        }
    },
    // 129: orc shaman
    {
        110, // speed
        41, // hp
        22, // ac
        9, // depth
        1, // rarity
        30, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 6, 0},
            {19, 1, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 130: baby blue dragon
    {
        110, // speed
        88, // hp
        36, // ac
        9, // depth
        2, // rarity
        35, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 3, 0},
            {19, 1, 3, 0},
            {19, 1, 5, 0},
            {0, 0, 0, 0}
        }
    },
    // 131: baby white dragon
    {
        110, // speed
        88, // hp
        36, // ac
        9, // depth
        2, // rarity
        35, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 3, 0},
            {19, 1, 3, 0},
            {19, 1, 5, 0},
            {0, 0, 0, 0}
        }
    },
    // 132: baby green dragon
    {
        110, // speed
        88, // hp
        36, // ac
        9, // depth
        2, // rarity
        35, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 3, 0},
            {19, 1, 3, 0},
            {19, 1, 5, 0},
            {0, 0, 0, 0}
        }
    },
    // 133: baby black dragon
    {
        110, // speed
        88, // hp
        36, // ac
        9, // depth
        2, // rarity
        35, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 3, 0},
            {19, 1, 3, 0},
            {19, 1, 5, 0},
            {0, 0, 0, 0}
        }
    },
    // 134: baby red dragon
    {
        110, // speed
        88, // hp
        36, // ac
        9, // depth
        2, // rarity
        35, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 3, 0},
            {19, 1, 3, 0},
            {19, 1, 5, 0},
            {0, 0, 0, 0}
        }
    },
    // 135: giant red ant
    {
        110, // speed
        18, // hp
        40, // ac
        9, // depth
        2, // rarity
        22, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 4, 0},
            {24, 1, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 136: Brodda, the Easterling
    {
        110, // speed
        210, // hp
        37, // ac
        9, // depth
        2, // rarity
        100, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {19, 1, 12, 0},
            {19, 1, 12, 0},
            {19, 1, 12, 0},
            {19, 1, 12, 0}
        }
    },
    // 137: king cobra
    {
        110, // speed
        44, // hp
        45, // ac
        9, // depth
        2, // rarity
        28, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {3, 1, 2, 0},
            {27, 3, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 138: baby gold dragon
    {
        110, // speed
        88, // hp
        36, // ac
        9, // depth
        2, // rarity
        35, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 3, 0},
            {19, 1, 3, 0},
            {19, 1, 5, 0},
            {0, 0, 0, 0}
        }
    },
    // 139: cave bear
    {
        110, // speed
        36, // hp
        52, // ac
        9, // depth
        1, // rarity
        25, // exp
        3, // num_blows
        0x20, // flags
        { // blows
            {19, 1, 6, 0},
            {19, 1, 6, 0},
            {19, 1, 8, 0},
            {0, 0, 0, 0}
        }
    },
    // 140: giant spider
    {
        110, // speed
        55, // hp
        24, // ac
        10, // depth
        2, // rarity
        35, // exp
        4, // num_blows
        0x20, // flags
        { // blows
            {19, 1, 10, 0},
            {27, 1, 6, 0},
            {27, 1, 6, 0},
            {19, 1, 10, 0}
        }
    },
    // 141: blacklock mage
    {
        120, // speed
        39, // hp
        24, // ac
        10, // depth
        1, // rarity
        50, // exp
        2, // num_blows
        0x02, // flags
        { // blows
            {19, 1, 6, 0},
            {19, 1, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 142: Orfax, Son of Boldor
    {
        120, // speed
        120, // hp
        24, // ac
        10, // depth
        3, // rarity
        80, // exp
        4, // num_blows
        0x43, // flags
        { // blows
            {19, 1, 9, 0},
            {19, 1, 8, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 143: stonefoot warrior
    {
        110, // speed
        60, // hp
        24, // ac
        10, // depth
        1, // rarity
        50, // exp
        2, // num_blows
        0x02, // flags
        { // blows
            {19, 1, 8, 0},
            {19, 1, 8, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 144: clear mushroom patch
    {
        120, // speed
        1, // hp
        1, // ac
        10, // depth
        2, // rarity
        3, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 1, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 145: Grishnákh, the Hill Orc
    {
        110, // speed
        230, // hp
        30, // ac
        10, // depth
        3, // rarity
        160, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {19, 1, 12, 0},
            {19, 1, 10, 0},
            {19, 1, 12, 0},
            {19, 1, 10, 0}
        }
    },
    // 146: giant white tick
    {
        100, // speed
        54, // hp
        150, // ac
        10, // depth
        2, // rarity
        27, // exp
        1, // num_blows
        0x20, // flags
        { // blows
            {27, 2, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 147: hairy mold
    {
        110, // speed
        68, // hp
        22, // ac
        10, // depth
        1, // rarity
        32, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {27, 1, 3, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 148: disenchanter mold
    {
        110, // speed
        72, // hp
        30, // ac
        10, // depth
        1, // rarity
        40, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {6, 1, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 149: pseudo-dragon
    {
        110, // speed
        176, // hp
        36, // ac
        10, // depth
        2, // rarity
        150, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 3, 0},
            {19, 1, 3, 0},
            {19, 1, 5, 0},
            {0, 0, 0, 0}
        }
    },
    // 150: tengu
    {
        120, // speed
        80, // hp
        38, // ac
        10, // depth
        1, // rarity
        40, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 8, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 151: creeping gold coins
    {
        100, // speed
        81, // hp
        43, // ac
        10, // depth
        3, // rarity
        32, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 5, 0},
            {27, 3, 5, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 152: wolf
    {
        120, // speed
        21, // hp
        45, // ac
        10, // depth
        1, // rarity
        30, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 153: giant fruit fly
    {
        120, // speed
        3, // hp
        16, // ac
        10, // depth
        3, // rarity
        4, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 2, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 154: panther
    {
        120, // speed
        45, // hp
        36, // ac
        10, // depth
        2, // rarity
        25, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 8, 0},
            {19, 1, 8, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 155: witch
    {
        110, // speed
        41, // hp
        48, // ac
        10, // depth
        2, // rarity
        35, // exp
        1, // num_blows
        0x02, // flags
        { // blows
            {19, 1, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 156: baby multi-hued dragon
    {
        110, // speed
        114, // hp
        36, // ac
        11, // depth
        2, // rarity
        45, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 4, 0},
            {19, 1, 4, 0},
            {19, 1, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 157: hippogriff
    {
        110, // speed
        100, // hp
        21, // ac
        11, // depth
        1, // rarity
        30, // exp
        2, // num_blows
        0x20, // flags
        { // blows
            {19, 2, 5, 0},
            {19, 2, 5, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 158: zombified orc
    {
        110, // speed
        50, // hp
        36, // ac
        11, // depth
        1, // rarity
        30, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 4, 0},
            {19, 1, 4, 0},
            {19, 1, 4, 0},
            {0, 0, 0, 0}
        }
    },
    // 159: drúadan mage
    {
        110, // speed
        52, // hp
        30, // ac
        11, // depth
        2, // rarity
        48, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 5, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 160: old forest tree
    {
        100, // speed
        800, // hp
        50, // ac
        11, // depth
        2, // rarity
        100, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 20, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 161: black mamba
    {
        120, // speed
        45, // hp
        48, // ac
        12, // depth
        3, // rarity
        40, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {27, 4, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 162: white wolf
    {
        120, // speed
        28, // hp
        45, // ac
        12, // depth
        1, // rarity
        30, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 3, 0},
            {19, 1, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 163: grape jelly
    {
        110, // speed
        234, // hp
        1, // ac
        12, // depth
        3, // rarity
        60, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {13, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 164: nether worm mass
    {
        100, // speed
        23, // hp
        22, // ac
        12, // depth
        3, // rarity
        6, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {13, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 165: Golfimbul, the Hill Orc Chief
    {
        110, // speed
        240, // hp
        90, // ac
        12, // depth
        3, // rarity
        230, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {19, 1, 12, 0},
            {19, 1, 12, 0},
            {19, 1, 10, 0},
            {19, 1, 10, 0}
        }
    },
    // 166: master yeek
    {
        110, // speed
        60, // hp
        28, // ac
        12, // depth
        2, // rarity
        28, // exp
        1, // num_blows
        0x02, // flags
        { // blows
            {19, 1, 8, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 167: priest
    {
        110, // speed
        54, // hp
        26, // ac
        12, // depth
        1, // rarity
        36, // exp
        2, // num_blows
        0x40, // flags
        { // blows
            {19, 2, 3, 0},
            {19, 2, 3, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 168: ironfist priest
    {
        120, // speed
        39, // hp
        45, // ac
        12, // depth
        1, // rarity
        50, // exp
        2, // num_blows
        0x42, // flags
        { // blows
            {19, 1, 9, 0},
            {19, 1, 10, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 169: air spirit
    {
        130, // speed
        36, // hp
        48, // ac
        12, // depth
        2, // rarity
        40, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 3, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 170: skeleton human
    {
        110, // speed
        45, // hp
        45, // ac
        12, // depth
        1, // rarity
        38, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 8, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 171: zombified human
    {
        110, // speed
        54, // hp
        36, // ac
        12, // depth
        1, // rarity
        34, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 4, 0},
            {19, 1, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 172: tiger
    {
        120, // speed
        66, // hp
        48, // ac
        12, // depth
        2, // rarity
        40, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 8, 0},
            {19, 1, 8, 0},
            {19, 1, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 173: moaning spirit
    {
        120, // speed
        23, // hp
        24, // ac
        12, // depth
        2, // rarity
        44, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {29, 0, 0, 0},
            {22, 1, 8, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 174: warrior
    {
        110, // speed
        54, // hp
        51, // ac
        12, // depth
        1, // rarity
        40, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 3, 5, 0},
            {19, 3, 5, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 175: stegocentipede
    {
        120, // speed
        59, // hp
        36, // ac
        12, // depth
        2, // rarity
        40, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 4, 0},
            {19, 2, 4, 0},
            {19, 2, 4, 0},
            {0, 0, 0, 0}
        }
    },
    // 176: spotted jelly
    {
        120, // speed
        59, // hp
        27, // ac
        12, // depth
        3, // rarity
        33, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {1, 1, 10, 0},
            {1, 2, 6, 0},
            {1, 2, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 177: killer brown beetle
    {
        110, // speed
        59, // hp
        72, // ac
        13, // depth
        1, // rarity
        45, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 4, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 178: ochre jelly
    {
        120, // speed
        59, // hp
        21, // ac
        13, // depth
        3, // rarity
        40, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {1, 1, 10, 0},
            {1, 2, 6, 0},
            {1, 2, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 179: Boldor, King of the Yeeks
    {
        120, // speed
        180, // hp
        28, // ac
        13, // depth
        3, // rarity
        200, // exp
        3, // num_blows
        0x43, // flags
        { // blows
            {19, 1, 9, 0},
            {19, 1, 9, 0},
            {19, 1, 8, 0},
            {0, 0, 0, 0}
        }
    },
    // 180: ogre
    {
        110, // speed
        65, // hp
        49, // ac
        13, // depth
        2, // rarity
        50, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 8, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 181: creeping mithril coins
    {
        110, // speed
        90, // hp
        60, // ac
        13, // depth
        3, // rarity
        45, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 5, 0},
            {27, 3, 5, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 182: illusionist
    {
        120, // speed
        54, // hp
        15, // ac
        13, // depth
        2, // rarity
        50, // exp
        1, // num_blows
        0x42, // flags
        { // blows
            {19, 2, 2, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 183: druid
    {
        110, // speed
        78, // hp
        15, // ac
        13, // depth
        2, // rarity
        50, // exp
        2, // num_blows
        0x40, // flags
        { // blows
            {19, 2, 4, 0},
            {19, 2, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 184: orc archer
    {
        110, // speed
        66, // hp
        54, // ac
        13, // depth
        1, // rarity
        45, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 4, 0},
            {19, 2, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 185: giant flea
    {
        120, // speed
        3, // hp
        30, // ac
        14, // depth
        3, // rarity
        4, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 2, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 186: Ufthak of Cirith Ungol
    {
        110, // speed
        320, // hp
        75, // ac
        14, // depth
        3, // rarity
        250, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {19, 3, 4, 0},
            {19, 3, 4, 0},
            {19, 3, 4, 0},
            {19, 3, 4, 0}
        }
    },
    // 187: blue icky thing
    {
        100, // speed
        35, // hp
        30, // ac
        14, // depth
        4, // rarity
        20, // exp
        4, // num_blows
        0x02, // flags
        { // blows
            {27, 1, 4, 0},
            {8, 0, 0, 0},
            {19, 1, 4, 0},
            {19, 1, 4, 0}
        }
    },
    // 188: flesh golem
    {
        110, // speed
        54, // hp
        36, // ac
        14, // depth
        2, // rarity
        50, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 6, 0},
            {19, 1, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 189: warg
    {
        120, // speed
        36, // hp
        30, // ac
        14, // depth
        2, // rarity
        40, // exp
        1, // num_blows
        0x02, // flags
        { // blows
            {19, 1, 8, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 190: lurker
    {
        110, // speed
        176, // hp
        30, // ac
        14, // depth
        3, // rarity
        80, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 8, 0},
            {19, 1, 8, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 191: wererat
    {
        110, // speed
        90, // hp
        15, // ac
        15, // depth
        2, // rarity
        45, // exp
        3, // num_blows
        0x02, // flags
        { // blows
            {19, 1, 8, 0},
            {19, 1, 8, 0},
            {19, 2, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 192: black ogre
    {
        110, // speed
        100, // hp
        49, // ac
        15, // depth
        2, // rarity
        75, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 8, 0},
            {19, 2, 8, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 193: magic mushroom patch
    {
        140, // speed
        1, // hp
        12, // ac
        15, // depth
        2, // rarity
        10, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {5, 0, 0, 0},
            {5, 0, 0, 0},
            {18, 0, 0, 0},
            {18, 0, 0, 0}
        }
    },
    // 194: guardian naga
    {
        110, // speed
        144, // hp
        78, // ac
        15, // depth
        2, // rarity
        80, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 8, 0},
            {19, 1, 8, 0},
            {19, 1, 8, 0},
            {0, 0, 0, 0}
        }
    },
    // 195: light hound
    {
        110, // speed
        21, // hp
        36, // ac
        15, // depth
        3, // rarity
        50, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 4, 0},
            {19, 1, 4, 0},
            {19, 1, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 196: dark hound
    {
        110, // speed
        21, // hp
        36, // ac
        15, // depth
        3, // rarity
        50, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 4, 0},
            {19, 1, 4, 0},
            {19, 1, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 197: half-orc
    {
        110, // speed
        88, // hp
        60, // ac
        15, // depth
        2, // rarity
        50, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 3, 4, 0},
            {19, 3, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 198: giant tarantula
    {
        120, // speed
        80, // hp
        48, // ac
        15, // depth
        3, // rarity
        70, // exp
        3, // num_blows
        0x20, // flags
        { // blows
            {27, 1, 6, 0},
            {27, 1, 6, 0},
            {27, 1, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 199: giant clear centipede
    {
        110, // speed
        23, // hp
        36, // ac
        15, // depth
        2, // rarity
        30, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 4, 0},
            {19, 2, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 200: griffon
    {
        110, // speed
        135, // hp
        22, // ac
        15, // depth
        1, // rarity
        70, // exp
        2, // num_blows
        0x20, // flags
        { // blows
            {19, 3, 4, 0},
            {19, 2, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 201: homunculus
    {
        110, // speed
        36, // hp
        48, // ac
        15, // depth
        3, // rarity
        40, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {26, 1, 2, 0},
            {19, 1, 10, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 202: clear hound
    {
        110, // speed
        21, // hp
        36, // ac
        15, // depth
        3, // rarity
        50, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 6, 0},
            {19, 1, 6, 0},
            {19, 1, 8, 0},
            {0, 0, 0, 0}
        }
    },
    // 203: clay golem
    {
        110, // speed
        63, // hp
        36, // ac
        15, // depth
        2, // rarity
        60, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 8, 0},
            {19, 1, 8, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 204: giant tan bat
    {
        130, // speed
        14, // hp
        30, // ac
        15, // depth
        2, // rarity
        18, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {29, 1, 3, 0},
            {19, 1, 2, 0},
            {19, 1, 2, 0},
            {0, 0, 0, 0}
        }
    },
    // 205: umber hulk
    {
        110, // speed
        110, // hp
        75, // ac
        16, // depth
        1, // rarity
        75, // exp
        4, // num_blows
        0x22, // flags
        { // blows
            {5, 0, 0, 0},
            {19, 1, 6, 0},
            {19, 1, 6, 0},
            {19, 2, 6, 0}
        }
    },
    // 206: gelatinous cube
    {
        110, // speed
        316, // hp
        21, // ac
        16, // depth
        4, // rarity
        80, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {1, 1, 10, 0},
            {1, 1, 10, 0},
            {1, 1, 10, 0},
            {0, 0, 0, 0}
        }
    },
    // 207: rogue
    {
        115, // speed
        62, // hp
        48, // ac
        16, // depth
        2, // rarity
        50, // exp
        3, // num_blows
        0x02, // flags
        { // blows
            {19, 2, 4, 0},
            {19, 2, 4, 0},
            {10, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 208: Ulfast, Son of Ulfang
    {
        110, // speed
        340, // hp
        48, // ac
        16, // depth
        3, // rarity
        200, // exp
        4, // num_blows
        0x03, // flags
        { // blows
            {19, 3, 5, 0},
            {19, 3, 5, 0},
            {19, 3, 5, 0},
            {19, 3, 5, 0}
        }
    },
    // 209: quasit
    {
        110, // speed
        27, // hp
        36, // ac
        16, // depth
        2, // rarity
        50, // exp
        3, // num_blows
        0x40, // flags
        { // blows
            {22, 1, 6, 0},
            {19, 1, 3, 0},
            {19, 1, 3, 0},
            {0, 0, 0, 0}
        }
    },
    // 210: uruk
    {
        110, // speed
        70, // hp
        75, // ac
        16, // depth
        1, // rarity
        60, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 3, 5, 0},
            {19, 3, 5, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 211: grizzly bear
    {
        110, // speed
        78, // hp
        52, // ac
        16, // depth
        2, // rarity
        55, // exp
        4, // num_blows
        0x20, // flags
        { // blows
            {19, 1, 8, 0},
            {19, 1, 8, 0},
            {19, 1, 12, 0},
            {19, 1, 10, 0}
        }
    },
    // 212: craban
    {
        120, // speed
        9, // hp
        14, // ac
        16, // depth
        4, // rarity
        20, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 4, 0},
            {19, 1, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 213: imp
    {
        110, // speed
        27, // hp
        36, // ac
        17, // depth
        2, // rarity
        55, // exp
        2, // num_blows
        0x40, // flags
        { // blows
            {27, 3, 4, 0},
            {27, 3, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 214: forest troll
    {
        110, // speed
        110, // hp
        75, // ac
        17, // depth
        1, // rarity
        70, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 4, 0},
            {19, 1, 4, 0},
            {19, 1, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 215: Nár, the Dwarf
    {
        110, // speed
        450, // hp
        84, // ac
        17, // depth
        2, // rarity
        250, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {19, 3, 5, 0},
            {19, 3, 5, 0},
            {19, 3, 5, 0},
            {19, 3, 5, 0}
        }
    },
    // 216: 2-headed hydra
    {
        110, // speed
        200, // hp
        90, // ac
        17, // depth
        2, // rarity
        80, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 6, 0},
            {19, 2, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 217: water spirit
    {
        120, // speed
        41, // hp
        42, // ac
        17, // depth
        2, // rarity
        58, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 4, 0},
            {19, 2, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 218: giant red scorpion
    {
        115, // speed
        50, // hp
        52, // ac
        17, // depth
        1, // rarity
        62, // exp
        3, // num_blows
        0x20, // flags
        { // blows
            {19, 2, 4, 0},
            {19, 2, 4, 0},
            {24, 1, 7, 0},
            {0, 0, 0, 0}
        }
    },
    // 219: earth spirit
    {
        120, // speed
        59, // hp
        60, // ac
        17, // depth
        2, // rarity
        64, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 8, 0},
            {19, 1, 8, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 220: hummerhorn
    {
        120, // speed
        3, // hp
        16, // ac
        18, // depth
        4, // rarity
        4, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {5, 2, 2, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 221: orc captain
    {
        110, // speed
        110, // hp
        88, // ac
        18, // depth
        3, // rarity
        80, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 3, 5, 0},
            {19, 3, 5, 0},
            {19, 3, 5, 0},
            {0, 0, 0, 0}
        }
    },
    // 222: blackguard
    {
        115, // speed
        120, // hp
        85, // ac
        18, // depth
        1, // rarity
        180, // exp
        3, // num_blows
        0x02, // flags
        { // blows
            {19, 3, 5, 0},
            {19, 3, 5, 0},
            {19, 3, 5, 0},
            {0, 0, 0, 0}
        }
    },
    // 223: fire spirit
    {
        120, // speed
        50, // hp
        36, // ac
        18, // depth
        2, // rarity
        75, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {17, 2, 6, 0},
            {17, 2, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 224: fire hound
    {
        110, // speed
        35, // hp
        36, // ac
        18, // depth
        3, // rarity
        70, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 3, 3, 0},
            {19, 3, 3, 0},
            {17, 2, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 225: cold hound
    {
        110, // speed
        35, // hp
        36, // ac
        18, // depth
        3, // rarity
        70, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 3, 3, 0},
            {19, 3, 3, 0},
            {4, 2, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 226: energy hound
    {
        110, // speed
        35, // hp
        36, // ac
        18, // depth
        3, // rarity
        70, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 3, 3, 0},
            {19, 3, 3, 0},
            {12, 2, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 227: potion mimic
    {
        110, // speed
        55, // hp
        36, // ac
        18, // depth
        3, // rarity
        60, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {27, 3, 4, 0},
            {19, 2, 3, 0},
            {19, 2, 3, 0},
            {0, 0, 0, 0}
        }
    },
    // 228: blink dog
    {
        120, // speed
        36, // hp
        24, // ac
        18, // depth
        2, // rarity
        50, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 8, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 229: shambling mound
    {
        110, // speed
        70, // hp
        19, // ac
        18, // depth
        2, // rarity
        75, // exp
        2, // num_blows
        0x02, // flags
        { // blows
            {19, 1, 8, 0},
            {19, 1, 8, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 230: evil eye
    {
        110, // speed
        68, // hp
        7, // ac
        18, // depth
        3, // rarity
        80, // exp
        2, // num_blows
        0x02, // flags
        { // blows
            {13, 0, 0, 0},
            {13, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 231: Shagrat, the Orc Captain
    {
        110, // speed
        400, // hp
        72, // ac
        19, // depth
        2, // rarity
        400, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {19, 3, 8, 0},
            {19, 3, 8, 0},
            {19, 3, 5, 0},
            {19, 3, 5, 0}
        }
    },
    // 232: Gorbag, the Orc Captain
    {
        110, // speed
        400, // hp
        72, // ac
        19, // depth
        2, // rarity
        400, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {19, 3, 8, 0},
            {19, 3, 8, 0},
            {19, 3, 5, 0},
            {19, 3, 5, 0}
        }
    },
    // 233: stone golem
    {
        100, // speed
        126, // hp
        90, // ac
        19, // depth
        2, // rarity
        100, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 10, 0},
            {19, 1, 10, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 234: red mold
    {
        110, // speed
        77, // hp
        19, // ac
        19, // depth
        1, // rarity
        64, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {17, 4, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 235: Old Man Willow
    {
        110, // speed
        1000, // hp
        100, // ac
        19, // depth
        3, // rarity
        3000, // exp
        3, // num_blows
        0x01, // flags
        { // blows
            {26, 1, 14, 0},
            {26, 1, 14, 0},
            {19, 2, 12, 0},
            {0, 0, 0, 0}
        }
    },
    // 236: blood falcon
    {
        140, // speed
        6, // hp
        2, // ac
        20, // depth
        2, // rarity
        50, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 10, 0},
            {19, 2, 10, 0},
            {19, 3, 8, 0},
            {0, 0, 0, 0}
        }
    },
    // 237: Mirkwood spider
    {
        120, // speed
        41, // hp
        30, // ac
        20, // depth
        2, // rarity
        25, // exp
        3, // num_blows
        0x22, // flags
        { // blows
            {19, 1, 8, 0},
            {27, 1, 6, 0},
            {27, 1, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 238: Bolg, Son of Azog
    {
        120, // speed
        500, // hp
        60, // ac
        20, // depth
        4, // rarity
        800, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {19, 3, 6, 0},
            {19, 3, 6, 0},
            {19, 3, 6, 0},
            {19, 3, 6, 0}
        }
    },
    // 239: 3-headed hydra
    {
        120, // speed
        300, // hp
        97, // ac
        20, // depth
        2, // rarity
        350, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 6, 0},
            {19, 2, 6, 0},
            {27, 2, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 240: earth hound
    {
        110, // speed
        68, // hp
        36, // ac
        20, // depth
        3, // rarity
        200, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 8, 0},
            {19, 1, 8, 0},
            {19, 3, 3, 0},
            {19, 3, 3, 0}
        }
    },
    // 241: air hound
    {
        110, // speed
        68, // hp
        36, // ac
        20, // depth
        3, // rarity
        200, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {27, 1, 8, 0},
            {27, 1, 8, 0},
            {19, 3, 3, 0},
            {19, 3, 3, 0}
        }
    },
    // 242: sabre-tooth tiger
    {
        120, // speed
        150, // hp
        60, // ac
        20, // depth
        2, // rarity
        120, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 10, 0},
            {19, 1, 10, 0},
            {19, 1, 10, 0},
            {19, 1, 10, 0}
        }
    },
    // 243: water hound
    {
        110, // speed
        68, // hp
        36, // ac
        20, // depth
        3, // rarity
        200, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {1, 1, 8, 0},
            {1, 1, 8, 0},
            {19, 3, 3, 0},
            {19, 3, 3, 0}
        }
    },
    // 244: chimaera
    {
        110, // speed
        160, // hp
        22, // ac
        20, // depth
        2, // rarity
        200, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 8, 0},
            {19, 2, 10, 0},
            {17, 2, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 245: quylthulg
    {
        110, // speed
        27, // hp
        1, // ac
        20, // depth
        1, // rarity
        250, // exp
        0, // num_blows
        0x00, // flags
        { // blows
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 246: sasquatch
    {
        120, // speed
        200, // hp
        60, // ac
        20, // depth
        3, // rarity
        180, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 10, 0},
            {19, 1, 10, 0},
            {19, 2, 8, 0},
            {0, 0, 0, 0}
        }
    },
    // 247: werewolf
    {
        110, // speed
        230, // hp
        36, // ac
        20, // depth
        1, // rarity
        150, // exp
        3, // num_blows
        0x02, // flags
        { // blows
            {19, 2, 6, 0},
            {19, 2, 6, 0},
            {19, 2, 10, 0},
            {0, 0, 0, 0}
        }
    },
    // 248: dark dwarven lord
    {
        120, // speed
        144, // hp
        48, // ac
        20, // depth
        2, // rarity
        500, // exp
        2, // num_blows
        0x02, // flags
        { // blows
            {19, 3, 8, 0},
            {19, 3, 5, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 249: ranger
    {
        110, // speed
        90, // hp
        60, // ac
        20, // depth
        1, // rarity
        55, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 5, 4, 0},
            {19, 5, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 250: paladin
    {
        110, // speed
        90, // hp
        48, // ac
        20, // depth
        1, // rarity
        55, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 4, 6, 0},
            {19, 4, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 251: Lugdush, the Uruk
    {
        110, // speed
        640, // hp
        108, // ac
        21, // depth
        4, // rarity
        550, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {19, 3, 7, 0},
            {19, 3, 7, 0},
            {19, 3, 5, 0},
            {19, 3, 5, 0}
        }
    },
    // 252: blue dragon bat
    {
        130, // speed
        10, // hp
        39, // ac
        21, // depth
        1, // rarity
        54, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {12, 1, 3, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 253: scroll mimic
    {
        110, // speed
        75, // hp
        48, // ac
        21, // depth
        3, // rarity
        70, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {27, 3, 4, 0},
            {27, 3, 4, 0},
            {19, 2, 3, 0},
            {19, 2, 3, 0}
        }
    },
    // 254: fire vortex
    {
        110, // speed
        45, // hp
        36, // ac
        21, // depth
        1, // rarity
        100, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {17, 3, 3, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 255: water vortex
    {
        110, // speed
        45, // hp
        36, // ac
        21, // depth
        1, // rarity
        100, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {1, 3, 3, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 256: cold vortex
    {
        110, // speed
        45, // hp
        36, // ac
        21, // depth
        1, // rarity
        100, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {4, 3, 3, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 257: mummified orc
    {
        110, // speed
        86, // hp
        33, // ac
        21, // depth
        1, // rarity
        56, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 4, 0},
            {19, 2, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 258: neekerbreeker
    {
        120, // speed
        5, // hp
        21, // ac
        22, // depth
        4, // rarity
        4, // exp
        1, // num_blows
        0x02, // flags
        { // blows
            {27, 2, 2, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 259: Uglúk, the Uruk
    {
        110, // speed
        720, // hp
        114, // ac
        22, // depth
        3, // rarity
        600, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {19, 3, 8, 0},
            {19, 3, 8, 0},
            {19, 3, 5, 0},
            {19, 3, 5, 0}
        }
    },
    // 260: killer stag beetle
    {
        110, // speed
        68, // hp
        86, // ac
        22, // depth
        1, // rarity
        80, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 3, 6, 0},
            {19, 3, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 261: iron golem
    {
        110, // speed
        520, // hp
        120, // ac
        22, // depth
        2, // rarity
        160, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 12, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 262: giant yellow scorpion
    {
        110, // speed
        54, // hp
        45, // ac
        22, // depth
        1, // rarity
        60, // exp
        3, // num_blows
        0x20, // flags
        { // blows
            {19, 1, 8, 0},
            {19, 1, 8, 0},
            {27, 2, 5, 0},
            {0, 0, 0, 0}
        }
    },
    // 263: wyvern
    {
        120, // speed
        203, // hp
        79, // ac
        22, // depth
        2, // rarity
        250, // exp
        3, // num_blows
        0x20, // flags
        { // blows
            {19, 2, 6, 0},
            {19, 2, 6, 0},
            {27, 2, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 264: phase spider
    {
        120, // speed
        27, // hp
        30, // ac
        23, // depth
        2, // rarity
        60, // exp
        3, // num_blows
        0x20, // flags
        { // blows
            {19, 1, 8, 0},
            {27, 1, 6, 0},
            {27, 1, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 265: black ooze
    {
        90, // speed
        27, // hp
        7, // ac
        23, // depth
        1, // rarity
        7, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {1, 2, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 266: Easterling champion
    {
        110, // speed
        90, // hp
        60, // ac
        23, // depth
        1, // rarity
        60, // exp
        2, // num_blows
        0x02, // flags
        { // blows
            {19, 6, 5, 0},
            {19, 6, 5, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 267: Azog, Enemy of the Dwarves
    {
        120, // speed
        900, // hp
        96, // ac
        23, // depth
        5, // rarity
        1111, // exp
        3, // num_blows
        0x01, // flags
        { // blows
            {19, 5, 5, 0},
            {19, 5, 5, 0},
            {19, 5, 5, 0},
            {0, 0, 0, 0}
        }
    },
    // 268: brigand
    {
        120, // speed
        75, // hp
        45, // ac
        23, // depth
        2, // rarity
        110, // exp
        4, // num_blows
        0x02, // flags
        { // blows
            {19, 2, 8, 0},
            {19, 2, 8, 0},
            {9, 4, 4, 0},
            {10, 4, 4, 0}
        }
    },
    // 269: red dragon bat
    {
        130, // speed
        14, // hp
        42, // ac
        23, // depth
        1, // rarity
        60, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {17, 1, 3, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 270: giant silver ant
    {
        110, // speed
        41, // hp
        100, // ac
        23, // depth
        1, // rarity
        80, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {1, 4, 4, 0},
            {1, 4, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 271: forest wight
    {
        110, // speed
        54, // hp
        36, // ac
        24, // depth
        1, // rarity
        140, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 6, 0},
            {19, 1, 6, 0},
            {14, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 272: Ibun, Son of Mîm
    {
        110, // speed
        820, // hp
        96, // ac
        24, // depth
        2, // rarity
        300, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {19, 3, 6, 0},
            {19, 3, 6, 0},
            {19, 3, 6, 0},
            {6, 0, 0, 0}
        }
    },
    // 273: Khîm, Son of Mîm
    {
        110, // speed
        820, // hp
        96, // ac
        24, // depth
        2, // rarity
        300, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {19, 3, 6, 0},
            {19, 3, 6, 0},
            {19, 3, 6, 0},
            {6, 0, 0, 0}
        }
    },
    // 274: 4-headed hydra
    {
        120, // speed
        350, // hp
        105, // ac
        24, // depth
        2, // rarity
        450, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 6, 0},
            {19, 2, 6, 0},
            {27, 2, 6, 0},
            {17, 2, 6, 0}
        }
    },
    // 275: mummified human
    {
        110, // speed
        105, // hp
        51, // ac
        24, // depth
        1, // rarity
        70, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 4, 0},
            {19, 2, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 276: vampire bat
    {
        120, // speed
        50, // hp
        60, // ac
        24, // depth
        2, // rarity
        150, // exp
        2, // num_blows
        0x86, // flags
        { // blows
            {15, 1, 4, 0},
            {15, 1, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 277: Sangahyando of Umbar
    {
        110, // speed
        800, // hp
        96, // ac
        24, // depth
        2, // rarity
        400, // exp
        4, // num_blows
        0x03, // flags
        { // blows
            {19, 4, 6, 0},
            {19, 4, 6, 0},
            {19, 4, 6, 0},
            {19, 4, 6, 0}
        }
    },
    // 278: Angamaitë of Umbar
    {
        110, // speed
        800, // hp
        96, // ac
        24, // depth
        2, // rarity
        400, // exp
        4, // num_blows
        0x03, // flags
        { // blows
            {19, 4, 6, 0},
            {19, 4, 6, 0},
            {19, 4, 6, 0},
            {19, 4, 6, 0}
        }
    },
    // 279: banshee
    {
        120, // speed
        27, // hp
        28, // ac
        24, // depth
        2, // rarity
        60, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {29, 0, 0, 0},
            {14, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 280: werebear
    {
        110, // speed
        325, // hp
        75, // ac
        24, // depth
        2, // rarity
        200, // exp
        4, // num_blows
        0x22, // flags
        { // blows
            {19, 1, 10, 0},
            {19, 1, 10, 0},
            {19, 2, 8, 0},
            {19, 2, 6, 0}
        }
    },
    // 281: nruling
    {
        120, // speed
        50, // hp
        16, // ac
        25, // depth
        2, // rarity
        75, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {17, 1, 8, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 282: necromancer
    {
        110, // speed
        164, // hp
        55, // ac
        25, // depth
        2, // rarity
        330, // exp
        2, // num_blows
        0x42, // flags
        { // blows
            {19, 2, 4, 0},
            {19, 2, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 283: hill giant
    {
        110, // speed
        240, // hp
        54, // ac
        25, // depth
        1, // rarity
        150, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 4, 8, 0},
            {19, 4, 8, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 284: pukelman
    {
        110, // speed
        520, // hp
        120, // ac
        25, // depth
        3, // rarity
        600, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 12, 0},
            {19, 3, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 285: stone troll
    {
        110, // speed
        127, // hp
        60, // ac
        25, // depth
        1, // rarity
        85, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 6, 0},
            {19, 1, 6, 0},
            {19, 3, 4, 0},
            {0, 0, 0, 0}
        }
    },
    // 286: carrion crawler
    {
        110, // speed
        130, // hp
        60, // ac
        25, // depth
        2, // rarity
        60, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {26, 2, 6, 0},
            {26, 2, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 287: killer red beetle
    {
        110, // speed
        90, // hp
        90, // ac
        25, // depth
        1, // rarity
        90, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {24, 4, 4, 0},
            {19, 4, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 288: abyss worm mass
    {
        100, // speed
        35, // hp
        25, // ac
        26, // depth
        5, // rarity
        6, // exp
        1, // num_blows
        0x02, // flags
        { // blows
            {13, 1, 5, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 289: giant firefly
    {
        120, // speed
        5, // hp
        21, // ac
        26, // depth
        4, // rarity
        4, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {3, 1, 2, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 290: giant grey ant
    {
        110, // speed
        86, // hp
        200, // ac
        26, // depth
        1, // rarity
        120, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 12, 0},
            {19, 2, 12, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 291: Ulwarth, Son of Ulfang
    {
        110, // speed
        850, // hp
        48, // ac
        26, // depth
        4, // rarity
        500, // exp
        3, // num_blows
        0x03, // flags
        { // blows
            {19, 4, 6, 0},
            {19, 4, 6, 0},
            {19, 4, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 292: displacer beast
    {
        110, // speed
        138, // hp
        150, // ac
        26, // depth
        2, // rarity
        100, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 8, 0},
            {19, 1, 10, 0},
            {19, 1, 10, 0},
            {19, 1, 10, 0}
        }
    },
    // 293: giant fire tick
    {
        110, // speed
        72, // hp
        120, // ac
        26, // depth
        1, // rarity
        90, // exp
        1, // num_blows
        0x20, // flags
        { // blows
            {17, 3, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 294: cave ogre
    {
        110, // speed
        150, // hp
        49, // ac
        26, // depth
        2, // rarity
        80, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 3, 8, 0},
            {19, 3, 8, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 295: disenchanter bat
    {
        130, // speed
        27, // hp
        42, // ac
        26, // depth
        4, // rarity
        75, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {6, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 296: wolf chieftain
    {
        120, // speed
        422, // hp
        24, // ac
        26, // depth
        5, // rarity
        120, // exp
        4, // num_blows
        0x42, // flags
        { // blows
            {19, 2, 6, 0},
            {19, 2, 6, 0},
            {19, 2, 10, 0},
            {29, 0, 0, 0}
        }
    },
    // 297: ghoul
    {
        110, // speed
        83, // hp
        36, // ac
        26, // depth
        2, // rarity
        95, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {26, 1, 4, 0},
            {26, 1, 4, 0},
            {27, 1, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 298: Mîm, Betrayer of Turin
    {
        120, // speed
        1100, // hp
        96, // ac
        27, // depth
        4, // rarity
        1000, // exp
        4, // num_blows
        0x03, // flags
        { // blows
            {19, 3, 8, 0},
            {19, 3, 8, 0},
            {19, 3, 8, 0},
            {6, 0, 0, 0}
        }
    },
    // 299: killer fire beetle
    {
        110, // speed
        99, // hp
        64, // ac
        27, // depth
        1, // rarity
        95, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {17, 3, 4, 0},
            {17, 4, 5, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 300: creeping adamantite coins
    {
        120, // speed
        260, // hp
        60, // ac
        27, // depth
        3, // rarity
        60, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {27, 3, 4, 0},
            {27, 3, 5, 0},
            {19, 1, 12, 0},
            {19, 1, 12, 0}
        }
    },
    // 301: troll scavenger
    {
        110, // speed
        137, // hp
        90, // ac
        27, // depth
        1, // rarity
        150, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {27, 3, 3, 0},
            {27, 3, 3, 0},
            {19, 1, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 302: vibration hound
    {
        110, // speed
        138, // hp
        36, // ac
        27, // depth
        4, // rarity
        250, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 8, 0},
            {19, 2, 8, 0},
            {19, 3, 3, 0},
            {19, 3, 3, 0}
        }
    },
    // 303: nexus hound
    {
        110, // speed
        138, // hp
        36, // ac
        27, // depth
        4, // rarity
        250, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 8, 0},
            {19, 2, 8, 0},
            {19, 3, 3, 0},
            {19, 3, 3, 0}
        }
    },
    // 304: vampire
    {
        110, // speed
        163, // hp
        67, // ac
        27, // depth
        1, // rarity
        175, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 6, 0},
            {19, 1, 6, 0},
            {14, 2, 6, 0},
            {14, 2, 6, 0}
        }
    },
    // 305: gorgimaera
    {
        110, // speed
        263, // hp
        82, // ac
        27, // depth
        2, // rarity
        400, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 10, 0},
            {17, 2, 10, 0},
            {26, 2, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 306: ogre shaman
    {
        110, // speed
        163, // hp
        82, // ac
        27, // depth
        2, // rarity
        300, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 3, 6, 0},
            {19, 3, 6, 0},
            {19, 3, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 307: shimmering mold
    {
        110, // speed
        144, // hp
        36, // ac
        27, // depth
        1, // rarity
        140, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {12, 5, 4, 0},
            {12, 5, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 308: frost giant
    {
        110, // speed
        256, // hp
        60, // ac
        28, // depth
        1, // rarity
        180, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {4, 5, 8, 0},
            {4, 5, 8, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 309: spirit naga
    {
        110, // speed
        240, // hp
        90, // ac
        28, // depth
        2, // rarity
        60, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 8, 0},
            {19, 2, 8, 0},
            {19, 1, 8, 0},
            {19, 1, 8, 0}
        }
    },
    // 310: 5-headed hydra
    {
        120, // speed
        450, // hp
        120, // ac
        28, // depth
        2, // rarity
        650, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 6, 0},
            {19, 2, 6, 0},
            {27, 2, 6, 0},
            {17, 2, 6, 0}
        }
    },
    // 311: black knight
    {
        120, // speed
        165, // hp
        105, // ac
        28, // depth
        1, // rarity
        240, // exp
        3, // num_blows
        0x02, // flags
        { // blows
            {19, 5, 5, 0},
            {19, 5, 5, 0},
            {19, 5, 5, 0},
            {0, 0, 0, 0}
        }
    },
    // 312: Uldor the Accursed
    {
        110, // speed
        1000, // hp
        84, // ac
        28, // depth
        4, // rarity
        600, // exp
        4, // num_blows
        0x03, // flags
        { // blows
            {19, 4, 6, 0},
            {19, 4, 6, 0},
            {19, 4, 6, 0},
            {19, 4, 6, 0}
        }
    },
    // 313: mage
    {
        110, // speed
        68, // hp
        48, // ac
        28, // depth
        1, // rarity
        150, // exp
        2, // num_blows
        0x40, // flags
        { // blows
            {19, 2, 5, 0},
            {19, 2, 5, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 314: mind flayer
    {
        110, // speed
        132, // hp
        72, // ac
        28, // depth
        1, // rarity
        200, // exp
        2, // num_blows
        0x02, // flags
        { // blows
            {23, 2, 6, 0},
            {25, 2, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 315: Draebor, the Imp
    {
        120, // speed
        520, // hp
        75, // ac
        28, // depth
        5, // rarity
        750, // exp
        4, // num_blows
        0x41, // flags
        { // blows
            {27, 3, 4, 0},
            {27, 3, 4, 0},
            {19, 3, 4, 0},
            {0, 0, 0, 0}
        }
    },
    // 316: basilisk
    {
        120, // speed
        310, // hp
        108, // ac
        28, // depth
        3, // rarity
        300, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {26, 0, 0, 0},
            {19, 2, 12, 0},
            {19, 2, 12, 0},
            {19, 2, 12, 0}
        }
    },
    // 317: snow troll
    {
        110, // speed
        132, // hp
        67, // ac
        28, // depth
        1, // rarity
        200, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 5, 0},
            {19, 1, 5, 0},
            {4, 2, 6, 0},
            {4, 2, 6, 0}
        }
    },
    // 318: bat of Gorgoroth
    {
        120, // speed
        110, // hp
        36, // ac
        28, // depth
        3, // rarity
        100, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {27, 1, 10, 0},
            {19, 1, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 319: spectator
    {
        110, // speed
        171, // hp
        1, // ac
        28, // depth
        2, // rarity
        150, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {26, 1, 4, 0},
            {5, 1, 4, 0},
            {19, 1, 8, 0},
            {0, 0, 0, 0}
        }
    },
    // 320: Beorn, the Shape-Changer
    {
        110, // speed
        1400, // hp
        72, // ac
        28, // depth
        3, // rarity
        1000, // exp
        4, // num_blows
        0x41, // flags
        { // blows
            {19, 2, 8, 0},
            {19, 2, 8, 0},
            {19, 2, 6, 0},
            {19, 2, 6, 0}
        }
    },
    // 321: Beorn, the Mountain Bear
    {
        120, // speed
        1, // hp
        72, // ac
        28, // depth
        0, // rarity
        1000, // exp
        4, // num_blows
        0x61, // flags
        { // blows
            {19, 2, 10, 0},
            {19, 2, 10, 0},
            {19, 4, 8, 0},
            {19, 3, 6, 0}
        }
    },
    // 322: green elf archer
    {
        120, // speed
        210, // hp
        112, // ac
        29, // depth
        3, // rarity
        500, // exp
        2, // num_blows
        0x40, // flags
        { // blows
            {19, 1, 7, 0},
            {19, 1, 7, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 323: giant black scorpion
    {
        120, // speed
        189, // hp
        60, // ac
        29, // depth
        3, // rarity
        425, // exp
        4, // num_blows
        0x20, // flags
        { // blows
            {19, 1, 11, 0},
            {19, 1, 11, 0},
            {27, 3, 4, 0},
            {3, 3, 4, 0}
        }
    },
    // 324: purple worm
    {
        110, // speed
        293, // hp
        78, // ac
        29, // depth
        3, // rarity
        400, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 8, 0},
            {1, 2, 8, 0},
            {27, 1, 8, 0},
            {0, 0, 0, 0}
        }
    },
    // 325: catoblepas
    {
        110, // speed
        165, // hp
        66, // ac
        29, // depth
        2, // rarity
        400, // exp
        4, // num_blows
        0x20, // flags
        { // blows
            {29, 2, 4, 0},
            {3, 2, 4, 0},
            {19, 2, 6, 0},
            {19, 2, 12, 0}
        }
    },
    // 326: ring mimic
    {
        120, // speed
        180, // hp
        72, // ac
        29, // depth
        3, // rarity
        200, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {27, 3, 4, 0},
            {27, 3, 4, 0},
            {27, 3, 4, 0},
            {27, 3, 4, 0}
        }
    },
    // 327: young blue dragon
    {
        110, // speed
        237, // hp
        60, // ac
        29, // depth
        1, // rarity
        500, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 4, 0},
            {19, 2, 4, 0},
            {19, 2, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 328: young white dragon
    {
        110, // speed
        237, // hp
        60, // ac
        29, // depth
        1, // rarity
        500, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 4, 0},
            {19, 2, 4, 0},
            {19, 2, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 329: young green dragon
    {
        110, // speed
        237, // hp
        60, // ac
        29, // depth
        1, // rarity
        500, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 4, 0},
            {19, 2, 4, 0},
            {19, 2, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 330: energy vortex
    {
        110, // speed
        65, // hp
        36, // ac
        29, // depth
        1, // rarity
        140, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {12, 3, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 331: fire giant
    {
        110, // speed
        289, // hp
        72, // ac
        30, // depth
        1, // rarity
        220, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {17, 6, 8, 0},
            {17, 6, 8, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 332: mithril golem
    {
        110, // speed
        640, // hp
        150, // ac
        30, // depth
        4, // rarity
        500, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 3, 8, 0},
            {19, 3, 8, 0},
            {19, 5, 5, 0},
            {19, 5, 5, 0}
        }
    },
    // 333: skeleton troll
    {
        110, // speed
        110, // hp
        82, // ac
        30, // depth
        1, // rarity
        225, // exp
        3, // num_blows
        0x80, // flags
        { // blows
            {19, 1, 6, 0},
            {19, 1, 6, 0},
            {19, 3, 4, 0},
            {0, 0, 0, 0}
        }
    },
    // 334: manticore
    {
        120, // speed
        220, // hp
        22, // ac
        30, // depth
        2, // rarity
        300, // exp
        4, // num_blows
        0x02, // flags
        { // blows
            {19, 3, 4, 0},
            {19, 3, 4, 0},
            {19, 3, 4, 0},
            {19, 3, 4, 0}
        }
    },
    // 335: giant blue ant
    {
        110, // speed
        36, // hp
        75, // ac
        30, // depth
        2, // rarity
        80, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {12, 5, 5, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 336: giant army ant
    {
        120, // speed
        67, // hp
        60, // ac
        30, // depth
        3, // rarity
        90, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 12, 0},
            {19, 2, 12, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 337: grave wight
    {
        110, // speed
        66, // hp
        60, // ac
        30, // depth
        1, // rarity
        325, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 7, 0},
            {19, 1, 7, 0},
            {14, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 338: killer slicer beetle
    {
        110, // speed
        138, // hp
        92, // ac
        30, // depth
        2, // rarity
        250, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 7, 8, 0},
            {19, 7, 8, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 339: ogre chieftain
    {
        120, // speed
        240, // hp
        66, // ac
        30, // depth
        5, // rarity
        600, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 3, 8, 0},
            {19, 3, 8, 0},
            {19, 3, 8, 0},
            {19, 3, 8, 0}
        }
    },
    // 340: ghast
    {
        120, // speed
        165, // hp
        60, // ac
        30, // depth
        3, // rarity
        130, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {26, 2, 4, 0},
            {26, 2, 4, 0},
            {21, 2, 4, 0},
            {0, 0, 0, 0}
        }
    },
    // 341: Bert the Stone Troll
    {
        120, // speed
        1100, // hp
        84, // ac
        30, // depth
        7, // rarity
        2000, // exp
        3, // num_blows
        0x01, // flags
        { // blows
            {19, 6, 6, 0},
            {19, 6, 6, 0},
            {19, 3, 10, 0},
            {0, 0, 0, 0}
        }
    },
    // 342: Bill the Stone Troll
    {
        120, // speed
        1100, // hp
        84, // ac
        30, // depth
        7, // rarity
        2000, // exp
        3, // num_blows
        0x01, // flags
        { // blows
            {19, 6, 6, 0},
            {19, 6, 6, 0},
            {19, 3, 10, 0},
            {0, 0, 0, 0}
        }
    },
    // 343: Tom the Stone Troll
    {
        120, // speed
        1100, // hp
        84, // ac
        30, // depth
        7, // rarity
        2000, // exp
        3, // num_blows
        0x01, // flags
        { // blows
            {19, 6, 6, 0},
            {19, 6, 6, 0},
            {19, 3, 10, 0},
            {0, 0, 0, 0}
        }
    },
    // 344: ghost
    {
        120, // speed
        59, // hp
        36, // ac
        31, // depth
        1, // rarity
        350, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {29, 0, 0, 0},
            {14, 0, 0, 0},
            {23, 1, 6, 0},
            {25, 1, 6, 0}
        }
    },
    // 345: death watch beetle
    {
        110, // speed
        163, // hp
        72, // ac
        31, // depth
        3, // rarity
        300, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 5, 6, 0},
            {19, 5, 6, 0},
            {29, 5, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 346: young black dragon
    {
        110, // speed
        264, // hp
        72, // ac
        31, // depth
        1, // rarity
        700, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 6, 0},
            {19, 2, 6, 0},
            {19, 2, 8, 0},
            {0, 0, 0, 0}
        }
    },
    // 347: young gold dragon
    {
        110, // speed
        264, // hp
        72, // ac
        31, // depth
        1, // rarity
        700, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 6, 0},
            {19, 2, 6, 0},
            {19, 2, 8, 0},
            {0, 0, 0, 0}
        }
    },
    // 348: young red dragon
    {
        110, // speed
        264, // hp
        72, // ac
        31, // depth
        1, // rarity
        700, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 6, 0},
            {19, 2, 6, 0},
            {19, 2, 8, 0},
            {0, 0, 0, 0}
        }
    },
    // 349: huorn
    {
        115, // speed
        1000, // hp
        80, // ac
        31, // depth
        1, // rarity
        1200, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 3, 6, 0},
            {19, 3, 6, 0},
            {19, 3, 6, 0},
            {19, 3, 6, 0}
        }
    },
    // 350: serpent of the brownlands
    {
        120, // speed
        150, // hp
        90, // ac
        32, // depth
        2, // rarity
        300, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {3, 1, 6, 0},
            {1, 2, 12, 0},
            {27, 2, 12, 0},
            {19, 4, 4, 0}
        }
    },
    // 351: ogre mage
    {
        115, // speed
        220, // hp
        60, // ac
        32, // depth
        2, // rarity
        300, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 3, 8, 0},
            {19, 3, 8, 0},
            {19, 3, 8, 0},
            {19, 3, 8, 0}
        }
    },
    // 352: nexus quylthulg
    {
        110, // speed
        65, // hp
        1, // ac
        32, // depth
        1, // rarity
        300, // exp
        0, // num_blows
        0x00, // flags
        { // blows
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 353: southron assassin
    {
        120, // speed
        85, // hp
        72, // ac
        32, // depth
        2, // rarity
        300, // exp
        3, // num_blows
        0x02, // flags
        { // blows
            {27, 3, 4, 0},
            {24, 3, 4, 0},
            {24, 3, 4, 0},
            {0, 0, 0, 0}
        }
    },
    // 354: memory moss
    {
        110, // speed
        2, // hp
        1, // ac
        32, // depth
        3, // rarity
        150, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {5, 1, 4, 0},
            {5, 1, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 355: young multi-hued dragon
    {
        110, // speed
        281, // hp
        72, // ac
        32, // depth
        1, // rarity
        900, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 6, 0},
            {19, 2, 6, 0},
            {19, 3, 8, 0},
            {0, 0, 0, 0}
        }
    },
    // 356: doombat
    {
        130, // speed
        180, // hp
        112, // ac
        32, // depth
        2, // rarity
        250, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {17, 5, 4, 0},
            {17, 5, 4, 0},
            {17, 5, 4, 0},
            {0, 0, 0, 0}
        }
    },
    // 357: drúadan druid
    {
        120, // speed
        240, // hp
        112, // ac
        33, // depth
        3, // rarity
        600, // exp
        3, // num_blows
        0x40, // flags
        { // blows
            {19, 1, 7, 0},
            {19, 1, 7, 0},
            {19, 3, 8, 0},
            {0, 0, 0, 0}
        }
    },
    // 358: stone giant
    {
        110, // speed
        333, // hp
        90, // ac
        33, // depth
        1, // rarity
        250, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 7, 8, 0},
            {19, 7, 8, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 359: shadow drake
    {
        110, // speed
        264, // hp
        84, // ac
        33, // depth
        3, // rarity
        1100, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 4, 0},
            {19, 2, 4, 0},
            {13, 3, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 360: cave troll
    {
        110, // speed
        156, // hp
        60, // ac
        33, // depth
        1, // rarity
        350, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 3, 5, 0},
            {19, 3, 5, 0},
            {19, 1, 8, 0},
            {19, 1, 8, 0}
        }
    },
    // 361: mystic
    {
        120, // speed
        308, // hp
        60, // ac
        33, // depth
        3, // rarity
        500, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 10, 2, 0},
            {19, 10, 2, 0},
            {19, 10, 2, 0},
            {19, 10, 2, 0}
        }
    },
    // 362: barrow wight
    {
        110, // speed
        83, // hp
        60, // ac
        33, // depth
        3, // rarity
        375, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 8, 0},
            {19, 1, 8, 0},
            {15, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 363: skeleton etten
    {
        110, // speed
        396, // hp
        75, // ac
        33, // depth
        1, // rarity
        325, // exp
        4, // num_blows
        0x80, // flags
        { // blows
            {19, 1, 9, 0},
            {19, 1, 9, 0},
            {19, 1, 5, 0},
            {19, 1, 5, 0}
        }
    },
    // 364: chaos drake
    {
        110, // speed
        440, // hp
        84, // ac
        33, // depth
        3, // rarity
        1400, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 4, 0},
            {19, 2, 4, 0},
            {19, 3, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 365: law drake
    {
        110, // speed
        440, // hp
        84, // ac
        33, // depth
        3, // rarity
        1400, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 4, 0},
            {19, 2, 4, 0},
            {19, 3, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 366: balance drake
    {
        110, // speed
        528, // hp
        84, // ac
        33, // depth
        3, // rarity
        1600, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 4, 0},
            {19, 2, 4, 0},
            {19, 3, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 367: ethereal drake
    {
        110, // speed
        352, // hp
        84, // ac
        33, // depth
        3, // rarity
        1200, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 4, 0},
            {19, 2, 4, 0},
            {19, 3, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 368: fire elemental
    {
        110, // speed
        135, // hp
        60, // ac
        33, // depth
        2, // rarity
        350, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {17, 4, 6, 0},
            {17, 4, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 369: water elemental
    {
        110, // speed
        113, // hp
        48, // ac
        33, // depth
        2, // rarity
        325, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 10, 0},
            {19, 1, 10, 0},
            {19, 1, 10, 0},
            {0, 0, 0, 0}
        }
    },
    // 370: crystal drake
    {
        110, // speed
        396, // hp
        84, // ac
        33, // depth
        3, // rarity
        1350, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 4, 0},
            {19, 2, 4, 0},
            {19, 3, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 371: troll priest
    {
        110, // speed
        340, // hp
        75, // ac
        34, // depth
        1, // rarity
        120, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 5, 0},
            {19, 2, 5, 0},
            {19, 3, 5, 0},
            {19, 3, 5, 0}
        }
    },
    // 372: Maia
    {
        110, // speed
        528, // hp
        81, // ac
        34, // depth
        4, // rarity
        1000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 3, 5, 0},
            {19, 3, 5, 0},
            {19, 3, 5, 0},
            {19, 3, 5, 0}
        }
    },
    // 373: troll blackguard
    {
        110, // speed
        340, // hp
        60, // ac
        34, // depth
        2, // rarity
        400, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {27, 2, 5, 0},
            {27, 2, 5, 0},
            {19, 2, 8, 0},
            {19, 2, 8, 0}
        }
    },
    // 374: shade
    {
        120, // speed
        147, // hp
        36, // ac
        34, // depth
        3, // rarity
        350, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {29, 0, 0, 0},
            {15, 0, 0, 0},
            {23, 1, 10, 0},
            {0, 0, 0, 0}
        }
    },
    // 375: master thief
    {
        130, // speed
        99, // hp
        45, // ac
        34, // depth
        2, // rarity
        350, // exp
        4, // num_blows
        0x02, // flags
        { // blows
            {19, 2, 8, 0},
            {19, 3, 4, 0},
            {9, 4, 4, 0},
            {10, 4, 5, 0}
        }
    },
    // 376: Ulfang the Black
    {
        120, // speed
        1000, // hp
        108, // ac
        34, // depth
        5, // rarity
        1200, // exp
        4, // num_blows
        0x03, // flags
        { // blows
            {19, 5, 5, 0},
            {19, 5, 5, 0},
            {19, 5, 5, 0},
            {19, 5, 5, 0}
        }
    },
    // 377: lich
    {
        110, // speed
        264, // hp
        72, // ac
        34, // depth
        2, // rarity
        1000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {15, 0, 0, 0},
            {7, 0, 0, 0},
            {22, 2, 8, 0},
            {22, 2, 8, 0}
        }
    },
    // 378: earth elemental
    {
        100, // speed
        165, // hp
        90, // ac
        34, // depth
        2, // rarity
        375, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 4, 6, 0},
            {19, 4, 6, 0},
            {19, 4, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 379: air elemental
    {
        120, // speed
        90, // hp
        60, // ac
        34, // depth
        2, // rarity
        390, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 10, 0},
            {5, 1, 4, 0},
            {19, 1, 10, 0},
            {0, 0, 0, 0}
        }
    },
    // 380: Eog golem
    {
        100, // speed
        1050, // hp
        187, // ac
        34, // depth
        4, // rarity
        1200, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 8, 6, 0},
            {19, 8, 6, 0},
            {19, 6, 6, 0},
            {19, 6, 6, 0}
        }
    },
    // 381: wereworm
    {
        100, // speed
        200, // hp
        105, // ac
        35, // depth
        3, // rarity
        300, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {14, 0, 0, 0},
            {1, 2, 4, 0},
            {19, 1, 10, 0},
            {27, 1, 6, 0}
        }
    },
    // 382: Lokkak, the Ogre Chieftain
    {
        120, // speed
        1500, // hp
        120, // ac
        35, // depth
        2, // rarity
        1500, // exp
        3, // num_blows
        0x01, // flags
        { // blows
            {19, 6, 6, 0},
            {19, 6, 6, 0},
            {19, 6, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 383: hill troll
    {
        110, // speed
        316, // hp
        60, // ac
        35, // depth
        1, // rarity
        420, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 9, 0},
            {19, 2, 9, 0},
            {19, 2, 4, 0},
            {19, 2, 4, 0}
        }
    },
    // 384: invisible stalker
    {
        130, // speed
        124, // hp
        69, // ac
        35, // depth
        3, // rarity
        300, // exp
        3, // num_blows
        0x02, // flags
        { // blows
            {19, 1, 6, 0},
            {19, 1, 6, 0},
            {19, 1, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 385: gravity hound
    {
        110, // speed
        193, // hp
        36, // ac
        35, // depth
        4, // rarity
        500, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 8, 0},
            {19, 2, 8, 0},
            {19, 2, 8, 0},
            {19, 2, 8, 0}
        }
    },
    // 386: inertia hound
    {
        110, // speed
        193, // hp
        36, // ac
        35, // depth
        4, // rarity
        500, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 8, 0},
            {19, 2, 8, 0},
            {19, 2, 8, 0},
            {19, 2, 8, 0}
        }
    },
    // 387: impact hound
    {
        110, // speed
        193, // hp
        36, // ac
        35, // depth
        4, // rarity
        500, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 8, 0},
            {19, 2, 8, 0},
            {19, 2, 8, 0},
            {19, 2, 8, 0}
        }
    },
    // 388: mûmak
    {
        110, // speed
        495, // hp
        82, // ac
        35, // depth
        2, // rarity
        2100, // exp
        3, // num_blows
        0x20, // flags
        { // blows
            {19, 4, 6, 0},
            {19, 4, 6, 0},
            {19, 4, 4, 0},
            {0, 0, 0, 0}
        }
    },
    // 389: giant fire ant
    {
        110, // speed
        176, // hp
        58, // ac
        35, // depth
        1, // rarity
        350, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {17, 3, 12, 0},
            {17, 3, 12, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 390: chest mimic
    {
        110, // speed
        308, // hp
        48, // ac
        35, // depth
        4, // rarity
        250, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {27, 4, 4, 0},
            {27, 4, 4, 0},
            {5, 4, 4, 0},
            {3, 4, 4, 0}
        }
    },
    // 391: silent watcher
    {
        110, // speed
        1040, // hp
        96, // ac
        35, // depth
        3, // rarity
        800, // exp
        4, // num_blows
        0x02, // flags
        { // blows
            {29, 0, 0, 0},
            {26, 0, 0, 0},
            {24, 0, 0, 0},
            {18, 0, 0, 0}
        }
    },
    // 392: olog
    {
        115, // speed
        440, // hp
        60, // ac
        36, // depth
        1, // rarity
        450, // exp
        4, // num_blows
        0x40, // flags
        { // blows
            {19, 2, 9, 0},
            {19, 2, 9, 0},
            {19, 2, 3, 0},
            {19, 2, 3, 0}
        }
    },
    // 393: colbran
    {
        120, // speed
        520, // hp
        120, // ac
        36, // depth
        2, // rarity
        900, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {12, 3, 8, 0},
            {12, 3, 8, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 394: cloud giant
    {
        110, // speed
        368, // hp
        90, // ac
        36, // depth
        1, // rarity
        500, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {12, 8, 8, 0},
            {12, 8, 8, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 395: master vampire
    {
        110, // speed
        299, // hp
        72, // ac
        36, // depth
        1, // rarity
        750, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 6, 0},
            {19, 2, 6, 0},
            {15, 3, 6, 0},
            {15, 3, 6, 0}
        }
    },
    // 396: ooze elemental
    {
        110, // speed
        72, // hp
        120, // ac
        36, // depth
        3, // rarity
        300, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {1, 1, 10, 0},
            {1, 1, 10, 0},
            {1, 1, 10, 0},
            {0, 0, 0, 0}
        }
    },
    // 397: smoke elemental
    {
        120, // speed
        83, // hp
        120, // ac
        36, // depth
        3, // rarity
        375, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 6, 0},
            {19, 2, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 398: xorn
    {
        110, // speed
        140, // hp
        96, // ac
        36, // depth
        2, // rarity
        650, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 6, 0},
            {19, 1, 6, 0},
            {19, 1, 6, 0},
            {19, 1, 6, 0}
        }
    },
    // 399: colossus
    {
        100, // speed
        2640, // hp
        180, // ac
        36, // depth
        4, // rarity
        850, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 10, 10, 0},
            {19, 10, 10, 0},
            {19, 6, 6, 0},
            {19, 6, 6, 0}
        }
    },
    // 400: trapper
    {
        120, // speed
        528, // hp
        90, // ac
        36, // depth
        3, // rarity
        580, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 3, 8, 0},
            {19, 3, 8, 0},
            {26, 15, 1, 0},
            {26, 15, 1, 0}
        }
    },
    // 401: bodak
    {
        110, // speed
        193, // hp
        81, // ac
        36, // depth
        2, // rarity
        750, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {17, 4, 6, 0},
            {17, 4, 6, 0},
            {14, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 402: dúnadan of Angmar
    {
        120, // speed
        246, // hp
        75, // ac
        36, // depth
        2, // rarity
        630, // exp
        2, // num_blows
        0x42, // flags
        { // blows
            {19, 2, 6, 0},
            {19, 2, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 403: Lorgan, Chief of the Easterlings
    {
        120, // speed
        1800, // hp
        120, // ac
        36, // depth
        2, // rarity
        1200, // exp
        4, // num_blows
        0x03, // flags
        { // blows
            {19, 6, 6, 0},
            {19, 6, 6, 0},
            {19, 3, 8, 0},
            {19, 3, 8, 0}
        }
    },
    // 404: demonologist
    {
        120, // speed
        246, // hp
        75, // ac
        36, // depth
        2, // rarity
        700, // exp
        3, // num_blows
        0x42, // flags
        { // blows
            {19, 2, 6, 0},
            {19, 2, 6, 0},
            {19, 2, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 405: gauth
    {
        110, // speed
        264, // hp
        75, // ac
        36, // depth
        2, // rarity
        600, // exp
        4, // num_blows
        0x02, // flags
        { // blows
            {6, 5, 2, 0},
            {6, 5, 2, 0},
            {7, 5, 2, 0},
            {7, 5, 2, 0}
        }
    },
    // 406: ice elemental
    {
        110, // speed
        193, // hp
        90, // ac
        37, // depth
        3, // rarity
        650, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {4, 4, 3, 0},
            {19, 4, 6, 0},
            {4, 4, 3, 0},
            {0, 0, 0, 0}
        }
    },
    // 407: mummified troll
    {
        110, // speed
        207, // hp
        75, // ac
        37, // depth
        1, // rarity
        420, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 6, 0},
            {19, 2, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 408: spectre
    {
        120, // speed
        167, // hp
        36, // ac
        37, // depth
        3, // rarity
        380, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {29, 0, 0, 0},
            {15, 0, 0, 0},
            {25, 5, 5, 0},
            {0, 0, 0, 0}
        }
    },
    // 409: magma elemental
    {
        110, // speed
        193, // hp
        105, // ac
        37, // depth
        3, // rarity
        950, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {17, 3, 7, 0},
            {19, 4, 6, 0},
            {17, 3, 7, 0},
            {0, 0, 0, 0}
        }
    },
    // 410: killer iridescent beetle
    {
        110, // speed
        330, // hp
        90, // ac
        37, // depth
        2, // rarity
        850, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {12, 2, 12, 0},
            {12, 2, 12, 0},
            {26, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 411: nexus vortex
    {
        120, // speed
        176, // hp
        48, // ac
        37, // depth
        1, // rarity
        800, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 5, 5, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 412: plasma vortex
    {
        120, // speed
        176, // hp
        48, // ac
        37, // depth
        1, // rarity
        800, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {17, 4, 8, 0},
            {12, 4, 8, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 413: mountain troll
    {
        110, // speed
        660, // hp
        120, // ac
        37, // depth
        3, // rarity
        800, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 8, 4, 0},
            {19, 8, 4, 0},
            {19, 5, 4, 0},
            {19, 5, 4, 0}
        }
    },
    // 414: shardstorm
    {
        120, // speed
        176, // hp
        14, // ac
        37, // depth
        1, // rarity
        800, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {19, 6, 6, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 415: emperor wight
    {
        110, // speed
        334, // hp
        48, // ac
        37, // depth
        2, // rarity
        1600, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 12, 0},
            {19, 1, 12, 0},
            {16, 0, 0, 0},
            {16, 0, 0, 0}
        }
    },
    // 416: mummified chieftain
    {
        115, // speed
        299, // hp
        102, // ac
        38, // depth
        3, // rarity
        800, // exp
        4, // num_blows
        0x40, // flags
        { // blows
            {21, 3, 6, 0},
            {21, 3, 6, 0},
            {15, 3, 4, 0},
            {29, 3, 4, 0}
        }
    },
    // 417: will o' the wisp
    {
        130, // speed
        176, // hp
        180, // ac
        38, // depth
        4, // rarity
        500, // exp
        3, // num_blows
        0x40, // flags
        { // blows
            {19, 1, 9, 0},
            {19, 1, 9, 0},
            {18, 1, 9, 0},
            {0, 0, 0, 0}
        }
    },
    // 418: death knight
    {
        120, // speed
        528, // hp
        120, // ac
        38, // depth
        1, // rarity
        1000, // exp
        3, // num_blows
        0x42, // flags
        { // blows
            {19, 6, 6, 0},
            {19, 5, 5, 0},
            {19, 5, 5, 0},
            {0, 0, 0, 0}
        }
    },
    // 419: Castamir the Usurper
    {
        120, // speed
        880, // hp
        108, // ac
        38, // depth
        5, // rarity
        1600, // exp
        4, // num_blows
        0x43, // flags
        { // blows
            {19, 5, 5, 0},
            {19, 5, 5, 0},
            {19, 5, 5, 0},
            {19, 5, 5, 0}
        }
    },
    // 420: time vortex
    {
        130, // speed
        176, // hp
        48, // ac
        38, // depth
        4, // rarity
        800, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {14, 5, 5, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 421: shimmering vortex
    {
        140, // speed
        176, // hp
        48, // ac
        38, // depth
        4, // rarity
        800, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {3, 4, 4, 0},
            {3, 4, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 422: white wraith
    {
        120, // speed
        450, // hp
        48, // ac
        38, // depth
        1, // rarity
        355, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 9, 0},
            {19, 1, 9, 0},
            {14, 0, 0, 0},
            {14, 0, 0, 0}
        }
    },
    // 423: etten
    {
        110, // speed
        1320, // hp
        120, // ac
        38, // depth
        3, // rarity
        1000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 4, 6, 0},
            {19, 4, 6, 0},
            {19, 4, 6, 0},
            {19, 4, 6, 0}
        }
    },
    // 424: abyss spider
    {
        120, // speed
        264, // hp
        54, // ac
        38, // depth
        3, // rarity
        250, // exp
        4, // num_blows
        0x22, // flags
        { // blows
            {19, 2, 8, 0},
            {27, 2, 6, 0},
            {27, 2, 6, 0},
            {5, 1, 3, 0}
        }
    },
    // 425: southron archer
    {
        120, // speed
        209, // hp
        114, // ac
        39, // depth
        4, // rarity
        700, // exp
        4, // num_blows
        0x02, // flags
        { // blows
            {27, 3, 4, 0},
            {24, 3, 4, 0},
            {24, 3, 4, 0},
            {27, 3, 4, 0}
        }
    },
    // 426: mature white dragon
    {
        115, // speed
        700, // hp
        84, // ac
        39, // depth
        1, // rarity
        1400, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 3, 8, 0},
            {19, 3, 8, 0},
            {19, 4, 8, 0},
            {0, 0, 0, 0}
        }
    },
    // 427: mature blue dragon
    {
        115, // speed
        700, // hp
        84, // ac
        39, // depth
        1, // rarity
        1400, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 3, 8, 0},
            {19, 3, 8, 0},
            {19, 4, 8, 0},
            {0, 0, 0, 0}
        }
    },
    // 428: mature green dragon
    {
        115, // speed
        700, // hp
        84, // ac
        39, // depth
        1, // rarity
        1400, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 3, 8, 0},
            {19, 3, 8, 0},
            {19, 4, 8, 0},
            {0, 0, 0, 0}
        }
    },
    // 429: 6-headed hydra
    {
        120, // speed
        550, // hp
        135, // ac
        39, // depth
        2, // rarity
        2000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 6, 0},
            {19, 2, 6, 0},
            {27, 2, 6, 0},
            {17, 2, 6, 0}
        }
    },
    // 430: night mare
    {
        120, // speed
        1320, // hp
        102, // ac
        39, // depth
        3, // rarity
        2900, // exp
        4, // num_blows
        0x06, // flags
        { // blows
            {16, 2, 6, 0},
            {19, 3, 8, 0},
            {19, 3, 8, 0},
            {5, 6, 6, 0}
        }
    },
    // 431: beholder
    {
        120, // speed
        1400, // hp
        96, // ac
        40, // depth
        3, // rarity
        6000, // exp
        4, // num_blows
        0x02, // flags
        { // blows
            {14, 2, 6, 0},
            {7, 2, 6, 0},
            {23, 2, 6, 0},
            {19, 6, 6, 0}
        }
    },
    // 432: disenchanter worm mass
    {
        100, // speed
        45, // hp
        6, // ac
        40, // depth
        3, // rarity
        30, // exp
        1, // num_blows
        0x00, // flags
        { // blows
            {6, 1, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 433: spirit troll
    {
        110, // speed
        880, // hp
        108, // ac
        40, // depth
        3, // rarity
        900, // exp
        3, // num_blows
        0x80, // flags
        { // blows
            {19, 3, 6, 0},
            {19, 3, 6, 0},
            {19, 3, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 434: phantom
    {
        120, // speed
        260, // hp
        36, // ac
        40, // depth
        3, // rarity
        400, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {16, 0, 0, 0},
            {15, 0, 0, 0},
            {23, 2, 8, 0},
            {25, 2, 8, 0}
        }
    },
    // 435: 7-headed hydra
    {
        120, // speed
        650, // hp
        114, // ac
        40, // depth
        2, // rarity
        3000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 6, 0},
            {19, 2, 6, 0},
            {27, 2, 6, 0},
            {17, 2, 6, 0}
        }
    },
    // 436: enchantress
    {
        130, // speed
        457, // hp
        72, // ac
        40, // depth
        4, // rarity
        2100, // exp
        3, // num_blows
        0x02, // flags
        { // blows
            {19, 2, 6, 0},
            {19, 2, 6, 0},
            {19, 2, 8, 0},
            {0, 0, 0, 0}
        }
    },
    // 437: sorcerer
    {
        130, // speed
        457, // hp
        72, // ac
        40, // depth
        2, // rarity
        2150, // exp
        3, // num_blows
        0x02, // flags
        { // blows
            {19, 2, 8, 0},
            {19, 2, 8, 0},
            {19, 2, 8, 0},
            {0, 0, 0, 0}
        }
    },
    // 438: xaren
    {
        120, // speed
        280, // hp
        96, // ac
        40, // depth
        1, // rarity
        1200, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 3, 4, 0},
            {19, 3, 4, 0},
            {19, 3, 4, 0},
            {19, 3, 4, 0}
        }
    },
    // 439: giant roc
    {
        110, // speed
        560, // hp
        84, // ac
        40, // depth
        3, // rarity
        1000, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 8, 12, 0},
            {19, 8, 12, 0},
            {12, 12, 12, 0},
            {0, 0, 0, 0}
        }
    },
    // 440: minotaur
    {
        130, // speed
        550, // hp
        30, // ac
        40, // depth
        2, // rarity
        2100, // exp
        4, // num_blows
        0x02, // flags
        { // blows
            {19, 4, 6, 0},
            {19, 4, 6, 0},
            {19, 2, 6, 0},
            {19, 2, 6, 0}
        }
    },
    // 441: vrock
    {
        110, // speed
        352, // hp
        75, // ac
        40, // depth
        2, // rarity
        2000, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 3, 4, 0},
            {19, 8, 12, 0},
            {19, 8, 12, 0},
            {0, 0, 0, 0}
        }
    },
    // 442: death quasit
    {
        130, // speed
        387, // hp
        96, // ac
        40, // depth
        3, // rarity
        1000, // exp
        3, // num_blows
        0x40, // flags
        { // blows
            {22, 3, 6, 0},
            {19, 3, 3, 0},
            {19, 3, 3, 0},
            {0, 0, 0, 0}
        }
    },
    // 443: patriarch
    {
        120, // speed
        457, // hp
        90, // ac
        40, // depth
        2, // rarity
        1800, // exp
        3, // num_blows
        0x40, // flags
        { // blows
            {19, 3, 4, 0},
            {19, 3, 4, 0},
            {19, 3, 5, 0},
            {0, 0, 0, 0}
        }
    },
    // 444: troll chieftain
    {
        120, // speed
        792, // hp
        60, // ac
        40, // depth
        5, // rarity
        3000, // exp
        4, // num_blows
        0x40, // flags
        { // blows
            {19, 2, 12, 0},
            {19, 2, 12, 0},
            {19, 3, 6, 0},
            {19, 3, 6, 0}
        }
    },
    // 445: mature red dragon
    {
        115, // speed
        740, // hp
        96, // ac
        41, // depth
        1, // rarity
        1600, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 3, 10, 0},
            {19, 3, 10, 0},
            {19, 4, 10, 0},
            {0, 0, 0, 0}
        }
    },
    // 446: mature gold dragon
    {
        115, // speed
        740, // hp
        96, // ac
        41, // depth
        1, // rarity
        1600, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 3, 10, 0},
            {19, 3, 10, 0},
            {19, 4, 10, 0},
            {0, 0, 0, 0}
        }
    },
    // 447: mature black dragon
    {
        115, // speed
        740, // hp
        96, // ac
        41, // depth
        1, // rarity
        1600, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 3, 10, 0},
            {19, 3, 10, 0},
            {19, 4, 10, 0},
            {0, 0, 0, 0}
        }
    },
    // 448: stiffbeard sorcerer
    {
        130, // speed
        700, // hp
        84, // ac
        41, // depth
        2, // rarity
        3000, // exp
        3, // num_blows
        0x42, // flags
        { // blows
            {19, 2, 8, 0},
            {19, 2, 8, 0},
            {19, 2, 8, 0},
            {0, 0, 0, 0}
        }
    },
    // 449: master lich
    {
        120, // speed
        1584, // hp
        96, // ac
        41, // depth
        2, // rarity
        10000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {16, 0, 0, 0},
            {7, 0, 0, 0},
            {22, 2, 12, 0},
            {22, 2, 12, 0}
        }
    },
    // 450: Gorlim, Betrayer of Barahir
    {
        120, // speed
        1600, // hp
        144, // ac
        41, // depth
        3, // rarity
        7000, // exp
        4, // num_blows
        0x41, // flags
        { // blows
            {19, 8, 6, 0},
            {19, 8, 6, 0},
            {6, 6, 8, 0},
            {6, 6, 8, 0}
        }
    },
    // 451: ranger chieftain
    {
        120, // speed
        880, // hp
        72, // ac
        41, // depth
        2, // rarity
        1600, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 5, 5, 0},
            {19, 5, 5, 0},
            {19, 5, 5, 0},
            {0, 0, 0, 0}
        }
    },
    // 452: Kavlax the Many-Headed
    {
        120, // speed
        1300, // hp
        102, // ac
        41, // depth
        3, // rarity
        3000, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {19, 2, 12, 0},
            {19, 2, 12, 0},
            {19, 2, 12, 0},
            {19, 2, 12, 0}
        }
    },
    // 453: hellhound
    {
        120, // speed
        352, // hp
        120, // ac
        42, // depth
        2, // rarity
        600, // exp
        3, // num_blows
        0x02, // flags
        { // blows
            {17, 3, 12, 0},
            {17, 3, 12, 0},
            {17, 3, 12, 0},
            {0, 0, 0, 0}
        }
    },
    // 454: black-hearted huorn
    {
        120, // speed
        1200, // hp
        120, // ac
        42, // depth
        1, // rarity
        2900, // exp
        4, // num_blows
        0x02, // flags
        { // blows
            {19, 5, 6, 0},
            {19, 5, 6, 0},
            {19, 5, 6, 0},
            {19, 5, 6, 0}
        }
    },
    // 455: The Queen Ant
    {
        120, // speed
        1500, // hp
        120, // ac
        42, // depth
        2, // rarity
        2000, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {19, 2, 12, 0},
            {19, 2, 12, 0},
            {19, 2, 8, 0},
            {19, 2, 8, 0}
        }
    },
    // 456: Rogrog the Black Troll
    {
        120, // speed
        2000, // hp
        105, // ac
        42, // depth
        3, // rarity
        5000, // exp
        4, // num_blows
        0x81, // flags
        { // blows
            {19, 6, 7, 0},
            {19, 6, 7, 0},
            {19, 4, 10, 0},
            {1, 4, 8, 0}
        }
    },
    // 457: Maia of Nienna
    {
        120, // speed
        1250, // hp
        81, // ac
        43, // depth
        4, // rarity
        1900, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 3, 5, 0},
            {19, 3, 5, 0},
            {19, 3, 5, 0},
            {19, 3, 5, 0}
        }
    },
    // 458: Maia of Mandos
    {
        120, // speed
        1250, // hp
        81, // ac
        43, // depth
        4, // rarity
        1900, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 4, 5, 0},
            {19, 4, 5, 0},
            {19, 4, 5, 0},
            {19, 4, 5, 0}
        }
    },
    // 459: mature multi-hued dragon
    {
        115, // speed
        1000, // hp
        96, // ac
        43, // depth
        1, // rarity
        1950, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 3, 12, 0},
            {19, 3, 12, 0},
            {19, 4, 12, 0},
            {0, 0, 0, 0}
        }
    },
    // 460: Vargo, Tyrant of Fire
    {
        120, // speed
        2400, // hp
        75, // ac
        43, // depth
        3, // rarity
        4000, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {17, 4, 6, 0},
            {17, 4, 6, 0},
            {17, 4, 6, 0},
            {17, 4, 6, 0}
        }
    },
    // 461: Waldern, King of Water
    {
        120, // speed
        2500, // hp
        120, // ac
        43, // depth
        3, // rarity
        4250, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {19, 5, 5, 0},
            {19, 5, 5, 0},
            {19, 5, 5, 0},
            {19, 5, 5, 0}
        }
    },
    // 462: glabrezu
    {
        110, // speed
        616, // hp
        60, // ac
        43, // depth
        2, // rarity
        3000, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 3, 4, 0},
            {19, 3, 4, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 463: Quaker, Master of Earth
    {
        110, // speed
        2800, // hp
        145, // ac
        43, // depth
        3, // rarity
        4500, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {19, 6, 6, 0},
            {19, 6, 6, 0},
            {19, 6, 6, 0},
            {28, 10, 10, 0}
        }
    },
    // 464: Ariel, Queen of Air
    {
        130, // speed
        2700, // hp
        75, // ac
        43, // depth
        3, // rarity
        4750, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {19, 4, 6, 0},
            {19, 4, 6, 0},
            {5, 4, 4, 0},
            {5, 4, 4, 0}
        }
    },
    // 465: multi-hued hound
    {
        110, // speed
        220, // hp
        48, // ac
        43, // depth
        6, // rarity
        600, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 8, 0},
            {19, 2, 8, 0},
            {19, 4, 4, 0},
            {19, 4, 4, 0}
        }
    },
    // 466: 8-headed hydra
    {
        120, // speed
        950, // hp
        120, // ac
        44, // depth
        2, // rarity
        6000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 6, 0},
            {19, 2, 6, 0},
            {27, 2, 6, 0},
            {17, 2, 6, 0}
        }
    },
    // 467: dread
    {
        120, // speed
        263, // hp
        36, // ac
        44, // depth
        2, // rarity
        600, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 6, 6, 0},
            {19, 6, 6, 0},
            {24, 3, 4, 0},
            {0, 0, 0, 0}
        }
    },
    // 468: killer white beetle
    {
        120, // speed
        800, // hp
        106, // ac
        45, // depth
        2, // rarity
        850, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {4, 8, 5, 0},
            {19, 8, 5, 0},
            {4, 8, 5, 0},
            {0, 0, 0, 0}
        }
    },
    // 469: rotting quylthulg
    {
        120, // speed
        420, // hp
        1, // ac
        45, // depth
        1, // rarity
        3000, // exp
        0, // num_blows
        0x00, // flags
        { // blows
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 470: nalfeshnee
    {
        110, // speed
        792, // hp
        60, // ac
        45, // depth
        2, // rarity
        5000, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 3, 4, 0},
            {19, 3, 4, 0},
            {19, 3, 4, 0},
            {0, 0, 0, 0}
        }
    },
    // 471: undead beholder
    {
        120, // speed
        2376, // hp
        120, // ac
        45, // depth
        3, // rarity
        8000, // exp
        4, // num_blows
        0x06, // flags
        { // blows
            {15, 3, 6, 0},
            {7, 3, 6, 0},
            {23, 3, 6, 0},
            {15, 7, 6, 0}
        }
    },
    // 472: demonic quylthulg
    {
        120, // speed
        420, // hp
        1, // ac
        45, // depth
        1, // rarity
        3000, // exp
        0, // num_blows
        0x00, // flags
        { // blows
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 473: draconic quylthulg
    {
        120, // speed
        420, // hp
        1, // ac
        45, // depth
        1, // rarity
        3000, // exp
        0, // num_blows
        0x00, // flags
        { // blows
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 474: greater basilisk
    {
        120, // speed
        1010, // hp
        120, // ac
        45, // depth
        2, // rarity
        10000, // exp
        4, // num_blows
        0x02, // flags
        { // blows
            {26, 3, 12, 0},
            {26, 3, 12, 0},
            {27, 2, 12, 0},
            {27, 2, 12, 0}
        }
    },
    // 475: berserker
    {
        120, // speed
        1320, // hp
        96, // ac
        45, // depth
        2, // rarity
        2500, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 6, 6, 0},
            {19, 6, 6, 0},
            {19, 6, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 476: cyclops
    {
        120, // speed
        1050, // hp
        144, // ac
        45, // depth
        2, // rarity
        1500, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 9, 9, 0},
            {19, 9, 9, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 477: grey wraith
    {
        120, // speed
        650, // hp
        60, // ac
        45, // depth
        1, // rarity
        1700, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 10, 0},
            {19, 1, 10, 0},
            {15, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 478: lord of Carn Dûm
    {
        120, // speed
        557, // hp
        90, // ac
        46, // depth
        2, // rarity
        2100, // exp
        3, // num_blows
        0x42, // flags
        { // blows
            {19, 5, 4, 0},
            {19, 5, 4, 0},
            {19, 3, 5, 0},
            {0, 0, 0, 0}
        }
    },
    // 479: vampire lord
    {
        120, // speed
        1400, // hp
        84, // ac
        46, // depth
        1, // rarity
        1800, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 3, 6, 0},
            {19, 3, 6, 0},
            {16, 4, 6, 0},
            {16, 4, 6, 0}
        }
    },
    // 480: marilith
    {
        120, // speed
        1232, // hp
        112, // ac
        47, // depth
        2, // rarity
        7000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 4, 6, 0},
            {19, 4, 6, 0},
            {19, 4, 6, 0},
            {19, 4, 6, 0}
        }
    },
    // 481: death mold
    {
        140, // speed
        1050, // hp
        72, // ac
        47, // depth
        1, // rarity
        1000, // exp
        4, // num_blows
        0x02, // flags
        { // blows
            {6, 7, 7, 0},
            {6, 7, 7, 0},
            {6, 7, 7, 0},
            {16, 5, 5, 0}
        }
    },
    // 482: ancient spider
    {
        120, // speed
        1050, // hp
        78, // ac
        48, // depth
        5, // rarity
        2500, // exp
        4, // num_blows
        0x62, // flags
        { // blows
            {27, 5, 8, 0},
            {27, 5, 8, 0},
            {27, 5, 6, 0},
            {27, 5, 6, 0}
        }
    },
    // 483: winged horror
    {
        120, // speed
        1013, // hp
        96, // ac
        48, // depth
        3, // rarity
        4000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 3, 8, 0},
            {19, 3, 8, 0},
            {15, 4, 6, 0},
            {15, 4, 6, 0}
        }
    },
    // 484: gorgon
    {
        120, // speed
        1500, // hp
        120, // ac
        48, // depth
        2, // rarity
        6000, // exp
        4, // num_blows
        0x40, // flags
        { // blows
            {16, 0, 0, 0},
            {26, 0, 0, 0},
            {19, 8, 6, 0},
            {27, 8, 6, 0}
        }
    },
    // 485: storm giant
    {
        120, // speed
        1200, // hp
        110, // ac
        49, // depth
        1, // rarity
        1400, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {12, 10, 8, 0},
            {12, 10, 8, 0},
            {19, 10, 8, 0},
            {0, 0, 0, 0}
        }
    },
    // 486: lesser Balrog
    {
        120, // speed
        1760, // hp
        75, // ac
        49, // depth
        2, // rarity
        10000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {17, 4, 12, 0},
            {17, 4, 12, 0},
            {19, 3, 12, 0},
            {7, 0, 0, 0}
        }
    },
    // 487: black wraith
    {
        120, // speed
        840, // hp
        66, // ac
        49, // depth
        2, // rarity
        2700, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 12, 0},
            {19, 1, 12, 0},
            {15, 0, 0, 0},
            {15, 0, 0, 0}
        }
    },
    // 488: Eöl, the Dark Elf
    {
        130, // speed
        2400, // hp
        150, // ac
        49, // depth
        2, // rarity
        25000, // exp
        3, // num_blows
        0x43, // flags
        { // blows
            {19, 3, 8, 0},
            {19, 3, 8, 0},
            {19, 3, 8, 0},
            {0, 0, 0, 0}
        }
    },
    // 489: ancient blue dragon
    {
        120, // speed
        1500, // hp
        108, // ac
        50, // depth
        1, // rarity
        2500, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 6, 8, 0},
            {19, 6, 8, 0},
            {12, 7, 8, 0},
            {0, 0, 0, 0}
        }
    },
    // 490: ancient white dragon
    {
        120, // speed
        1500, // hp
        108, // ac
        50, // depth
        1, // rarity
        2500, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 6, 8, 0},
            {19, 6, 8, 0},
            {4, 7, 8, 0},
            {0, 0, 0, 0}
        }
    },
    // 491: ancient green dragon
    {
        120, // speed
        1500, // hp
        108, // ac
        50, // depth
        1, // rarity
        2500, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 6, 8, 0},
            {19, 6, 8, 0},
            {27, 7, 8, 0},
            {0, 0, 0, 0}
        }
    },
    // 492: death drake
    {
        120, // speed
        1950, // hp
        120, // ac
        50, // depth
        2, // rarity
        11000, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 6, 10, 0},
            {19, 6, 10, 0},
            {16, 7, 10, 0},
            {0, 0, 0, 0}
        }
    },
    // 493: great crystal drake
    {
        120, // speed
        1950, // hp
        120, // ac
        50, // depth
        2, // rarity
        11000, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 6, 10, 0},
            {19, 6, 10, 0},
            {19, 7, 10, 0},
            {0, 0, 0, 0}
        }
    },
    // 494: ethereal dragon
    {
        120, // speed
        1950, // hp
        120, // ac
        50, // depth
        2, // rarity
        11000, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 6, 10, 0},
            {19, 6, 10, 0},
            {5, 7, 10, 0},
            {0, 0, 0, 0}
        }
    },
    // 495: master mystic
    {
        130, // speed
        968, // hp
        72, // ac
        50, // depth
        3, // rarity
        6000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 10, 2, 0},
            {19, 10, 2, 0},
            {26, 20, 1, 0},
            {26, 15, 1, 0}
        }
    },
    // 496: shadow
    {
        120, // speed
        175, // hp
        36, // ac
        50, // depth
        3, // rarity
        800, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {16, 0, 0, 0},
            {15, 0, 0, 0},
            {23, 3, 10, 0},
            {25, 3, 10, 0}
        }
    },
    // 497: The Balrog of Moria
    {
        130, // speed
        3000, // hp
        150, // ac
        50, // depth
        3, // rarity
        30000, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {17, 6, 12, 0},
            {17, 6, 12, 0},
            {19, 5, 12, 0},
            {7, 0, 0, 0}
        }
    },
    // 498: ancient black dragon
    {
        120, // speed
        1560, // hp
        108, // ac
        51, // depth
        1, // rarity
        2800, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 6, 9, 0},
            {19, 6, 9, 0},
            {1, 7, 9, 0},
            {0, 0, 0, 0}
        }
    },
    // 499: ancient red dragon
    {
        120, // speed
        1560, // hp
        108, // ac
        51, // depth
        1, // rarity
        2800, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 6, 9, 0},
            {19, 6, 9, 0},
            {17, 7, 9, 0},
            {0, 0, 0, 0}
        }
    },
    // 500: ancient gold dragon
    {
        120, // speed
        1560, // hp
        108, // ac
        51, // depth
        1, // rarity
        2800, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 6, 9, 0},
            {19, 6, 9, 0},
            {19, 7, 9, 0},
            {0, 0, 0, 0}
        }
    },
    // 501: nether hound
    {
        120, // speed
        330, // hp
        120, // ac
        51, // depth
        4, // rarity
        5000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 12, 0},
            {19, 2, 12, 0},
            {19, 2, 12, 0},
            {19, 2, 12, 0}
        }
    },
    // 502: time hound
    {
        130, // speed
        330, // hp
        120, // ac
        51, // depth
        4, // rarity
        5000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 12, 0},
            {19, 2, 12, 0},
            {19, 2, 12, 0},
            {19, 2, 12, 0}
        }
    },
    // 503: plasma hound
    {
        120, // speed
        330, // hp
        120, // ac
        51, // depth
        4, // rarity
        5000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 12, 0},
            {19, 2, 12, 0},
            {19, 2, 12, 0},
            {19, 2, 12, 0}
        }
    },
    // 504: Harowen the Black Hand
    {
        140, // speed
        2500, // hp
        108, // ac
        51, // depth
        3, // rarity
        20000, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {9, 5, 5, 0},
            {10, 5, 5, 0},
            {3, 10, 5, 0},
            {27, 8, 5, 0}
        }
    },
    // 505: Maia of Oromë
    {
        120, // speed
        2000, // hp
        96, // ac
        52, // depth
        4, // rarity
        2500, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 4, 6, 0},
            {19, 6, 5, 0},
            {19, 4, 6, 0},
            {19, 6, 5, 0}
        }
    },
    // 506: hasty ent
    {
        120, // speed
        3000, // hp
        120, // ac
        52, // depth
        3, // rarity
        3500, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 10, 5, 0},
            {19, 10, 5, 0},
            {19, 3, 10, 0},
            {19, 3, 10, 0}
        }
    },
    // 507: Itangast the Fire Drake
    {
        120, // speed
        2200, // hp
        144, // ac
        52, // depth
        2, // rarity
        20000, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {19, 4, 11, 0},
            {19, 4, 11, 0},
            {17, 4, 15, 0},
            {17, 4, 15, 0}
        }
    },
    // 508: hezrou
    {
        100, // speed
        1850, // hp
        60, // ac
        53, // depth
        3, // rarity
        7000, // exp
        3, // num_blows
        0x40, // flags
        { // blows
            {19, 3, 4, 0},
            {1, 3, 10, 0},
            {3, 3, 10, 0},
            {0, 0, 0, 0}
        }
    },
    // 509: ancient multi-hued dragon
    {
        120, // speed
        2400, // hp
        120, // ac
        53, // depth
        1, // rarity
        14000, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 6, 10, 0},
            {19, 6, 10, 0},
            {19, 7, 10, 0},
            {0, 0, 0, 0}
        }
    },
    // 510: chaos vortex
    {
        120, // speed
        315, // hp
        96, // ac
        53, // depth
        3, // rarity
        4000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {5, 5, 5, 0},
            {5, 5, 5, 0},
            {5, 5, 5, 0},
            {18, 5, 5, 0}
        }
    },
    // 511: Scatha the Worm
    {
        120, // speed
        2500, // hp
        156, // ac
        53, // depth
        2, // rarity
        18000, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {19, 4, 11, 0},
            {19, 4, 11, 0},
            {4, 4, 15, 0},
            {4, 4, 15, 0}
        }
    },
    // 512: The Phoenix
    {
        120, // speed
        3600, // hp
        156, // ac
        54, // depth
        3, // rarity
        40000, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {17, 12, 6, 0},
            {17, 12, 6, 0},
            {17, 9, 12, 0},
            {17, 9, 12, 0}
        }
    },
    // 513: drolem
    {
        120, // speed
        2200, // hp
        195, // ac
        54, // depth
        3, // rarity
        12000, // exp
        4, // num_blows
        0x08, // flags
        { // blows
            {19, 3, 10, 0},
            {19, 3, 10, 0},
            {27, 5, 10, 0},
            {27, 5, 10, 0}
        }
    },
    // 514: demilich
    {
        120, // speed
        2816, // hp
        150, // ac
        54, // depth
        2, // rarity
        12500, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {16, 0, 0, 0},
            {7, 0, 0, 0},
            {22, 4, 12, 0},
            {22, 4, 12, 0}
        }
    },
    // 515: dreadmaster
    {
        120, // speed
        1056, // hp
        120, // ac
        54, // depth
        2, // rarity
        8000, // exp
        4, // num_blows
        0x40, // flags
        { // blows
            {19, 6, 6, 0},
            {19, 6, 6, 0},
            {24, 3, 4, 0},
            {24, 3, 4, 0}
        }
    },
    // 516: nether wraith
    {
        120, // speed
        1000, // hp
        66, // ac
        54, // depth
        2, // rarity
        3900, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 1, 12, 0},
            {19, 1, 12, 0},
            {16, 0, 0, 0},
            {16, 0, 0, 0}
        }
    },
    // 517: Shelob, Spider of Darkness
    {
        120, // speed
        3500, // hp
        180, // ac
        55, // depth
        3, // rarity
        27000, // exp
        4, // num_blows
        0x63, // flags
        { // blows
            {27, 5, 6, 0},
            {27, 5, 6, 0},
            {26, 5, 10, 0},
            {24, 5, 4, 0}
        }
    },
    // 518: dracolich
    {
        120, // speed
        3080, // hp
        144, // ac
        55, // depth
        2, // rarity
        18000, // exp
        3, // num_blows
        0x04, // flags
        { // blows
            {19, 6, 12, 0},
            {19, 6, 12, 0},
            {16, 7, 14, 0},
            {0, 0, 0, 0}
        }
    },
    // 519: dracolisk
    {
        120, // speed
        3080, // hp
        144, // ac
        55, // depth
        2, // rarity
        18000, // exp
        4, // num_blows
        0x20, // flags
        { // blows
            {19, 6, 12, 0},
            {19, 6, 12, 0},
            {17, 7, 14, 0},
            {26, 0, 0, 0}
        }
    },
    // 520: Ar-Pharazôn the Golden
    {
        130, // speed
        4000, // hp
        54, // ac
        55, // depth
        1, // rarity
        32500, // exp
        4, // num_blows
        0x43, // flags
        { // blows
            {19, 8, 8, 0},
            {19, 8, 8, 0},
            {19, 8, 8, 0},
            {19, 8, 8, 0}
        }
    },
    // 521: barbazu
    {
        120, // speed
        700, // hp
        90, // ac
        55, // depth
        4, // rarity
        3000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 4, 10, 0},
            {19, 4, 10, 0},
            {21, 10, 2, 0},
            {27, 5, 5, 0}
        }
    },
    // 522: lesser titan
    {
        120, // speed
        2100, // hp
        96, // ac
        56, // depth
        3, // rarity
        6000, // exp
        4, // num_blows
        0x40, // flags
        { // blows
            {5, 9, 9, 0},
            {5, 9, 9, 0},
            {5, 9, 9, 0},
            {5, 9, 9, 0}
        }
    },
    // 523: grand master mystic
    {
        130, // speed
        1936, // hp
        96, // ac
        57, // depth
        3, // rarity
        15000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 20, 2, 0},
            {19, 10, 2, 0},
            {26, 20, 1, 0},
            {26, 15, 1, 0}
        }
    },
    // 524: hand druj
    {
        130, // speed
        528, // hp
        132, // ac
        57, // depth
        2, // rarity
        12000, // exp
        0, // num_blows
        0x40, // flags
        { // blows
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 525: 9-headed hydra
    {
        120, // speed
        1200, // hp
        168, // ac
        57, // depth
        2, // rarity
        7000, // exp
        4, // num_blows
        0xC0, // flags
        { // blows
            {19, 2, 6, 0},
            {19, 2, 6, 0},
            {27, 2, 6, 0},
            {17, 2, 6, 0}
        }
    },
    // 526: Baphomet the Minotaur Lord
    {
        130, // speed
        4700, // hp
        144, // ac
        58, // depth
        4, // rarity
        24000, // exp
        4, // num_blows
        0x03, // flags
        { // blows
            {19, 12, 13, 0},
            {19, 12, 13, 0},
            {19, 10, 10, 0},
            {19, 10, 10, 0}
        }
    },
    // 527: elder vampire
    {
        120, // speed
        2640, // hp
        108, // ac
        59, // depth
        3, // rarity
        4500, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 4, 6, 0},
            {19, 4, 6, 0},
            {16, 5, 6, 0},
            {16, 5, 6, 0}
        }
    },
    // 528: Fundin Bluecloak
    {
        130, // speed
        5000, // hp
        234, // ac
        59, // depth
        2, // rarity
        20000, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {19, 8, 8, 0},
            {19, 8, 8, 0},
            {19, 8, 8, 0},
            {19, 8, 8, 0}
        }
    },
    // 529: Gilim, the Giant of Eruman
    {
        120, // speed
        4000, // hp
        96, // ac
        60, // depth
        3, // rarity
        6000, // exp
        4, // num_blows
        0x41, // flags
        { // blows
            {4, 9, 10, 0},
            {4, 9, 10, 0},
            {19, 10, 9, 0},
            {5, 10, 9, 0}
        }
    },
    // 530: Maia of Yavanna
    {
        120, // speed
        3500, // hp
        90, // ac
        60, // depth
        4, // rarity
        3600, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 4, 6, 0},
            {19, 6, 5, 0},
            {19, 4, 6, 0},
            {19, 6, 5, 0}
        }
    },
    // 531: Maia of Aulë
    {
        120, // speed
        3500, // hp
        120, // ac
        60, // depth
        4, // rarity
        3600, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 6, 6, 0},
            {19, 6, 6, 0},
            {19, 6, 6, 0},
            {28, 10, 10, 0}
        }
    },
    // 532: great earth elemental
    {
        120, // speed
        1, // hp
        145, // ac
        60, // depth
        0, // rarity
        0, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 6, 6, 0},
            {19, 6, 6, 0},
            {19, 6, 6, 0},
            {28, 10, 10, 0}
        }
    },
    // 533: aether vortex
    {
        130, // speed
        420, // hp
        48, // ac
        60, // depth
        4, // rarity
        5000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {12, 5, 5, 0},
            {17, 5, 5, 0},
            {1, 5, 5, 0},
            {4, 5, 5, 0}
        }
    },
    // 534: Saruman of Many Colours
    {
        120, // speed
        5000, // hp
        120, // ac
        60, // depth
        1, // rarity
        35000, // exp
        4, // num_blows
        0x03, // flags
        { // blows
            {6, 6, 8, 0},
            {6, 6, 8, 0},
            {19, 5, 5, 0},
            {19, 5, 5, 0}
        }
    },
    // 535: nightwing
    {
        120, // speed
        1300, // hp
        144, // ac
        61, // depth
        3, // rarity
        10000, // exp
        4, // num_blows
        0x40, // flags
        { // blows
            {27, 6, 5, 0},
            {27, 6, 5, 0},
            {6, 6, 8, 0},
            {6, 6, 8, 0}
        }
    },
    // 536: bile demon
    {
        120, // speed
        2464, // hp
        135, // ac
        61, // depth
        2, // rarity
        12000, // exp
        3, // num_blows
        0x00, // flags
        { // blows
            {19, 8, 8, 0},
            {19, 8, 8, 0},
            {1, 9, 9, 0},
            {0, 0, 0, 0}
        }
    },
    // 537: dreadlord
    {
        120, // speed
        2640, // hp
        180, // ac
        62, // depth
        2, // rarity
        20000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {15, 6, 6, 0},
            {15, 6, 6, 0},
            {24, 4, 6, 0},
            {24, 4, 6, 0}
        }
    },
    // 538: Smaug the Golden
    {
        120, // speed
        4200, // hp
        195, // ac
        62, // depth
        2, // rarity
        30000, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {19, 6, 12, 0},
            {19, 6, 12, 0},
            {17, 7, 14, 0},
            {17, 7, 14, 0}
        }
    },
    // 539: werewolf of Sauron
    {
        120, // speed
        2200, // hp
        88, // ac
        63, // depth
        2, // rarity
        4000, // exp
        3, // num_blows
        0x42, // flags
        { // blows
            {14, 2, 4, 0},
            {27, 4, 10, 0},
            {27, 4, 10, 0},
            {0, 0, 0, 0}
        }
    },
    // 540: spider of Gorgoroth
    {
        120, // speed
        4000, // hp
        180, // ac
        63, // depth
        3, // rarity
        30000, // exp
        4, // num_blows
        0x62, // flags
        { // blows
            {27, 6, 6, 0},
            {27, 6, 6, 0},
            {26, 5, 10, 0},
            {24, 5, 4, 0}
        }
    },
    // 541: chaos hound
    {
        120, // speed
        930, // hp
        120, // ac
        64, // depth
        4, // rarity
        10000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 12, 0},
            {19, 2, 12, 0},
            {19, 2, 12, 0},
            {19, 2, 12, 0}
        }
    },
    // 542: archlich
    {
        120, // speed
        3520, // hp
        180, // ac
        64, // depth
        2, // rarity
        20000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {16, 0, 0, 0},
            {7, 0, 0, 0},
            {22, 8, 12, 0},
            {22, 8, 12, 0}
        }
    },
    // 543: Maia of Ulmo
    {
        120, // speed
        4700, // hp
        90, // ac
        65, // depth
        4, // rarity
        5000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {29, 6, 5, 0},
            {26, 6, 5, 0},
            {1, 6, 10, 0},
            {1, 6, 10, 0}
        }
    },
    // 544: great water elemental
    {
        120, // speed
        1, // hp
        120, // ac
        65, // depth
        0, // rarity
        4250, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {29, 6, 5, 0},
            {26, 6, 5, 0},
            {1, 6, 10, 0},
            {1, 6, 10, 0}
        }
    },
    // 545: The Mouth of Sauron
    {
        130, // speed
        7000, // hp
        120, // ac
        65, // depth
        3, // rarity
        38000, // exp
        4, // num_blows
        0x43, // flags
        { // blows
            {6, 6, 8, 0},
            {6, 6, 8, 0},
            {7, 5, 5, 0},
            {7, 5, 5, 0}
        }
    },
    // 546: osyluth
    {
        130, // speed
        2288, // hp
        112, // ac
        65, // depth
        2, // rarity
        13000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {25, 6, 6, 0},
            {21, 6, 6, 0},
            {27, 8, 8, 0},
            {24, 5, 5, 0}
        }
    },
    // 547: Fëanorian raider
    {
        120, // speed
        1200, // hp
        110, // ac
        65, // depth
        2, // rarity
        11000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 7, 12, 0},
            {19, 7, 12, 0},
            {19, 7, 12, 0},
            {19, 7, 12, 0}
        }
    },
    // 548: eye druj
    {
        130, // speed
        1080, // hp
        108, // ac
        66, // depth
        2, // rarity
        18000, // exp
        0, // num_blows
        0x40, // flags
        { // blows
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 549: greater titan
    {
        120, // speed
        3344, // hp
        150, // ac
        66, // depth
        3, // rarity
        23500, // exp
        4, // num_blows
        0x40, // flags
        { // blows
            {5, 12, 12, 0},
            {5, 12, 12, 0},
            {5, 12, 12, 0},
            {5, 12, 12, 0}
        }
    },
    // 550: Tevildo, Prince of Cats
    {
        130, // speed
        4800, // hp
        240, // ac
        66, // depth
        3, // rarity
        30000, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {5, 12, 12, 0},
            {22, 2, 12, 0},
            {3, 10, 5, 0},
            {26, 15, 1, 0}
        }
    },
    // 551: Thuringwethil, the Vampire Messenger
    {
        130, // speed
        5000, // hp
        174, // ac
        67, // depth
        4, // rarity
        23000, // exp
        4, // num_blows
        0x41, // flags
        { // blows
            {16, 6, 6, 0},
            {16, 6, 6, 0},
            {5, 6, 6, 0},
            {5, 6, 6, 0}
        }
    },
    // 552: beholder hive-mother
    {
        120, // speed
        3080, // hp
        96, // ac
        67, // depth
        3, // rarity
        17000, // exp
        4, // num_blows
        0x42, // flags
        { // blows
            {16, 6, 6, 0},
            {26, 5, 5, 0},
            {23, 5, 5, 0},
            {7, 5, 5, 0}
        }
    },
    // 553: jabberwock
    {
        130, // speed
        2816, // hp
        187, // ac
        68, // depth
        3, // rarity
        19000, // exp
        4, // num_blows
        0x20, // flags
        { // blows
            {19, 10, 10, 0},
            {19, 10, 10, 0},
            {19, 10, 10, 0},
            {19, 10, 10, 0}
        }
    },
    // 554: Tselakus, the Dreadlord
    {
        130, // speed
        6500, // hp
        180, // ac
        68, // depth
        2, // rarity
        35000, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {19, 10, 10, 0},
            {19, 10, 10, 0},
            {24, 4, 6, 0},
            {24, 4, 6, 0}
        }
    },
    // 555: bone golem
    {
        120, // speed
        3080, // hp
        255, // ac
        68, // depth
        3, // rarity
        23000, // exp
        4, // num_blows
        0x04, // flags
        { // blows
            {6, 8, 8, 0},
            {6, 8, 8, 0},
            {24, 6, 6, 0},
            {24, 6, 6, 0}
        }
    },
    // 556: hound of Tindalos
    {
        130, // speed
        480, // hp
        120, // ac
        69, // depth
        6, // rarity
        7000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 2, 12, 0},
            {19, 2, 12, 0},
            {19, 2, 12, 0},
            {19, 2, 12, 0}
        }
    },
    // 557: Nan, the Giant
    {
        120, // speed
        3344, // hp
        150, // ac
        69, // depth
        3, // rarity
        23500, // exp
        4, // num_blows
        0x41, // flags
        { // blows
            {19, 12, 12, 0},
            {19, 12, 12, 0},
            {5, 12, 12, 0},
            {5, 12, 12, 0}
        }
    },
    // 558: nightcrawler
    {
        120, // speed
        2440, // hp
        192, // ac
        69, // depth
        3, // rarity
        15000, // exp
        4, // num_blows
        0x46, // flags
        { // blows
            {21, 8, 8, 0},
            {21, 8, 8, 0},
            {1, 10, 10, 0},
            {1, 10, 10, 0}
        }
    },
    // 559: gelugon
    {
        130, // speed
        3080, // hp
        150, // ac
        69, // depth
        3, // rarity
        14000, // exp
        4, // num_blows
        0x80, // flags
        { // blows
            {4, 6, 8, 0},
            {4, 6, 8, 0},
            {4, 9, 9, 0},
            {26, 5, 5, 0}
        }
    },
    // 560: great storm wyrm
    {
        120, // speed
        4200, // hp
        225, // ac
        70, // depth
        2, // rarity
        25000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 6, 12, 0},
            {19, 6, 12, 0},
            {12, 6, 14, 0},
            {12, 6, 14, 0}
        }
    },
    // 561: great ice wyrm
    {
        120, // speed
        4200, // hp
        225, // ac
        70, // depth
        2, // rarity
        25000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 6, 12, 0},
            {19, 6, 12, 0},
            {4, 6, 14, 0},
            {4, 6, 14, 0}
        }
    },
    // 562: great swamp wyrm
    {
        120, // speed
        4200, // hp
        225, // ac
        70, // depth
        2, // rarity
        20000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 6, 12, 0},
            {19, 6, 12, 0},
            {27, 6, 14, 0},
            {27, 6, 14, 0}
        }
    },
    // 563: Maia of Manwë
    {
        130, // speed
        5000, // hp
        120, // ac
        71, // depth
        4, // rarity
        7000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {12, 10, 12, 0},
            {12, 10, 12, 0},
            {5, 8, 10, 0},
            {5, 8, 10, 0}
        }
    },
    // 564: Maia of Varda
    {
        130, // speed
        5000, // hp
        110, // ac
        71, // depth
        4, // rarity
        12000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {29, 6, 5, 0},
            {26, 6, 5, 0},
            {1, 6, 10, 0},
            {1, 6, 10, 0}
        }
    },
    // 565: greater demonic quylthulg
    {
        120, // speed
        1320, // hp
        1, // ac
        71, // depth
        2, // rarity
        10500, // exp
        0, // num_blows
        0x00, // flags
        { // blows
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 566: greater draconic quylthulg
    {
        120, // speed
        1320, // hp
        1, // ac
        71, // depth
        2, // rarity
        10500, // exp
        0, // num_blows
        0x00, // flags
        { // blows
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 567: greater rotting quylthulg
    {
        120, // speed
        1320, // hp
        1, // ac
        71, // depth
        2, // rarity
        10500, // exp
        0, // num_blows
        0x00, // flags
        { // blows
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 568: horned reaper
    {
        130, // speed
        3344, // hp
        180, // ac
        72, // depth
        3, // rarity
        18000, // exp
        4, // num_blows
        0x80, // flags
        { // blows
            {19, 11, 11, 0},
            {19, 11, 11, 0},
            {19, 11, 11, 0},
            {19, 11, 11, 0}
        }
    },
    // 569: Ossë, Herald of Ulmo
    {
        130, // speed
        6000, // hp
        204, // ac
        72, // depth
        3, // rarity
        30000, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {6, 10, 5, 0},
            {6, 10, 5, 0},
            {19, 10, 10, 0},
            {19, 10, 10, 0}
        }
    },
    // 570: great hell wyrm
    {
        120, // speed
        4500, // hp
        225, // ac
        73, // depth
        2, // rarity
        25000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 6, 12, 0},
            {19, 6, 12, 0},
            {17, 6, 14, 0},
            {17, 6, 14, 0}
        }
    },
    // 571: great bile wyrm
    {
        120, // speed
        4500, // hp
        225, // ac
        73, // depth
        2, // rarity
        23000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 6, 12, 0},
            {19, 6, 12, 0},
            {1, 6, 14, 0},
            {1, 6, 14, 0}
        }
    },
    // 572: great wyrm of thunder
    {
        120, // speed
        4500, // hp
        225, // ac
        73, // depth
        2, // rarity
        23000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 6, 12, 0},
            {19, 6, 12, 0},
            {19, 6, 14, 0},
            {19, 6, 14, 0}
        }
    },
    // 573: nightwalker
    {
        130, // speed
        1400, // hp
        170, // ac
        73, // depth
        3, // rarity
        20000, // exp
        4, // num_blows
        0x40, // flags
        { // blows
            {6, 10, 10, 0},
            {6, 10, 10, 0},
            {6, 8, 8, 0},
            {6, 8, 8, 0}
        }
    },
    // 574: Omarax, the Eye Tyrant
    {
        130, // speed
        6500, // hp
        96, // ac
        73, // depth
        3, // rarity
        25000, // exp
        4, // num_blows
        0x43, // flags
        { // blows
            {6, 6, 6, 0},
            {7, 6, 6, 0},
            {23, 6, 6, 0},
            {16, 8, 8, 0}
        }
    },
    // 575: skull druj
    {
        130, // speed
        1540, // hp
        144, // ac
        74, // depth
        2, // rarity
        23000, // exp
        0, // num_blows
        0x40, // flags
        { // blows
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 576: black reaver
    {
        120, // speed
        3960, // hp
        255, // ac
        74, // depth
        3, // rarity
        23000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {6, 6, 8, 0},
            {6, 6, 8, 0},
            {24, 4, 6, 0},
            {24, 4, 6, 0}
        }
    },
    // 577: aether hound
    {
        130, // speed
        1230, // hp
        120, // ac
        74, // depth
        5, // rarity
        10000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 3, 12, 0},
            {19, 3, 12, 0},
            {19, 3, 12, 0},
            {19, 3, 12, 0}
        }
    },
    // 578: great wyrm of chaos
    {
        120, // speed
        3960, // hp
        255, // ac
        75, // depth
        2, // rarity
        29000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 5, 12, 0},
            {19, 5, 12, 0},
            {19, 7, 14, 0},
            {19, 7, 14, 0}
        }
    },
    // 579: great wyrm of law
    {
        120, // speed
        3960, // hp
        255, // ac
        75, // depth
        2, // rarity
        29000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 5, 12, 0},
            {19, 5, 12, 0},
            {19, 7, 14, 0},
            {19, 7, 14, 0}
        }
    },
    // 580: Ungoliant, the Unlight
    {
        130, // speed
        13000, // hp
        240, // ac
        75, // depth
        1, // rarity
        35000, // exp
        4, // num_blows
        0x63, // flags
        { // blows
            {27, 8, 6, 0},
            {27, 8, 6, 0},
            {26, 8, 10, 0},
            {24, 8, 4, 0}
        }
    },
    // 581: bronze golem
    {
        120, // speed
        3520, // hp
        255, // ac
        75, // depth
        3, // rarity
        26000, // exp
        4, // num_blows
        0x10, // flags
        { // blows
            {19, 10, 10, 0},
            {19, 10, 10, 0},
            {19, 10, 10, 0},
            {19, 10, 10, 0}
        }
    },
    // 582: Glaurung, Father of the Dragons
    {
        130, // speed
        7500, // hp
        210, // ac
        75, // depth
        3, // rarity
        55000, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {19, 6, 12, 0},
            {19, 6, 12, 0},
            {27, 8, 14, 0},
            {17, 8, 14, 0}
        }
    },
    // 583: master quylthulg
    {
        120, // speed
        2640, // hp
        1, // ac
        76, // depth
        2, // rarity
        15000, // exp
        0, // num_blows
        0x00, // flags
        { // blows
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 584: Makar, the Warrior
    {
        120, // speed
        6000, // hp
        192, // ac
        76, // depth
        5, // rarity
        37000, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {28, 13, 13, 0},
            {5, 13, 13, 0},
            {28, 13, 13, 0},
            {5, 13, 13, 0}
        }
    },
    // 585: Feagwath, the Undead Sorcerer
    {
        130, // speed
        5000, // hp
        127, // ac
        77, // depth
        2, // rarity
        30000, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {16, 6, 12, 0},
            {7, 6, 12, 0},
            {22, 6, 12, 0},
            {22, 6, 12, 0}
        }
    },
    // 586: pit fiend
    {
        130, // speed
        3520, // hp
        180, // ac
        77, // depth
        3, // rarity
        22000, // exp
        4, // num_blows
        0x80, // flags
        { // blows
            {17, 6, 10, 0},
            {17, 6, 10, 0},
            {27, 5, 10, 0},
            {21, 5, 10, 0}
        }
    },
    // 587: Uvatha the Horseman
    {
        120, // speed
        4200, // hp
        72, // ac
        77, // depth
        3, // rarity
        33000, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {19, 6, 9, 0},
            {2, 6, 9, 0},
            {16, 4, 6, 0},
            {16, 4, 6, 0}
        }
    },
    // 588: serpent of chaos
    {
        130, // speed
        900, // hp
        145, // ac
        78, // depth
        3, // rarity
        17000, // exp
        2, // num_blows
        0x00, // flags
        { // blows
            {19, 15, 10, 0},
            {5, 15, 10, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 589: Adunaphel the Quiet
    {
        120, // speed
        4300, // hp
        72, // ac
        78, // depth
        3, // rarity
        34000, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {19, 6, 9, 0},
            {2, 6, 9, 0},
            {16, 5, 6, 0},
            {16, 5, 6, 0}
        }
    },
    // 590: Qlzqqlzuup, the Emperor Quylthulg
    {
        130, // speed
        5000, // hp
        1, // ac
        79, // depth
        3, // rarity
        20000, // exp
        0, // num_blows
        0x01, // flags
        { // blows
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // 591: Akhorahil the Blind
    {
        120, // speed
        4400, // hp
        84, // ac
        79, // depth
        3, // rarity
        35000, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {19, 7, 9, 0},
            {2, 7, 9, 0},
            {16, 6, 6, 0},
            {29, 6, 6, 0}
        }
    },
    // 592: fury
    {
        130, // speed
        4000, // hp
        120, // ac
        80, // depth
        2, // rarity
        23000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 6, 8, 0},
            {19, 6, 8, 0},
            {5, 5, 10, 0},
            {18, 4, 10, 0}
        }
    },
    // 593: great wyrm of annihilation
    {
        120, // speed
        5000, // hp
        255, // ac
        80, // depth
        2, // rarity
        31000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 6, 12, 0},
            {19, 6, 12, 0},
            {19, 7, 14, 0},
            {19, 7, 14, 0}
        }
    },
    // 594: sky dragon
    {
        120, // speed
        5000, // hp
        255, // ac
        80, // depth
        2, // rarity
        31000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 6, 12, 0},
            {19, 6, 12, 0},
            {19, 7, 14, 0},
            {19, 7, 14, 0}
        }
    },
    // 595: Wiruin, the Maelstrom
    {
        140, // speed
        3500, // hp
        10, // ac
        80, // depth
        2, // rarity
        18000, // exp
        3, // num_blows
        0x01, // flags
        { // blows
            {4, 3, 6, 0},
            {5, 3, 6, 0},
            {19, 3, 6, 0},
            {0, 0, 0, 0}
        }
    },
    // 596: Ren the Unclean
    {
        120, // speed
        4500, // hp
        84, // ac
        80, // depth
        3, // rarity
        36000, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {19, 7, 9, 0},
            {2, 7, 9, 0},
            {16, 6, 7, 0},
            {29, 6, 7, 0}
        }
    },
    // 597: Maeglin, the Traitor of Gondolin
    {
        130, // speed
        6000, // hp
        144, // ac
        81, // depth
        2, // rarity
        35000, // exp
        4, // num_blows
        0x03, // flags
        { // blows
            {19, 8, 8, 0},
            {19, 8, 8, 0},
            {19, 8, 8, 0},
            {19, 8, 8, 0}
        }
    },
    // 598: Ji Indur Dawndeath
    {
        120, // speed
        4600, // hp
        84, // ac
        81, // depth
        3, // rarity
        37000, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {19, 8, 9, 0},
            {2, 8, 9, 0},
            {15, 6, 7, 0},
            {15, 6, 7, 0}
        }
    },
    // 599: great wyrm of balance
    {
        120, // speed
        5200, // hp
        255, // ac
        82, // depth
        2, // rarity
        33000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 6, 12, 0},
            {19, 6, 12, 0},
            {19, 7, 14, 0},
            {19, 7, 14, 0}
        }
    },
    // 600: great wyrm of many colours
    {
        120, // speed
        5200, // hp
        255, // ac
        82, // depth
        2, // rarity
        33000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {19, 6, 12, 0},
            {19, 6, 12, 0},
            {19, 7, 14, 0},
            {19, 7, 14, 0}
        }
    },
    // 601: Pazuzu, Lord of Air
    {
        140, // speed
        5500, // hp
        150, // ac
        82, // depth
        2, // rarity
        30000, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {12, 12, 12, 0},
            {12, 12, 12, 0},
            {12, 12, 12, 0},
            {12, 12, 12, 0}
        }
    },
    // 602: Dwar, Dog Lord of Waw
    {
        120, // speed
        4700, // hp
        108, // ac
        82, // depth
        3, // rarity
        38000, // exp
        4, // num_blows
        0x41, // flags
        { // blows
            {19, 9, 9, 0},
            {2, 8, 9, 0},
            {15, 6, 7, 0},
            {29, 6, 7, 0}
        }
    },
    // 603: Hoarmurath of Dir
    {
        120, // speed
        4800, // hp
        120, // ac
        83, // depth
        3, // rarity
        39000, // exp
        4, // num_blows
        0x41, // flags
        { // blows
            {19, 9, 9, 0},
            {2, 9, 9, 0},
            {16, 6, 7, 0},
            {29, 6, 7, 0}
        }
    },
    // 604: Draugluin, Sire of All Werewolves
    {
        130, // speed
        7000, // hp
        135, // ac
        83, // depth
        2, // rarity
        40000, // exp
        4, // num_blows
        0x03, // flags
        { // blows
            {19, 6, 8, 0},
            {19, 6, 8, 0},
            {27, 6, 6, 0},
            {27, 6, 6, 0}
        }
    },
    // 605: Khamûl, the Black Easterling
    {
        120, // speed
        4900, // hp
        120, // ac
        84, // depth
        3, // rarity
        40000, // exp
        4, // num_blows
        0x41, // flags
        { // blows
            {19, 9, 10, 0},
            {2, 9, 10, 0},
            {15, 7, 7, 0},
            {15, 7, 7, 0}
        }
    },
    // 606: Cantoras, the Skeletal Lord
    {
        140, // speed
        7500, // hp
        180, // ac
        84, // depth
        2, // rarity
        45000, // exp
        4, // num_blows
        0x41, // flags
        { // blows
            {16, 5, 5, 0},
            {16, 5, 5, 0},
            {27, 5, 5, 0},
            {27, 5, 5, 0}
        }
    },
    // 607: greater Balrog
    {
        130, // speed
        4500, // hp
        210, // ac
        85, // depth
        3, // rarity
        25000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {17, 8, 12, 0},
            {17, 8, 12, 0},
            {19, 9, 12, 0},
            {7, 0, 0, 0}
        }
    },
    // 608: The Witch-King of Angmar
    {
        130, // speed
        5000, // hp
        144, // ac
        85, // depth
        3, // rarity
        60000, // exp
        4, // num_blows
        0x41, // flags
        { // blows
            {19, 10, 10, 0},
            {2, 10, 10, 0},
            {16, 7, 7, 0},
            {16, 7, 7, 0}
        }
    },
    // 609: Ancalagon the Black
    {
        130, // speed
        10000, // hp
        255, // ac
        85, // depth
        3, // rarity
        45000, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {19, 8, 12, 0},
            {19, 8, 12, 0},
            {19, 10, 14, 0},
            {19, 10, 14, 0}
        }
    },
    // 610: The Tarrasque
    {
        130, // speed
        8500, // hp
        222, // ac
        86, // depth
        2, // rarity
        35000, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {19, 10, 10, 0},
            {19, 10, 10, 0},
            {7, 0, 0, 0},
            {7, 0, 0, 0}
        }
    },
    // 611: Meássë, the Bloody
    {
        120, // speed
        7000, // hp
        180, // ac
        87, // depth
        3, // rarity
        42000, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {5, 12, 12, 0},
            {5, 12, 12, 0},
            {5, 12, 12, 0},
            {5, 12, 12, 0}
        }
    },
    // 612: Lungorthin, the Balrog of White Fire
    {
        130, // speed
        7000, // hp
        187, // ac
        88, // depth
        2, // rarity
        37000, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {17, 8, 12, 0},
            {17, 8, 12, 0},
            {19, 8, 12, 0},
            {7, 0, 0, 0}
        }
    },
    // 613: Huan, Wolfhound of the Valar
    {
        130, // speed
        8000, // hp
        192, // ac
        90, // depth
        1, // rarity
        40000, // exp
        4, // num_blows
        0x41, // flags
        { // blows
            {4, 9, 12, 0},
            {4, 9, 12, 0},
            {4, 9, 12, 0},
            {4, 9, 12, 0}
        }
    },
    // 614: Carcharoth, the Jaws of Thirst
    {
        130, // speed
        8000, // hp
        192, // ac
        90, // depth
        1, // rarity
        40000, // exp
        4, // num_blows
        0x43, // flags
        { // blows
            {17, 9, 12, 0},
            {17, 9, 12, 0},
            {17, 9, 12, 0},
            {17, 9, 12, 0}
        }
    },
    // 615: Vecna, the Emperor Lich
    {
        130, // speed
        6000, // hp
        120, // ac
        92, // depth
        3, // rarity
        45000, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {16, 7, 12, 0},
            {22, 7, 12, 0},
            {7, 7, 12, 0},
            {7, 7, 12, 0}
        }
    },
    // 616: storm of Unmagic
    {
        140, // speed
        600, // hp
        48, // ac
        95, // depth
        5, // rarity
        17000, // exp
        4, // num_blows
        0x00, // flags
        { // blows
            {16, 5, 7, 0},
            {7, 5, 7, 0},
            {6, 5, 7, 0},
            {20, 5, 7, 0}
        }
    },
    // 617: Gothmog, the High Captain of Balrogs
    {
        130, // speed
        8000, // hp
        168, // ac
        95, // depth
        1, // rarity
        43000, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {17, 9, 12, 0},
            {17, 9, 12, 0},
            {19, 8, 12, 0},
            {7, 0, 0, 0}
        }
    },
    // 618: Sauron, the Sorcerer
    {
        130, // speed
        8000, // hp
        192, // ac
        99, // depth
        1, // rarity
        50000, // exp
        4, // num_blows
        0x83, // flags
        { // blows
            {6, 10, 12, 0},
            {6, 10, 12, 0},
            {7, 8, 12, 0},
            {7, 8, 12, 0}
        }
    },
    // 619: Wolf-Sauron
    {
        140, // speed
        1, // hp
        164, // ac
        99, // depth
        0, // rarity
        0, // exp
        4, // num_blows
        0xC3, // flags
        { // blows
            {27, 8, 10, 0},
            {27, 8, 10, 0},
            {5, 6, 8, 0},
            {26, 6, 8, 0}
        }
    },
    // 620: Serpent-Sauron
    {
        130, // speed
        1, // hp
        164, // ac
        99, // depth
        0, // rarity
        0, // exp
        4, // num_blows
        0xC3, // flags
        { // blows
            {27, 7, 12, 0},
            {27, 7, 12, 0},
            {19, 8, 12, 0},
            {1, 4, 12, 0}
        }
    },
    // 621: Vampire-Sauron
    {
        130, // speed
        1, // hp
        164, // ac
        99, // depth
        0, // rarity
        0, // exp
        4, // num_blows
        0x01, // flags
        { // blows
            {16, 6, 12, 0},
            {29, 8, 12, 0},
            {7, 8, 12, 0},
            {7, 8, 12, 0}
        }
    },
    // 622: Morgoth, Lord of Darkness
    {
        140, // speed
        20000, // hp
        180, // ac
        100, // depth
        1, // rarity
        60000, // exp
        4, // num_blows
        0xC3, // flags
        { // blows
            {28, 20, 10, 0},
            {28, 20, 10, 0},
            {20, 10, 12, 0},
            {7, 0, 0, 0}
        }
    }
};

__device__ int NUM_MONSTER_RACES = MAX_MONSTER_RACES;

// Get appropriate monster for given depth
// Uses inverse rarity as probability weight
__device__ inline int get_monster_for_depth(int depth, curandState* rng) {
    // Simple algorithm: pick random monster at or below depth
    // In full implementation, would use rarity-weighted selection

    int candidates[MAX_MONSTER_RACES];
    int count = 0;

    // Find all monsters at or below this depth
    for (int i = 0; i < NUM_MONSTER_RACES; i++) {
        if (MONSTER_RACES[i].depth <= depth && MONSTER_RACES[i].depth > 0) {
            candidates[count++] = i;
        }
    }

    if (count == 0) {
        // No valid monsters, return weakest non-town monster
        for (int i = 0; i < NUM_MONSTER_RACES; i++) {
            if (MONSTER_RACES[i].depth > 0) {
                return i;
            }
        }
        return 0;
    }

    // Pick random candidate
    float r = curand_uniform(rng);
    int idx = (int)(r * count);
    if (idx >= count) idx = count - 1;

    return candidates[idx];
}

// Monster attack calculation
// Returns damage dealt to player
__device__ inline int monster_attack(int race_idx, int player_ac, curandState* rng) {
    if (race_idx < 0 || race_idx >= NUM_MONSTER_RACES) {
        return 0;
    }

    MonsterRace* monster = &MONSTER_RACES[race_idx];
    int total_damage = 0;

    // Process each blow
    for (int i = 0; i < monster->num_blows; i++) {
        MonsterBlow* blow = &monster->blows[i];
        if (blow->effect == 0) continue;

        // Hit roll: monster hits if random(100) > player_ac * 2/3
        float hit_roll = curand_uniform(rng) * 100.0f;
        float defense = player_ac * (2.0f / 3.0f);

        if (hit_roll > defense) {
            // Hit! Roll damage
            int damage = 0;
            for (int d = 0; d < blow->dd; d++) {
                float r = curand_uniform(rng);
                damage += (int)(r * blow->ds) + 1;
            }

            total_damage += damage;
        }
    }

    return total_damage;
}

// Get monster name index (for display)
// In full implementation, would have string table
__device__ inline const char* get_monster_name(int race_idx) {
    // Would need to add name strings to constant memory
    // For now, just return generic name
    return "monster";
}

#endif // ANGBAND_MONSTERS_CUH
