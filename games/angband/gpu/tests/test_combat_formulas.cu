/*
 * ANGBAND COMBAT FORMULA TESTS
 *
 * Purpose: Verify GPU implementations match REAL Angband formulas.
 * Reference: Angband 4.2.x source (github.com/angband/angband)
 *
 * CRITICAL: These tests detect LLM "hallucination drift" - where AI-generated
 * code diverges from actual game mechanics.
 *
 * Build: nvcc -o test_combat test_combat_formulas.cu -arch=sm_90
 * Run: ./test_combat
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// ============================================================================
// CPU VERSIONS OF FORMULAS (for testing without GPU)
// These MUST match the __device__ versions in angband_combat.cuh
// ============================================================================

// adjust_dam_armor - EXACT copy from angband_combat.cuh:132-137
// Reference: angband/src/mon-attack.c:295 (adjust_dam_armor)
// Original: return (dam - (dam * MIN(ac, 240) / 400));
int adjust_dam_armor_cpu(int damage, int ac) {
    int effective_ac = (ac < 240) ? ac : 240;
    return damage - (damage * effective_ac / 400);
}

// Expected damage calculation for dice XdY
// Reference: Standard RPG formula - expected value of XdY = X * (Y+1) / 2
int expected_dice_damage(int dice_count, int dice_sides) {
    return dice_count * (dice_sides + 1) / 2;
}

// ============================================================================
// TEST CASES
// ============================================================================

typedef struct {
    int damage;
    int ac;
    int expected;
    const char* description;
} ArmorTestCase;

// Test cases derived from Angband formula: damage - (damage * min(ac, 240) / 400)
// AC range: [0, 240] -> reduction range: [0%, 60%]
ArmorTestCase armor_tests[] = {
    // Basic cases
    {100, 0, 100, "AC 0: 0% reduction"},
    {100, 40, 90, "AC 40: 10% reduction"},
    {100, 80, 80, "AC 80: 20% reduction"},
    {100, 100, 75, "AC 100: 25% reduction"},
    {100, 120, 70, "AC 120: 30% reduction"},
    {100, 160, 60, "AC 160: 40% reduction"},
    {100, 200, 50, "AC 200: 50% reduction"},
    {100, 240, 40, "AC 240: 60% max reduction"},

    // AC cap verification (ac > 240 should clamp to 240)
    {100, 300, 40, "AC 300: capped at 60% reduction"},
    {100, 500, 40, "AC 500: capped at 60% reduction"},
    {100, 1000, 40, "AC 1000: capped at 60% reduction"},

    // Edge cases
    {0, 100, 0, "0 damage: always 0"},
    {1, 100, 1, "1 damage, AC 100: 1-(1*100/400)=1-0=1 (int div)"},
    {1, 0, 1, "1 damage, AC 0: unchanged"},
    {10, 200, 5, "10 damage, AC 200: 50% = 5"},

    // Real game scenarios (typical early/mid/late game)
    {50, 30, 47, "Early game: 50-(50*30/400)=50-3=47"},
    {200, 100, 150, "Mid game: 200 dmg vs AC 100"},
    {500, 200, 250, "Late game: 500 dmg vs AC 200"},
    {1000, 240, 400, "Endgame: 1000 dmg vs AC 240"},
};

// Dice expectation tests
typedef struct {
    int dd;  // dice count
    int ds;  // dice sides
    int expected;
    const char* description;
} DiceTestCase;

DiceTestCase dice_tests[] = {
    {1, 4, 2, "1d4: expected 2.5 -> 2"},
    {1, 6, 3, "1d6: expected 3.5 -> 3"},
    {1, 8, 4, "1d8: expected 4.5 -> 4"},
    {2, 6, 7, "2d6: expected 7"},
    {3, 6, 10, "3d6: expected 10.5 -> 10"},
    {4, 4, 10, "4d4: expected 10"},
    {6, 6, 21, "6d6: expected 21"},
    {10, 10, 55, "10d10: expected 55"},
};

// ============================================================================
// TEST RUNNER
// ============================================================================

int test_adjust_dam_armor() {
    printf("=== Testing adjust_dam_armor() ===\n");
    printf("Reference: angband/src/mon-attack.c:295\n");
    printf("Formula: damage - (damage * min(ac, 240) / 400)\n\n");

    int passed = 0;
    int failed = 0;
    int num_tests = sizeof(armor_tests) / sizeof(armor_tests[0]);

    for (int i = 0; i < num_tests; i++) {
        ArmorTestCase* tc = &armor_tests[i];
        int result = adjust_dam_armor_cpu(tc->damage, tc->ac);

        if (result == tc->expected) {
            printf("[PASS] %s\n", tc->description);
            printf("       adjust_dam_armor(%d, %d) = %d\n\n",
                   tc->damage, tc->ac, result);
            passed++;
        } else {
            printf("[FAIL] %s\n", tc->description);
            printf("       adjust_dam_armor(%d, %d) = %d (expected %d)\n\n",
                   tc->damage, tc->ac, result, tc->expected);
            failed++;
        }
    }

    printf("Armor tests: %d passed, %d failed\n\n", passed, failed);
    return failed;
}

int test_dice_expectations() {
    printf("=== Testing expected_dice_damage() ===\n");
    printf("Formula: dice_count * (dice_sides + 1) / 2\n\n");

    int passed = 0;
    int failed = 0;
    int num_tests = sizeof(dice_tests) / sizeof(dice_tests[0]);

    for (int i = 0; i < num_tests; i++) {
        DiceTestCase* tc = &dice_tests[i];
        int result = expected_dice_damage(tc->dd, tc->ds);

        if (result == tc->expected) {
            printf("[PASS] %s\n", tc->description);
            printf("       expected_dice_damage(%d, %d) = %d\n\n",
                   tc->dd, tc->ds, result);
            passed++;
        } else {
            printf("[FAIL] %s\n", tc->description);
            printf("       expected_dice_damage(%d, %d) = %d (expected %d)\n\n",
                   tc->dd, tc->ds, result, tc->expected);
            failed++;
        }
    }

    printf("Dice tests: %d passed, %d failed\n\n", passed, failed);
    return failed;
}

// Test that armor formula has correct mathematical properties
int test_armor_properties() {
    printf("=== Testing adjust_dam_armor() properties ===\n\n");

    int failed = 0;

    // Property 1: Damage should always be non-negative
    printf("Property 1: Result >= 0 for all inputs\n");
    for (int dam = 0; dam <= 1000; dam += 100) {
        for (int ac = 0; ac <= 500; ac += 50) {
            int result = adjust_dam_armor_cpu(dam, ac);
            if (result < 0) {
                printf("[FAIL] adjust_dam_armor(%d, %d) = %d < 0\n", dam, ac, result);
                failed++;
            }
        }
    }
    if (failed == 0) printf("[PASS] All results non-negative\n\n");

    // Property 2: Higher AC = more reduction (monotonic)
    printf("Property 2: Higher AC -> more reduction (monotonic)\n");
    int mono_failed = 0;
    for (int dam = 10; dam <= 1000; dam += 100) {
        int prev = adjust_dam_armor_cpu(dam, 0);
        for (int ac = 10; ac <= 240; ac += 10) {
            int curr = adjust_dam_armor_cpu(dam, ac);
            if (curr > prev) {
                printf("[FAIL] Non-monotonic: dam=%d, prev_ac=%d->%d, ac=%d->%d\n",
                       dam, ac-10, prev, ac, curr);
                mono_failed++;
            }
            prev = curr;
        }
    }
    if (mono_failed == 0) printf("[PASS] Monotonically decreasing with AC\n\n");
    failed += mono_failed;

    // Property 3: AC cap at 240
    printf("Property 3: AC capped at 240\n");
    int cap_failed = 0;
    for (int dam = 10; dam <= 1000; dam += 100) {
        int at_cap = adjust_dam_armor_cpu(dam, 240);
        for (int ac = 250; ac <= 1000; ac += 50) {
            int beyond = adjust_dam_armor_cpu(dam, ac);
            if (beyond != at_cap) {
                printf("[FAIL] AC beyond cap differs: dam=%d, at_240=%d, at_%d=%d\n",
                       dam, at_cap, ac, beyond);
                cap_failed++;
            }
        }
    }
    if (cap_failed == 0) printf("[PASS] AC correctly capped at 240\n\n");
    failed += cap_failed;

    // Property 4: Max reduction is 60%
    printf("Property 4: Max reduction is 60%%\n");
    int max_failed = 0;
    for (int dam = 10; dam <= 1000; dam += 100) {
        int result = adjust_dam_armor_cpu(dam, 240);
        int expected_min = dam * 40 / 100;  // 40% of original = 60% reduction
        if (result != expected_min) {
            printf("[FAIL] Max reduction wrong: dam=%d, result=%d, expected=%d\n",
                   dam, result, expected_min);
            max_failed++;
        }
    }
    if (max_failed == 0) printf("[PASS] Max reduction is exactly 60%%\n\n");
    failed += max_failed;

    return failed;
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║     ANGBAND COMBAT FORMULA VERIFICATION TESTS                ║\n");
    printf("║     Detecting LLM Hallucination Drift                        ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    int total_failed = 0;

    total_failed += test_adjust_dam_armor();
    total_failed += test_dice_expectations();
    total_failed += test_armor_properties();

    printf("═══════════════════════════════════════════════════════════════\n");
    if (total_failed == 0) {
        printf("ALL TESTS PASSED - Implementation matches Angband formulas\n");
        return 0;
    } else {
        printf("TESTS FAILED: %d - LLM DRIFT DETECTED\n", total_failed);
        printf("Review implementations against Angband source!\n");
        return 1;
    }
}
