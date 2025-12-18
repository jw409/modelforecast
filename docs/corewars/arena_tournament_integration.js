/**
 * Arena Tournament Integration
 * Integrates TournamentLoader with the arena visualization
 *
 * Call initTournamentLoader() after the arena is initialized
 */

// Global tournament loader instance
let tournamentLoader = null;

/**
 * Initialize tournament loader and load latest results
 * Meant to be called from arena_full.html after LLMS array is defined
 * @param {Array} llmsArray - Reference to the LLMS array in arena
 * @param {Object} leaderboardObj - Reference to the leaderboard object in arena
 * @param {Function} updateLeaderboardUIFn - Reference to updateLeaderboardUI function
 */
async function initTournamentLoader(llmsArray, leaderboardObj, updateLeaderboardUIFn) {
    console.log('Initializing tournament loader...');

    tournamentLoader = new TournamentLoader();

    // Try to load latest tournament
    const tournamentData = await tournamentLoader.loadLatestTournament();

    if (tournamentData) {
        console.log('Tournament data loaded successfully:', tournamentData);
        console.log('Tournament Summary:', tournamentLoader.getSummary());

        // Apply tournament data to leaderboard
        updateLeaderboardWithTournamentData(llmsArray, leaderboardObj, updateLeaderboardUIFn);

        return true;
    } else {
        console.warn('No tournament data available, using default leaderboard');
        return false;
    }
}

/**
 * Update leaderboard with tournament data
 * @param {Array} llmsArray - LLMS array from arena
 * @param {Object} leaderboardObj - leaderboard object from arena
 * @param {Function} updateLeaderboardUIFn - updateLeaderboardUI function reference
 */
function updateLeaderboardWithTournamentData(llmsArray, leaderboardObj, updateLeaderboardUIFn) {
    const mapping = tournamentLoader.createLlmMapping(llmsArray);

    // Reset leaderboard
    Object.keys(leaderboardObj).forEach(key => {
        leaderboardObj[key] = { wins: 0, cards: [], streak: 0 };
    });

    // Populate from tournament data
    const rankings = tournamentLoader.getRankings();
    rankings.forEach((stats, index) => {
        // Find corresponding LLM
        for (const [llmId, warrior] of Object.entries(mapping)) {
            if (warrior === stats.warrior) {
                const llm = llmsArray.find(l => l.id === llmId);
                if (llm) {
                    // Update leaderboard entry
                    leaderboardObj[llmId] = {
                        wins: stats.wins,
                        cards: [],
                        streak: 0,
                        winRate: stats.winRate,
                        totalBattles: stats.battleCount
                    };

                    // Update LLM metadata
                    llm.tournamentRank = index + 1;
                    llm.tournamentWins = stats.wins;
                    llm.tournamentWinRate = stats.winRate;
                }
                break;
            }
        }
    });

    // Update UI
    updateLeaderboardUIFn();

    // Display tournament info
    showTournamentStats();
}

/**
 * Display tournament statistics overlay
 */
function showTournamentStats() {
    if (!tournamentLoader) return;

    const summary = tournamentLoader.getSummary();

    // Create or update stats panel
    let statsPanel = document.getElementById('tournament-stats-panel');
    if (!statsPanel) {
        statsPanel = document.createElement('div');
        statsPanel.id = 'tournament-stats-panel';
        statsPanel.style.cssText = `
            position: fixed;
            top: 10px;
            left: 10px;
            background: rgba(5, 5, 8, 0.95);
            border: 1px solid #00fff7;
            border-radius: 4px;
            padding: 12px 16px;
            font-size: 11px;
            font-family: 'Courier New', monospace;
            color: #00fff7;
            z-index: 98;
            max-width: 300px;
            box-shadow: 0 4px 12px rgba(0, 255, 247, 0.15);
        `;
        document.body.appendChild(statsPanel);
    }

    const timestamp = summary.timestamp
        ? new Date(summary.timestamp).toLocaleString()
        : 'Unknown';

    statsPanel.innerHTML = `
        <div style="font-weight: bold; margin-bottom: 8px; color: #fff;">Tournament Results</div>
        <div style="margin-bottom: 4px;">Warriors: ${summary.totalWarriors}</div>
        <div style="margin-bottom: 4px;">Battles/Matchup: ${summary.battlesPerMatchup.toLocaleString()}</div>
        <div style="margin-bottom: 8px; color: #888; font-size: 10px;">${timestamp}</div>
        <div style="border-top: 1px solid rgba(0, 255, 247, 0.3); padding-top: 8px;">
            <div style="color: #ffd700; font-weight: bold; margin-bottom: 4px;">🏆 Top Warrior</div>
            <div style="margin-left: 8px;">
                ${summary.topWarrior
                    ? `${TournamentLoader.formatWarriorName(summary.topWarrior.warrior)}<br/>Wins: ${summary.topWarrior.wins}<br/>Rate: ${summary.topWarrior.winRate}%`
                    : 'N/A'
                }
            </div>
        </div>
    `;
}

/**
 * Display head-to-head matchup stats
 * @param {string} w1 - First warrior/LLM ID
 * @param {string} w2 - Second warrior/LLM ID
 */
function showHeadToHeadStats(w1, w2) {
    if (!tournamentLoader) return;

    // Get warrior names from mapping if w1/w2 are LLM IDs
    let warrior1 = w1;
    let warrior2 = w2;

    // Look up in existing leaderboard mapping if available
    // This is a simplified version - in real usage you'd maintain the mapping

    const h2h = tournamentLoader.getHeadToHead(warrior1, warrior2);
    if (!h2h) {
        console.log('No head-to-head data available');
        return;
    }

    let h2hPanel = document.getElementById('h2h-stats-panel');
    if (!h2hPanel) {
        h2hPanel = document.createElement('div');
        h2hPanel.id = 'h2h-stats-panel';
        h2hPanel.style.cssText = `
            position: fixed;
            top: 180px;
            left: 10px;
            background: rgba(5, 5, 8, 0.95);
            border: 1px solid #ff6b00;
            border-radius: 4px;
            padding: 12px 16px;
            font-size: 11px;
            font-family: 'Courier New', monospace;
            color: #ff6b00;
            z-index: 98;
            box-shadow: 0 4px 12px rgba(255, 107, 0, 0.15);
        `;
        document.body.appendChild(h2hPanel);
    }

    const total = h2h.w1Wins + h2h.w2Wins + h2h.ties;
    const w1Pct = total > 0 ? ((h2h.w1Wins / total) * 100).toFixed(1) : 0;
    const w2Pct = total > 0 ? ((h2h.w2Wins / total) * 100).toFixed(1) : 0;

    h2hPanel.innerHTML = `
        <div style="font-weight: bold; margin-bottom: 8px; color: #fff;">Head-to-Head</div>
        <div style="margin-bottom: 4px;">${TournamentLoader.formatWarriorName(h2h.warrior1)}</div>
        <div style="margin-bottom: 8px; color: #00fff7;">${h2h.w1Wins} wins (${w1Pct}%)</div>
        <div style="margin-bottom: 4px;">${TournamentLoader.formatWarriorName(h2h.warrior2)}</div>
        <div style="margin-bottom: 8px; color: #00fff7;">${h2h.w2Wins} wins (${w2Pct}%)</div>
        ${h2h.ties > 0 ? `<div style="color: #888;">Ties: ${h2h.ties}</div>` : ''}
    `;
}

/**
 * Export tournament rankings as CSV and trigger download
 */
function exportTournamentAsCSV() {
    if (!tournamentLoader) {
        alert('No tournament data available');
        return;
    }

    const csv = tournamentLoader.exportAsCSV();
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `tournament-rankings-${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
}

/**
 * Log detailed tournament statistics to console
 */
function logTournamentStats() {
    if (!tournamentLoader) {
        console.log('No tournament data available');
        return;
    }

    console.group('Tournament Statistics');
    console.log('Summary:', tournamentLoader.getSummary());
    console.log('Rankings:', tournamentLoader.getRankings());
    console.groupEnd();
}

// Make functions available globally
window.initTournamentLoader = initTournamentLoader;
window.showHeadToHeadStats = showHeadToHeadStats;
window.exportTournamentAsCSV = exportTournamentAsCSV;
window.logTournamentStats = logTournamentStats;
