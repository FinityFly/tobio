// src/statistics.ts
import type { VolleyballEvent } from './components/EventViewer';

// --- Interface Definitions ---

// Structure for raw aggregated data during processing
interface PlayerStats {
    gamesPlayed: number;
    totalTouches: number;
    serves: number;
    attacks: number;
    sets: number;
    digs: number;
    blocks: number;
    setHeightSum: number;
    setCount: number;
    spikeHeightSum: number;
    spikeCount: number;
    blockHeightSum: number;
    blockCount: number;
}

// Structure for the fully parsed data used by the UI
export interface ParsedPlayerStats {
  playerId: string;
  playerName: string;
  Points: number;
  Attacks: number;
  Blocks: number;
  Serves: number;
  Digs: number;
  Sets: number;
  avgBlockHeight: string;
  avgSpikeHeight: string;
  avgSetHeight: string;
}

// Structure for the set distribution metric
export interface SetDistribution {
    [key: string]: number | string;
}

/**
 * Generates a statistics CSV string by analyzing ONLY the volleyball_events array.
 */
export function generateStatsCsv(
    volleyballEvents: VolleyballEvent[] | null,
    playerNames: Record<string, string> = {}
): string {
    const stats: Map<string, PlayerStats> = new Map();
    const teamSetDistribution: { [teamId: string]: { [position: string]: number } } = { '0': {}, '1': {} };

    if (!volleyballEvents) {
        return "Player,Games Played,Total Touches,Serves,Attacks,Sets,Digs,Blocks,Attack Percentage,Avg Set Height (m),Avg Spike Height (m),Avg Block Height (m)\n";
    }

    // 1. Iterate through all events and aggregate stats
    for (const event of volleyballEvents) {
        if (event.player_id === undefined || event.player_id === null) continue;
        
        const playerId = String(event.player_id);

        if (!stats.has(playerId)) {
            stats.set(playerId, {
                gamesPlayed: 1, totalTouches: 0, serves: 0, attacks: 0,
                sets: 0, digs: 0, blocks: 0, setHeightSum: 0,
                setCount: 0, spikeHeightSum: 0, spikeCount: 0,
                blockHeightSum: 0, blockCount: 0,
            });
        }

        const playerStats = stats.get(playerId)!;
        playerStats.totalTouches++;

        switch (event.action) {
            case 'serve':
                playerStats.serves++;
                break;
            case 'spike':
                playerStats.attacks++;
                if (typeof event.ball_height_m === 'number') {
                    playerStats.spikeHeightSum += event.ball_height_m;
                    playerStats.spikeCount++;
                }
                break;
            case 'set':
                playerStats.sets++;
                if (typeof event.ball_height_m === 'number') {
                    playerStats.setHeightSum += event.ball_height_m;
                    playerStats.setCount++;
                }
                if (typeof event.set_position === 'number') {
                    const teamId = parseInt(playerId) <= 6 ? '0' : '1';
                    const roundedPosition = Math.round(event.set_position);
                    
                    if (roundedPosition >= 2 && roundedPosition <= 4) {
                        const posKey = `Position ${roundedPosition}`;
                        teamSetDistribution[teamId][posKey] = (teamSetDistribution[teamId][posKey] || 0) + 1;
                    }
                }
                break;
            case 'defense':
                playerStats.digs++;
                break;
            case 'block':
                playerStats.blocks++;
                if (typeof (event as any).block_height_m === 'number') {
                    playerStats.blockHeightSum += (event as any).block_height_m;
                    playerStats.blockCount++;
                }
                break;
        }
    }

    // 2. Generate CSV String
    const headers = "Player,Games Played,Total Touches,Serves,Attacks,Sets,Digs,Blocks,Attack Percentage,Avg Set Height (m),Avg Spike Height (m),Avg Block Height (m)";
    
    const rows = Array.from(stats.entries())
        .sort(([idA], [idB]) => parseInt(idA) - parseInt(idB))
        .map(([playerId, pStats]) => {
            const displayName = playerNames[playerId] || `Player ${playerId}`;
            const attackPct = pStats.totalTouches > 0 ? `${((pStats.attacks / pStats.totalTouches) * 100).toFixed(1)}%` : '0.0%';
            const avgSetHeight = pStats.setCount > 0 ? (pStats.setHeightSum / pStats.setCount).toFixed(2) : 'N/A';
            const avgSpikeHeight = pStats.spikeCount > 0 ? (pStats.spikeHeightSum / pStats.spikeCount).toFixed(2) : 'N/A';
            const avgBlockHeight = pStats.blockCount > 0 ? (pStats.blockHeightSum / pStats.blockCount).toFixed(2) : 'N/A';

            return `"${displayName}",${pStats.gamesPlayed},${pStats.totalTouches},${pStats.serves},${pStats.attacks},${pStats.sets},${pStats.digs},${pStats.blocks},${attackPct},${avgSetHeight},${avgSpikeHeight},${avgBlockHeight}`;
        });
    
    let csvString = [headers, ...rows].join('\n');
    csvString += '\n\n# Set Distribution\n';
    csvString += 'Team,Position 2,Position 3,Position 4\n';
    csvString += `Team 0,${teamSetDistribution['0']['Position 2'] || 0},${teamSetDistribution['0']['Position 3'] || 0},${teamSetDistribution['0']['Position 4'] || 0}\n`;
    csvString += `Team 1,${teamSetDistribution['1']['Position 2'] || 0},${teamSetDistribution['1']['Position 3'] || 0},${teamSetDistribution['1']['Position 4'] || 0}\n`;

    return csvString;
}

/**
 * Parses the CSV string to extract player stats for the UI panel and set distribution.
 */
export function parseStatsFromCsv(csvString: string | null): { players: ParsedPlayerStats[], teamSetDistribution: SetDistribution[] } {
  if (!csvString) return { players: [], teamSetDistribution: [] };

  const sections = csvString.split('\n\n# Set Distribution\n');
  const playerDataSection = sections[0];
  const setDistributionSection = sections[1];
  
  // --- Parse Player Data ---
  const playerLines = playerDataSection.trim().split('\n');
  if (playerLines.length < 2) return { players: [], teamSetDistribution: [] };
  const playerHeaders = playerLines[0].split(',');
  
  const headerMap = {
    Player: playerHeaders.indexOf('Player'),
    Attacks: playerHeaders.indexOf('Attacks'),
    Blocks: playerHeaders.indexOf('Blocks'),
    Serves: playerHeaders.indexOf('Serves'),
    Digs: playerHeaders.indexOf('Digs'),
    Sets: playerHeaders.indexOf('Sets'),
    AvgBlockHeight: playerHeaders.indexOf('Avg Block Height (m)'),
    AvgSpikeHeight: playerHeaders.indexOf('Avg Spike Height (m)'),
    AvgSetHeight: playerHeaders.indexOf('Avg Set Height (m)'),
  };

  const players = playerLines.slice(1).map(line => {
      const values = line.split(',');
      const playerName = values[headerMap.Player]?.replace(/"/g, '') || 'Unknown';
      if (playerName.trim() === 'Team Total') return null;

      const playerIdMatch = playerName.match(/\s(\d+)$/);
      const playerId = playerIdMatch ? playerIdMatch[1] : playerName;

      const attacks = parseInt(values[headerMap.Attacks]) || 0;
      const blocks = parseInt(values[headerMap.Blocks]) || 0;
      const serves = parseInt(values[headerMap.Serves]) || 0;

      return {
        playerId,
        playerName,
        Points: attacks + blocks + serves,
        Attacks: attacks,
        Blocks: blocks,
        Serves: serves,
        Digs: parseInt(values[headerMap.Digs]) || 0,
        Sets: parseInt(values[headerMap.Sets]) || 0,
        avgBlockHeight: values[headerMap.AvgBlockHeight] || 'N/A',
        avgSpikeHeight: values[headerMap.AvgSpikeHeight] || 'N/A',
        avgSetHeight: values[headerMap.AvgSetHeight] || 'N/A',
      };
  }).filter((p): p is ParsedPlayerStats => p !== null);

  // --- Parse Set Distribution Data ---
  const teamSetDistribution: SetDistribution[] = [];
  if (setDistributionSection) {
      const setLines = setDistributionSection.trim().split('\n');
      if (setLines.length > 1) {
          const team0Values = setLines[1].split(',');
          const team1Values = setLines[2] ? setLines[2].split(',') : ['Team 1', '0', '0', '0'];

          teamSetDistribution.push({ Team: 'Team 0', 'Position 2': parseInt(team0Values[1]) || 0, 'Position 3': parseInt(team0Values[2]) || 0, 'Position 4': parseInt(team0Values[3]) || 0 });
          teamSetDistribution.push({ Team: 'Team 1', 'Position 2': parseInt(team1Values[1]) || 0, 'Position 3': parseInt(team1Values[2]) || 0, 'Position 4': parseInt(team1Values[3]) || 0 });
      }
  }

  return { players, teamSetDistribution };
}