import React, { useMemo, useState, useEffect } from 'react';
import { parseStatsFromCsv } from '../statistics';
import CsvModal from './CsvModal';

interface StatisticsPanelProps {
  csvData: string | null;
  onExport: () => void;
  csvReady: boolean;
  playerNames: Record<string, string>;
  onPlayerNameChange: (playerId: string, newName: string) => void;
}

const EditablePlayerName: React.FC<{
  playerId: string;
  initialName: string;
  onNameChange: (playerId: string, newName: string) => void;
}> = ({ playerId, initialName, onNameChange }) => {
  const [isEditing, setIsEditing] = useState(false);
  const [name, setName] = useState(initialName);

  useEffect(() => {
    setName(initialName);
  }, [initialName]);

  const handleDoubleClick = () => setIsEditing(true);
  
  const handleSave = () => {
    setIsEditing(false);
    onNameChange(playerId, name);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') handleSave();
    if (e.key === 'Escape') {
      setName(initialName);
      setIsEditing(false);
    }
  };

  if (isEditing) {
    return (
      <input
        type="text"
        value={name}
        onChange={(e) => setName(e.target.value)}
        onBlur={handleSave}
        onKeyDown={handleKeyDown}
        autoFocus
        style={{ width: '100%', border: '1px solid #ccc', borderRadius: '4px', padding: '2px 4px' }}
      />
    );
  }

  return (
    <div onDoubleClick={handleDoubleClick} title="Double-click to rename">
      {name}
    </div>
  );
};

export const StatisticsPanel: React.FC<StatisticsPanelProps> = ({ csvData, onExport, csvReady, playerNames, onPlayerNameChange }) => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const { players, teamSetDistribution } = useMemo(() => {
    return parseStatsFromCsv(csvData);
  }, [csvData]);

  const handleOpenModal = () => {
    setIsModalOpen(true);
  }

  const handleCloseModal = () => {
    setIsModalOpen(false);
  };

  return (
    <div className="statistics-panel">
      <div className="statistics-list">
        {players.length > 0 ? (
          players.map((player) => (
            <div className="statistics-card" key={player.playerId}>
              <div className="statistics-label" style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: '8px', color: '#1e293b' }}>
                <EditablePlayerName 
                  playerId={player.playerId}
                  initialName={playerNames[player.playerId] || player.playerName}
                  onNameChange={onPlayerNameChange}
                />
              </div>
              
              {/* Row 1: Main Count Stats */}
              <div className="statistics-stats-row">
                <div className="statistics-stat"><strong>{player.Points}</strong><span>Pts</span></div>
                <div className="statistics-stat"><strong>{player.Attacks}</strong><span>Atk</span></div>
                <div className="statistics-stat"><strong>{player.Blocks}</strong><span>Blk</span></div>
                <div className="statistics-stat"><strong>{player.Serves}</strong><span>Srv</span></div>
              </div>
              
              {/* Row 2: Secondary Count Stats */}
              <div className="statistics-stats-row" style={{ marginTop: '4px' }}>
                <div className="statistics-stat"><strong>{player.Digs}</strong><span>Digs</span></div>
                <div className="statistics-stat"><strong>{player.Sets}</strong><span>Sets</span></div>
              </div>

              {/* Row 3: Average Metrics */}
              <div className="statistics-stats-row" style={{ marginTop: '8px', paddingTop: '8px', borderTop: '1px solid #e2e8f0', gap: '12px' }}>
                <div className="statistics-stat"><strong>{player.avgSpikeHeight}</strong><span>Atk Ht</span></div>
                <div className="statistics-stat"><strong>{player.avgSetHeight}</strong><span>Set Ht</span></div>
                <div className="statistics-stat"><strong>{player.avgBlockHeight}</strong><span>Blk Ht</span></div>
              </div>
            </div>
          ))
        ) : (
          <div style={{ textAlign: 'center', color: '#64748b', padding: '2rem' }}>
            {csvReady ? "No player data available." : "Generating statistics..."}
          </div>
        )}
        
        {teamSetDistribution.length > 0 && (
          <div className="statistics-card">
            <div className="statistics-label" style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: '8px', color: '#1e293b' }}>Set Distribution</div>
            {teamSetDistribution.map((teamData, idx) => (
              <div key={idx} style={{ marginBottom: '4px' }}>
                <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>{teamData.Team}</div>
                <div className="statistics-stats-row">
                  <div className="statistics-stat"><strong>{teamData['Position 2']}</strong><span>Pos 2</span></div>
                  <div className="statistics-stat"><strong>{teamData['Position 3']}</strong><span>Pos 3</span></div>
                  <div className="statistics-stat"><strong>{teamData['Position 4']}</strong><span>Pos 4</span></div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
      
      <div className="statistics-btn-container" style={{ flexGrow: 1 }}>
        <button
          className="statistics-view-csv-btn"
          title="View full statistics CSV"
          onClick={handleOpenModal}
          disabled={!csvData}
        >
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight: '0.5rem' }}>
            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
            <circle cx="12" cy="12" r="3"></circle>
          </svg>
          <span className="statistics-btn-label">View CSV</span>
        </button>
        <button 
          className="statistics-export-btn" 
          title="Download full statistics as CSV" 
          onClick={onExport} 
          disabled={!csvReady}
        >
          <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: '100%' }}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight: '0.5rem' }}>
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="7 10 12 15 17 10"/>
              <line x1="12" y1="15" x2="12" y2="3"/>
            </svg>
            <span className="statistics-btn-label">Export CSV</span>
          </span>
        </button>
      </div>
      {isModalOpen && csvData && (
        <CsvModal 
          csvData={csvData} 
          onClose={handleCloseModal} 
        />
      )}
    </div>
  );
};