import React, { useState } from 'react';
import { Toolbar } from './Toolbar';
import { StatisticsPanel } from './StatisticsPanel';
import { CameraSettings } from './CameraSettings';
import { ChatAgentService } from './ChatAgentService';

interface SubSidebarProps {
  minimized: boolean;
  onClose: () => void;
  selected: string | null;
  setMinimized: React.Dispatch<React.SetStateAction<boolean>>;
  toolbarToggles: {
    courtTracking: boolean;
    ballTracking: boolean;
    netView: boolean;
    points: boolean;
    ballHeight: boolean;
    playerTracking: boolean;
    events: boolean;
    netViewWindowSize: number;
    onToggle: (toggle: 'courtTracking' | 'ballTracking' | 'playerTracking' | 'netView' | 'points' | 'ballHeight' | 'events') => void;
    onNetViewWindowSizeChange: (value: number) => void;
  };
  cameraSettings: any;
  onCameraSettingsChange: (key: string, value: number) => void;
  onApplyCameraSettings: () => void;
  onExportStats: () => void;
  csvReady: boolean;
  csvData: string | null;
  playerNames: Record<string, string>;
  onPlayerNameChange: (playerId: string, newName: string) => void;
}

export const SubSidebar: React.FC<SubSidebarProps> = ({
  minimized,
  onClose,
  selected,
  setMinimized,
  toolbarToggles,
  cameraSettings,
  onCameraSettingsChange,
  onApplyCameraSettings,
  onExportStats,
  csvReady,
  csvData,
  playerNames,
  onPlayerNameChange,
}) => {
  const [hidingContent, setHidingContent] = useState(false);

  // Set sidebar width to 500px when agent submenu is selected
  React.useEffect(() => {
    if (selected === 'agent') {
      document.documentElement.style.setProperty('--sidebar-width', `500px`);
    } else {
      document.documentElement.style.setProperty('--sidebar-width', `320px`);
    }
    return () => {
      document.documentElement.style.setProperty('--sidebar-width', `320px`);
    };
  }, [selected]);

  const handleClose = () => {
    setHidingContent(true);
    setTimeout(() => {
      setMinimized(true);
      setTimeout(() => {
        onClose();
        setHidingContent(false);
      }, 350);
    }, 100);
  };

  return (
    <div className={`subsidebar ${minimized ? 'minimized' : ''}`} style={{ width: `var(--sidebar-width)` }}> 
      {!hidingContent && (
        <div className="subsidebar-header">
          <span className="subsidebar-header-title">
            {selected === 'view' ? 'View' : selected ? selected.charAt(0).toUpperCase() + selected.slice(1) : ''}
          </span>
          <button
            onClick={handleClose}
            className="subsidebar-close-btn"
            title="Minimize sidebar"
          >
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 15L6 9L12 3" stroke="#222" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
        </div>
      )}
      {!hidingContent && (
        <div className="subsidebar-content">
          {selected === 'view' ? (
            <Toolbar toggles={toolbarToggles} />
          ) : selected === 'stats' ? (
            <StatisticsPanel
              csvData={csvData}
              onExport={onExportStats}
              csvReady={csvReady}
              playerNames={playerNames}
              onPlayerNameChange={onPlayerNameChange}
            />
          ) : selected === 'camera' ? (
            <CameraSettings values={{
              cameraHeight: cameraSettings.cameraHeight,
              focalLength: cameraSettings.focalLength,
              ballHeightCalibration: cameraSettings.ballHeightCalibration,
              ballSideCalibration: cameraSettings.ballSideCalibration,
              groundPlaneOffset: cameraSettings.groundPlaneOffset,
              onChange: onCameraSettingsChange,
            }} onApply={onApplyCameraSettings} />
          ) : selected === 'agent' ? (
            <ChatAgentService csvData={csvData} />
          ) : (
            <div style={{ color: '#888', textAlign: 'center', marginTop: '2rem' }}>
              Placeholder for {selected} tools
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// SubSidebar no longer manages or renders the stats point cards.
