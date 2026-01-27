import React, { useState, useEffect } from 'react';
import { apiFetch } from '../api';
import { useNavigate } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { SubSidebar } from './SubSidebar';
import { generateStatsCsv } from '../statistics';

import { Navbar } from './Navbar';
import { VideoPlayer } from './VideoPlayer';
import { EventViewer } from './EventViewer';
import type { VolleyballEvent } from './EventViewer';
import loadingGif from '../assets/loading.gif';

interface DashboardProps {
  file: File;
}

/* ==========================================================================
   HELPER: Draggable Divider
   ========================================================================== */
const DraggableDivider: React.FC<{
  direction: 'vertical' | 'horizontal';
  onDrag: (delta: number) => void;
}> = ({ direction, onDrag }) => {
  const draggingRef = React.useRef(false);

  React.useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (draggingRef.current) {
        onDrag(direction === 'vertical' ? e.movementX : e.movementY);
      }
    };
    const handleMouseUp = () => {
      draggingRef.current = false;
      document.body.style.cursor = '';
    };
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [direction, onDrag]);

  return (
    <div
      className={direction === 'vertical' ? 'draggable-divider-vertical' : 'draggable-divider-horizontal'}
      onMouseDown={() => {
        draggingRef.current = true;
        document.body.style.cursor = direction === 'vertical' ? 'col-resize' : 'row-resize';
      }}
    />
  );
};

const downloadCsv = (csvData: string, fileName: string = 'tobio-stats-export.csv') => {
    const blob = new Blob([csvData], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    if (link.download !== undefined) {
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', fileName);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
};


/* ==========================================================================
   MAIN COMPONENT
   ========================================================================== */
export const Dashboard: React.FC<DashboardProps> = ({ file }) => {
  const navigate = useNavigate();

  // --- Layout State ---
  const [sidebarCollapsed, setSidebarCollapsed] = useState(true);
  const [subSidebarMinimized, setSubSidebarMinimized] = useState(true);
  const [subSidebarSelected, setSubSidebarSelected] = useState<string | null>(null);
  const [rightWidth, setRightWidth] = useState(280);
  const [bottomHeight, setBottomHeight] = useState(180);

  // --- Video & Processing State ---
  const [videoUrl, setVideoUrl] = useState('');
  const [videoMetadata, setVideoMetadata] = useState<any>(null); 
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlayerPlaying, setIsPlayerPlaying] = useState(false);
  
  // Phase 1: Court Lines
  const [editableCourtCorners, setEditableCourtCorners] = useState<[number, number][] | null>(null);
  const [courtLinesConfirmed, setCourtLinesConfirmed] = useState(false);
  
  // Phase 2: Analysis Data
  const [ballTrackingData, setBallTrackingData] = useState<any>(null);
  const [actionDetections, setActionDetections] = useState<any>(null);
  const [, setServeEvents] = useState<any>(null);
  const [playerTracks, setPlayerTracks] = useState<any>(null);
  const [ball3dPositions, setBall3dPositions] = useState<any>(null);
  const [volleyballEvents, setVolleyballEvents] = useState<VolleyballEvent[] | null>(null);
  const [points, setPoints] = useState<{ team: 0 | 1, time: number }[]>([]);
  const [csvData, setCsvData] = useState<string | null>(null);
  const [playerNames, setPlayerNames] = useState<Record<string, string>>({});

  const [cameraSettings, setCameraSettings] = useState({
    cameraHeight: 3.5,
    focalLength: 2.0,
    ballHeightCalibration: 2.0,
    ballSideCalibration: 2.0,
    groundPlaneOffset: 3.5,
  });

  // --- UI Toggles ---
  const [showCourtTracking, setShowCourtTracking] = useState(true);
  const [showBallTracking, setShowBallTracking] = useState(true);
  const [showNetView, setShowNetView] = useState(false);
  const [showPoints, setShowPoints] = useState(true);
  const [showBallHeight, setShowBallHeight] = useState(true);
  const [showPlayerTracking, setShowPlayerTracking] = useState(true);
  const [showEvents, setShowEvents] = useState(true);
  const [netViewWindowSize, setNetViewWindowSize] = useState(300);

  // --- Loading / Error State ---
  const [loading, setLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState<string>('');
  const [progress, setProgress] = useState(0);
  const [, setError] = useState<string | null>(null);

  // --- External Seek ---
  const [externalSeekSeconds, setExternalSeekSeconds] = useState<number | null>(null);

  useEffect(() => {
    let intervalId: number;

    if (loading) {
      intervalId = window.setInterval(() => {
        setProgress((prev) => {
          if (prev >= 100) return 100;
          if (prev < 90) {
            const diff = 90 - prev;
            const step = Math.max(0.2, diff * 0.05); 
            return prev + step;
          }
          return prev;
        });
      }, 200);
    }

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [loading]);

  // 1. Initialize Video URL and Basic Metadata
  useEffect(() => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setVideoUrl(url);

    const tempVideo = document.createElement('video');
    tempVideo.preload = 'metadata';
    tempVideo.src = url;
    tempVideo.onloadedmetadata = () => {
      setVideoMetadata({
        fps: 30, 
        width: tempVideo.videoWidth,
        height: tempVideo.videoHeight,
        total_frames: 0
      });
    };

    return () => URL.revokeObjectURL(url);
  }, [file]);

  // 2. Step 1: Get Initial Court Lines
  useEffect(() => {
    if (!file) return;

    const fetchCourtLines = async () => {
      setLoading(true);
      setProgress(0);
      setLoadingMessage('Detecting court lines...');
      setError(null);

      try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await apiFetch('/process-court-lines/', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) throw new Error('Failed to process court lines');
        
        const data = await response.json();
        
        if (data.court_corners) {
            setEditableCourtCorners(data.court_corners);
        }
        
        setProgress(100);
      } catch (err: any) {
        setError(err.message);
        console.error("Error fetching lines:", err);
        setLoading(false);
      } finally {
        setTimeout(() => {
          setLoading(false); 
        }, 500);
      }
    };

    fetchCourtLines();
  }, [file]);

  // 3. Step 2: Process Full Video
  useEffect(() => {
    if (!file || !courtLinesConfirmed || !editableCourtCorners) return;

    const sendVideoWithCourtLines = async () => {
      setLoading(true);
      setProgress(0);
      setLoadingMessage('Analyzing gameplay...');
      setError(null);

      try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('court_corners', JSON.stringify(editableCourtCorners));
        formData.append('camera_height', String(cameraSettings.cameraHeight));
        formData.append('focal_length', String(cameraSettings.focalLength));
        formData.append('ball_height_calibration', String(cameraSettings.ballHeightCalibration));
        formData.append('ball_side_calibration', String(cameraSettings.ballSideCalibration));
        formData.append('ground_plane_offset', String(cameraSettings.groundPlaneOffset));

        const response = await apiFetch('/process-video/', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) throw new Error('Failed to process video');
        
        const data = await response.json();
        
        if (data.serve_events && Array.isArray(data.serve_events)) {
          const mappedPoints = data.serve_events.slice(1).map((serve: any) => ({
            team: serve.serving_team,
            time: serve.timestamp
          }));
          setPoints(mappedPoints);
        }

        setBallTrackingData(data.ball_detections);
        setActionDetections(data.action_detections);
        setServeEvents(data.serve_events);
        setPlayerTracks(data.player_tracks);
        setBall3dPositions(data.ball_3d_positions);
        setVolleyballEvents(data.volleyball_events);
        
        if (data.video_metadata) {
          setVideoMetadata(data.video_metadata);
        }

        setShowNetView(true);
        setProgress(100);
      } catch (err: any) {
        setError(err.message);
        console.error("Error processing video:", err);
        setLoading(false);
      } finally {
        setTimeout(() => {
          setLoading(false);
        }, 500);
      }
    };

    sendVideoWithCourtLines();
  }, [courtLinesConfirmed, file, cameraSettings]);

  // 4. Generate Statistics CSV
  useEffect(() => {
    if (volleyballEvents) {
      console.log('Generating statistics CSV...');
      const generatedCsv = generateStatsCsv(volleyballEvents, playerNames);
      setCsvData(generatedCsv);
    }
  }, [volleyballEvents, playerNames]);

  // --- Handlers ---
  const handleExit = () => navigate('/');
  
  const handleSeek = (time: number) => {
    setExternalSeekSeconds(time);
    setTimeout(() => setExternalSeekSeconds(null), 250);
  };

  const handleSidebarButtonClick = (key: string) => {
    setSidebarCollapsed(true);
    setSubSidebarSelected(key);
    setSubSidebarMinimized(false);
  };

  const handleCloseSubSidebar = () => {
    setSubSidebarMinimized(true);
    setSubSidebarSelected(null);
  };

  const handleCameraSettingsChange = (key: string, value: number) => {
    setCameraSettings(prev => ({ ...prev, [key]: value }));
  };

  const handleApplyCameraSettings = async () => {
    if (!file || !editableCourtCorners) return;
    setLoading(true);
    setProgress(0);
    setLoadingMessage('Applying camera settings...');
    setError(null);
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('court_corners', JSON.stringify(editableCourtCorners));
      formData.append('camera_height', String(cameraSettings.cameraHeight));
      formData.append('focal_length', String(cameraSettings.focalLength));
      formData.append('ball_height_calibration', String(cameraSettings.ballHeightCalibration));
      formData.append('ball_side_calibration', String(cameraSettings.ballSideCalibration));
      formData.append('ground_plane_offset', String(cameraSettings.groundPlaneOffset));

      const response = await apiFetch('/process-video/', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Failed to process video');
      const data = await response.json();
      setBallTrackingData(data.ball_detections);
      setActionDetections(data.action_detections);
      setServeEvents(data.serve_events);
      setPlayerTracks(data.player_tracks);
      setBall3dPositions(data.ball_3d_positions);
      setVolleyballEvents(data.volleyball_events);
      if (data.video_metadata) {
        setVideoMetadata(data.video_metadata);
      }
      setShowNetView(true);
      setProgress(100);
    } catch (err: any) {
      setError(err.message);
      console.error('Error processing video:', err);
      setLoading(false);
    } finally {
      setTimeout(() => {
        setLoading(false);
      }, 500);
    }
  };

  const handleEventQualityChange = (eventId: string, quality: number | null) => {
    setVolleyballEvents(prevEvents => {
      if (!prevEvents) return null;
      return prevEvents.map((event, index) => {
        const currentEventId = `evt-${event.start_frame}-${index}`;
        if (currentEventId === eventId) {
          return { ...event, quality: quality ?? undefined };
        }
        return event;
      });
    });
  };

  const handlePlayerNameChange = (playerId: string, newName: string) => {
    setPlayerNames(prevNames => ({
      ...prevNames,
      [playerId]: newName,
    }));
  };

  const handleExportStats = () => {
    if (csvData) {
      downloadCsv(csvData);
    } else {
      alert('Statistics data is not yet available.');
    }
  };

  return (
    <div className="dashboard-layout">
      <Navbar onExit={handleExit} />
      
      <div className="dashboard-content-wrapper">
        <div className={`sidebar-wrapper${sidebarCollapsed && subSidebarMinimized ? ' minimized' : ''}`}> 
          <Sidebar
            onNavigate={key => handleSidebarButtonClick(key)}
            collapsed={sidebarCollapsed}
            setCollapsed={setSidebarCollapsed}
            selected={subSidebarSelected}
          />
          <SubSidebar
            minimized={subSidebarMinimized}
            onClose={handleCloseSubSidebar}
            selected={subSidebarSelected}
            setMinimized={setSubSidebarMinimized}
            toolbarToggles={{
              courtTracking: showCourtTracking,
              ballTracking: showBallTracking,
              points: showPoints,
              ballHeight: showBallHeight,
              netView: showNetView,
              playerTracking: showPlayerTracking,
              events: showEvents,
              netViewWindowSize,
              onToggle: (toggle) => {
                if (toggle === 'courtTracking') setShowCourtTracking(v => !v);
                if (toggle === 'ballTracking') setShowBallTracking(v => !v);
                if (toggle === 'points') setShowPoints(v => !v);
                if (toggle === 'ballHeight') setShowBallHeight(v => !v);
                if (toggle === 'netView') setShowNetView(v => !v);
                if (toggle === 'playerTracking') setShowPlayerTracking(v => !v);
                if (toggle === 'events') setShowEvents(v => !v);
              },
              onNetViewWindowSizeChange: setNetViewWindowSize,
            }}
            cameraSettings={cameraSettings}
            onCameraSettingsChange={handleCameraSettingsChange}
            onApplyCameraSettings={handleApplyCameraSettings}
            onExportStats={handleExportStats}
            csvReady={!!csvData}
            csvData={csvData}
            playerNames={playerNames}
            onPlayerNameChange={handlePlayerNameChange}
          />
        </div>

        <main className="dashboard-main-content">
          <div className="dashboard-workspace">
            <div className="dashboard-grid" style={{ position: 'relative' }}>
              
              <div className="dashboard-video-viewer">
                {loading ? (
                  <div className="loading-indicator" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                    <img src={loadingGif} alt="Loading..." style={{ width: '300px' }} />
                    <p style={{ marginBottom: '12px', fontWeight: 500, textAlign: 'center' }}>{loadingMessage}</p>
                    <div className="progress-bar-container" style={{ width: '300px', height: '8px', background: '#ffffffff', borderRadius: '4px', overflow: 'hidden', display: 'flex', justifyContent: 'center' }}>
                      <div 
                        className="progress-bar" 
                        style={{ 
                            width: `${progress}%`, 
                            height: '100%', 
                            background: '#000000ff',
                            transition: 'width 0.2s ease-out'
                        }}
                      ></div>
                    </div>
                  </div>
                ) : videoUrl && videoMetadata && editableCourtCorners ? (
                  <VideoPlayer
                    videoUrl={videoUrl}
                    metadata={videoMetadata}
                    avgCourtCorners={editableCourtCorners}
                    setAvgCourtCorners={setEditableCourtCorners}
                    courtLinesConfirmed={courtLinesConfirmed}
                    setCourtLinesConfirmed={setCourtLinesConfirmed}
                    onConfirmLines={() => {}}
                    ballTracks={ballTrackingData || []}
                    ball3dPositions={ball3dPositions}
                    actionDetections={actionDetections || []}
                    playerTracks={playerTracks || []}
                    playerNames={playerNames}
                    showCourtTracking={showCourtTracking}
                    showBallTracking={showBallTracking}
                    showNetView={showNetView}
                    showBallHeight={showBallHeight}
                    showPlayerTracking={showPlayerTracking}
                    externalSeekSeconds={externalSeekSeconds}
                    netViewWindowSize={netViewWindowSize}
                    onFrameChange={setCurrentFrame}
                    onPlayStateChange={setIsPlayerPlaying}
                  />
                ) : (
                  <div className="loading-indicator">
                    <p>Initializing...</p>
                  </div>
                )}
              </div>

              {showPoints && points.length > 0 && (
                <div className="dashboard-right-subwindows" style={{ display: 'flex', position: 'relative', height: '100%', alignItems: 'center' }}>
                  <DraggableDivider
                    direction="vertical"
                    onDrag={delta => setRightWidth(w => Math.max(220, Math.min(w - delta, window.innerWidth * 0.5)))}
                  />
                  <div
                    className="dashboard-right-subwindow volleyball-sidebar"
                    style={{ width: rightWidth }}
                  >
                    <div className="points-panel-wrapper">
                      <div className="points-header">
                        <strong>Points ({points.length})</strong>
                      </div>
                      <div className="points-list">
                        {points.map((p, idx) => {
                          const scoreA = points.slice(0, idx + 1).filter(x => x.team === 0).length;
                          const scoreB = points.slice(0, idx + 1).filter(x => x.team === 1).length;
                          const borderColor = p.team === 0 ? '#3b82f6' : '#ef4444';
                          return (
                            <button
                              key={idx}
                              onClick={() => handleSeek(p.time)}
                              title={`Jump to ${Math.floor(p.time/60)}:${Math.floor(p.time%60).toString().padStart(2,'0')}`}
                              className="point-card"
                              style={{ borderLeft: `4px solid ${borderColor}` }}
                            >
                              <div className="point-label">Set Score: {scoreA}-{scoreB}</div>
                              <div className="point-time">{Math.floor(p.time/60)}:{Math.floor(p.time%60).toString().padStart(2,'0')}</div>
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
            {showEvents && volleyballEvents && (
              <div className="dashboard-bottom-subwindows" style={{ display: 'flex', position: 'relative', width: '100%', alignItems: 'center', flexDirection: 'column' }}>
                <DraggableDivider
                  direction="horizontal"
                  onDrag={delta => setBottomHeight(h => Math.max(80, Math.min(h - delta, window.innerHeight * 0.4)))}
                />
                <div
                  className="dashboard-bottom-subwindow"
                  style={{ minHeight: 80, height: bottomHeight, maxHeight: '40vh', position: 'relative', padding: '10px' }}
                >
                  <EventViewer
                    events={volleyballEvents || []}
                    points={points}
                    fps={videoMetadata?.fps || 30}
                    onSeek={handleSeek}
                    currentFrame={currentFrame}
                    isPlaying={isPlayerPlaying}
                    onQualityChange={handleEventQualityChange}
                  />
                </div>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
};