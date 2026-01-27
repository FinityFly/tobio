import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';

/* ==========================================================================
   INTERFACES
   ========================================================================== */

interface VideoMetadata {
  fps: number;
  width: number;
  height: number;
  total_frames?: number;
}

interface ActionEvent {
  action: string;
  start_frame: number;
  end_frame: number;
  box?: [number, number, number, number]; // [x1, y1, x2, y2]
}

interface PlayerDetection {
  player_id: number;
  box: [number, number, number, number]; // [x1, y1, x2, y2]
  confidence?: number;
}

interface VideoPlayerProps {
  videoUrl: string;
  metadata: VideoMetadata;
  // Data Props
  avgCourtCorners: [number, number][];
  setAvgCourtCorners: React.Dispatch<React.SetStateAction<[number, number][] | null>>;
  courtLinesConfirmed: boolean;
  setCourtLinesConfirmed: React.Dispatch<React.SetStateAction<boolean>>;
  ballTracks?: (number[] | null)[];
  ball3dPositions?: (number[] | null)[];
  actionDetections?: ActionEvent[]; 
  playerTracks?: Record<string, PlayerDetection[]>;
  playerNames: Record<string, string>;
  // Toggles
  showCourtTracking?: boolean;
  showBallTracking?: boolean;
  showNetView?: boolean;
  showBallHeight?: boolean;
  showPlayerTracking?: boolean;
  // External control
  externalSeekSeconds?: number | null;
  netViewWindowSize?: number;
  onConfirmLines?: () => void;
  onFrameChange?: (frame: number) => void;
  onPlayStateChange?: (isPlaying: boolean) => void;
}

/* ==========================================================================
   CONSTANTS & NET VIEW CONFIGURATION
   ========================================================================== */
const NET_CONFIG = {
  SCALE: 40, // Pixels per meter
  COURT_WIDTH: 9.0,
  NET_HEIGHT: 2.43,
  ANTENNA_HEIGHT: 0.8,
  BAND_HEIGHT: 0.07,
  POST_OFFSET: 0.3048,
  POST_WIDTH: 0.15,
};

const m2px = (m: number) => m * NET_CONFIG.SCALE;

const NET_MATH = (() => {
  const totalNetWidth = NET_CONFIG.COURT_WIDTH + 2 * (NET_CONFIG.POST_OFFSET + NET_CONFIG.POST_WIDTH) + 2.0;
  const totalNetHeight = 7.0; 
  
  const vbWidth = totalNetWidth * NET_CONFIG.SCALE;
  const vbHeight = totalNetHeight * NET_CONFIG.SCALE;
  
  const groundY = vbHeight - (0.5 * NET_CONFIG.SCALE);
  const courtStartX = (vbWidth - (NET_CONFIG.COURT_WIDTH * NET_CONFIG.SCALE)) / 2;

  return {
    vbWidth,
    vbHeight,
    groundY,
    courtStartX,
    netTopY: groundY - m2px(NET_CONFIG.NET_HEIGHT),
    antennaTopY: groundY - m2px(NET_CONFIG.NET_HEIGHT) - m2px(NET_CONFIG.ANTENNA_HEIGHT),
    leftPostX: courtStartX - m2px(NET_CONFIG.POST_OFFSET) - m2px(NET_CONFIG.POST_WIDTH),
    rightPostX: courtStartX + m2px(NET_CONFIG.COURT_WIDTH) + m2px(NET_CONFIG.POST_OFFSET),
  };
})();

/* ==========================================================================
   HELPER UTILITIES
   ========================================================================== */
const lerp = (start: number, end: number, t: number) => {
  return start + (end - start) * t;
};

const drawBallBox = (ctx: CanvasRenderingContext2D, bbox: [number, number, number, number], scaleX: number, scaleY: number) => {
  const [x1, y1, x2, y2] = bbox;
  ctx.strokeStyle = '#facc15'; 
  ctx.lineWidth = 3;
  ctx.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);
};

const drawActionBox = (ctx: CanvasRenderingContext2D, action: string, bbox: [number, number, number, number], scaleX: number, scaleY: number) => {
  const [x1, y1, x2, y2] = bbox;
  
  ctx.strokeStyle = '#f97316'; 
  ctx.lineWidth = 4;
  ctx.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);

  ctx.fillStyle = '#f97316';
  const fontSize = 16;
  ctx.font = `bold ${fontSize}px Inter, sans-serif`;
  const textWidth = ctx.measureText(action.toUpperCase()).width;
  const padding = 6;
  const labelX = x1 * scaleX;
  const labelY = (y1 * scaleY) - (fontSize + padding * 2);

  ctx.fillRect(labelX, labelY, textWidth + (padding * 2), fontSize + (padding * 2));
  ctx.fillStyle = '#ffffff';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  ctx.fillText(action.toUpperCase(), labelX + padding, labelY + (fontSize + padding * 2) / 2 + 1);
};

const drawCourtPolygon = (ctx: CanvasRenderingContext2D, corners: [number, number][], scaleX: number, scaleY: number) => {
  if (corners.length !== 4) return;
  ctx.strokeStyle = 'rgba(45, 212, 191, 0.95)'; // Teal
  ctx.lineWidth = 4;
  ctx.beginPath();
  corners.forEach(([x, y], i) => {
    if (i === 0) ctx.moveTo(x * scaleX, y * scaleY);
    else ctx.lineTo(x * scaleX, y * scaleY);
  });
  ctx.closePath();
  ctx.stroke();
  
  // Note: We removed the canvas-drawn handles here because we use DOM handles now
};

const drawPlayerBox = (ctx: CanvasRenderingContext2D, bbox: number[], id: number, playerNames: Record<string, string>, scaleX: number, scaleY: number) => {
  const [x1, y1, x2, y2] = bbox;
  const drawX = x1 * scaleX;
  const drawY = y1 * scaleY;
  const playerName = playerNames[id] || `P${id}`;
  ctx.strokeStyle = '#22d3ee';
  ctx.lineWidth = 2;
  ctx.strokeRect(drawX, drawY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);
  ctx.fillStyle = '#22d3ee';
  ctx.font = 'bold 14px sans-serif';
  ctx.fillText(playerName, drawX, drawY - 5);
};

const getCanvasScale = (canvas: HTMLCanvasElement, metadata: VideoMetadata) => {
  return {
    scaleX: canvas.width / metadata.width,
    scaleY: canvas.height / metadata.height,
  };
};

const formatTime = (timeInSeconds: number) => {
  if (isNaN(timeInSeconds)) return '0:00';
  const minutes = Math.floor(timeInSeconds / 60);
  const seconds = Math.floor(timeInSeconds % 60).toString().padStart(2, '0');
  return `${minutes}:${seconds}`;
};

/* ==========================================================================
   MAIN COMPONENT
   ========================================================================== */
export const VideoPlayer: React.FC<VideoPlayerProps> = ({ 
  // --- Debug: Ball 3D Position at Current Frame ---
  // (Must be after hooks, before return)
  videoUrl,
  metadata,
  avgCourtCorners,
  setAvgCourtCorners,
  courtLinesConfirmed,
  setCourtLinesConfirmed,
  ballTracks,
  ball3dPositions,
  actionDetections,
  playerTracks,
  playerNames,
  showCourtTracking = true,
  showBallTracking = true,
  showPlayerTracking = true,
  showNetView = true,
  showBallHeight = true,
  externalSeekSeconds,
  netViewWindowSize = 320,
  onConfirmLines,
  onFrameChange,
  onPlayStateChange,
}) => {
  // --- Refs ---
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null); // New Container for perfect alignment
  
  const netBallRef = useRef<SVGCircleElement>(null);
  const netInfoRef = useRef<HTMLDivElement>(null);
  const netBallLabelRef = useRef<SVGTextElement>(null);
  const netBallLabelRectRef = useRef<SVGRectElement>(null);

  // --- State ---
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [draggingIdx, setDraggingIdx] = useState<number | null>(null);

  // --- Data Memos ---
  const ballTracksMap = useMemo(() => 
    ballTracks 
      ? new Map(ballTracks.map((bbox, frame) => [frame, bbox]).filter((entry): entry is [number, number[]] => entry[1] != null)) 
      : new Map(),
    [ballTracks]
  );

  const actionFrameMap = useMemo(() => {
    const map = new Map<number, ActionEvent>();
    if (!actionDetections) return map;
    actionDetections.forEach((event) => {
      for (let f = event.start_frame; f <= event.end_frame; f++) {
        map.set(f, event);
      }
    });
    return map;
  }, [actionDetections]);

  const playerTracksMap = useMemo(() => {
    if (!playerTracks) return new Map<number, PlayerDetection[]>();
    const map = new Map<number, PlayerDetection[]>();
    Object.keys(playerTracks).forEach((frameKey) => {
      const frameId = parseInt(frameKey, 10);
      if (!isNaN(frameId)) map.set(frameId, playerTracks[frameKey]);
    });
    return map;
  }, [playerTracks]);


  // --- Render Loop ---
  const draw = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const overlay = overlayRef.current;
    
    if (!video || !canvas || !metadata || !canvas.parentElement) return; 

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // A. Resize Logic: Fit Canvas & Overlay perfectly inside the container
    const { clientWidth: containerWidth, clientHeight: containerHeight } = canvas.parentElement;
    const aspectRatio = metadata.width / metadata.height;
    if (!aspectRatio || isNaN(aspectRatio)) return;

    // Calculate dimensions to fit ("letterbox" logic)
    let renderWidth = containerWidth;
    let renderHeight = containerWidth / aspectRatio;
    
    if (renderHeight > containerHeight) {
      renderHeight = containerHeight;
      renderWidth = containerHeight * aspectRatio;
    }
    
    // 1. Update Canvas Buffer Size
    canvas.width = renderWidth;
    canvas.height = renderHeight;
    
    // 2. Update Overlay/Canvas CSS Size (Centering)
    // We update this imperatively to avoid React render cycles on every frame
    const topOffset = (containerHeight - renderHeight) / 2;
    const leftOffset = (containerWidth - renderWidth) / 2;
    
    const styleString = `width: ${renderWidth}px; height: ${renderHeight}px; top: ${topOffset}px; left: ${leftOffset}px; position: absolute;`;
    
    if (overlay) overlay.style.cssText = styleString;
    canvas.style.cssText = styleString;

    // B. Draw Content
    if (video.readyState >= 2) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    } else {
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    
    const { scaleX, scaleY } = getCanvasScale(canvas, metadata);
    const exactFrame = video.currentTime * metadata.fps;
    const currentFrameInt = Math.floor(exactFrame);

    // C. Overlays
    if (showBallTracking) {
      const ballBbox = ballTracksMap.get(currentFrameInt);
      if (ballBbox) drawBallBox(ctx, ballBbox as [number, number, number, number], scaleX, scaleY);
    }

    // Court Lines (Polygon Only - Handles are now DOM elements)
    if (showCourtTracking && avgCourtCorners?.length === 4) {
      const isNormalized = avgCourtCorners[0][0] < 2;
      let drawCorners = avgCourtCorners;
      if (isNormalized) {
         drawCorners = avgCourtCorners.map(([x, y]) => [x * metadata.width, y * metadata.height]);
      }
      drawCourtPolygon(ctx, drawCorners, scaleX, scaleY);
    }

    if (showPlayerTracking) {
      const playersInFrame = playerTracksMap.get(currentFrameInt);
      if (playersInFrame) {
        playersInFrame.forEach(player => drawPlayerBox(ctx, player.box, player.player_id, playerNames, scaleX, scaleY));
      }
    }

    const currentAction = actionFrameMap.get(currentFrameInt);
    if (currentAction && currentAction.box) {
      drawActionBox(ctx, currentAction.action, currentAction.box, scaleX, scaleY);
    } else if (currentAction) {
      ctx.save();
      ctx.font = 'bold 24px Inter, sans-serif';
      ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
      ctx.shadowColor = 'rgba(0,0,0,0.5)';
      ctx.shadowBlur = 4;
      ctx.textAlign = 'center';
      ctx.fillText(currentAction.action.toUpperCase(), canvas.width / 2, 50);
      ctx.restore();
    }

    // 3D Net View Updates
    // console.log("netBallRef.current:", netBallRef.current);
    if (ball3dPositions && netBallRef.current) {
        const alpha = exactFrame - currentFrameInt;
        const posCurrent = ball3dPositions[currentFrameInt];
        const posNext = (currentFrameInt + 1 < ball3dPositions.length) ? ball3dPositions[currentFrameInt + 1] : posCurrent;

        if (posCurrent && posCurrent.length >= 3 && posCurrent[0] !== null) {
            let [x, y, z] = posCurrent;
            if (posNext && posNext.length >= 3 && posNext[0] !== null) {
                x = lerp(x, posNext[0], alpha);
                y = lerp(y, posNext[1], alpha);
                z = lerp(z, posNext[2], alpha);
            }

            const cx = NET_MATH.courtStartX + m2px(x);
            const cy = NET_MATH.groundY - m2px(z);
            const depthFactor = Math.max(0.0, Math.min(1.0, y / 18.0)); 
            const radius = 10 - (depthFactor * 5); 


            netBallRef.current.setAttribute('cx', cx.toString());
            netBallRef.current.setAttribute('cy', cy.toString());
            netBallRef.current.setAttribute('r', radius.toString());
            netBallRef.current.style.opacity = '1';

            if (netInfoRef.current) {
              netInfoRef.current.textContent = `H: ${z.toFixed(2)}m`;
              netInfoRef.current.style.opacity = showBallHeight ? '1' : '0';
            }

            try {
              if (netBallLabelRef.current && netBallLabelRectRef.current) {
                const labelText = `H: ${z.toFixed(2)}m`;
                netBallLabelRef.current.textContent = labelText;
                
                const bbox = netBallLabelRef.current.getBBox ? netBallLabelRef.current.getBBox() : { width: 0, height: 0 };
                const pad = 8;
                const rectWidth = bbox.width + pad * 2;
                const rectHeight = bbox.height + pad * 2;
                const rectX = cx - rectWidth / 2;
                const rectY = cy - radius - rectHeight - 6; 

                netBallLabelRectRef.current.setAttribute('x', rectX.toString());
                netBallLabelRectRef.current.setAttribute('y', rectY.toString());
                netBallLabelRectRef.current.setAttribute('width', rectWidth.toString());
                netBallLabelRectRef.current.setAttribute('height', rectHeight.toString());
                netBallLabelRectRef.current.style.opacity = showBallHeight ? '1' : '0';

                netBallLabelRef.current.setAttribute('x', cx.toString());
                netBallLabelRef.current.setAttribute('y', (rectY + rectHeight / 2).toString());
                netBallLabelRef.current.style.opacity = showBallHeight ? '1' : '0';
              }
            } catch (e) {
              // ignore
            }
        } else {
             if (netBallRef.current) netBallRef.current.style.opacity = '0';
             if (netBallLabelRef.current) netBallLabelRef.current.style.opacity = '0';
             if (netBallLabelRectRef.current) netBallLabelRectRef.current.style.opacity = '0';
             if (netInfoRef.current) netInfoRef.current.style.opacity = '0';
        }
    }
  }, [metadata, ballTracksMap, avgCourtCorners, actionFrameMap, playerTracksMap, showCourtTracking, showBallTracking, showPlayerTracking, ball3dPositions, showBallHeight]);

  // --- Global Drag Handler (Mouse Move / Up) ---
  // We attach this to Window so you can drag fast outside the handle
  useEffect(() => {
    const handleGlobalMove = (e: MouseEvent) => {
      if (draggingIdx === null || !overlayRef.current) return;
      
      const rect = overlayRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      
      // Convert Overlay Position -> Video Coordinate
      // Normalized: x / rect.width
      // Absolute: (x / rect.width) * metadata.width
      
      const isNormalized = avgCourtCorners[0][0] < 2;
      
      let newX = x / rect.width;
      let newY = y / rect.height;
      
      // Clamp to 0-1
      newX = Math.max(0, Math.min(1, newX));
      newY = Math.max(0, Math.min(1, newY));
      
      if (!isNormalized) {
        newX *= metadata.width;
        newY *= metadata.height;
      }
      
      setAvgCourtCorners(prev => {
        if (!prev) return null;
        const copy = [...prev];
        copy[draggingIdx] = [newX, newY];
        return copy;
      });
    };

    const handleGlobalUp = () => {
      setDraggingIdx(null);
    };

    if (draggingIdx !== null) {
      window.addEventListener('mousemove', handleGlobalMove);
      window.addEventListener('mouseup', handleGlobalUp);
    }
    return () => {
      window.removeEventListener('mousemove', handleGlobalMove);
      window.removeEventListener('mouseup', handleGlobalUp);
    };
  }, [draggingIdx, avgCourtCorners, metadata, setAvgCourtCorners]);


  // --- Effects ---

useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const handleCanPlay = () => {
      setDuration(video.duration);
      draw();
    };
    const handleTimeUpdate = () => {
      setCurrentTime(video.currentTime);
      if (typeof onFrameChange === 'function' && metadata?.fps) {
        const frame = Math.floor(video.currentTime * metadata.fps);
        onFrameChange(frame);
      }
      draw();
    };
    const handlePlay = () => {
      setIsPlaying(true);
      onPlayStateChange?.(true);
    };
    const handlePause = () => {
      setIsPlaying(false);
      onPlayStateChange?.(false);
    };

    video.addEventListener('canplay', handleCanPlay);
    video.addEventListener('timeupdate', handleTimeUpdate);
    video.addEventListener('seeked', draw);
    video.addEventListener('play', handlePlay);
    video.addEventListener('pause', handlePause);

    return () => {
      video.removeEventListener('canplay', handleCanPlay);
      video.removeEventListener('timeupdate', handleTimeUpdate);
      video.removeEventListener('seeked', draw);
      video.removeEventListener('play', handlePlay);
      video.removeEventListener('pause', handlePause);
    };
  }, [draw, metadata, onFrameChange, onPlayStateChange]);

  useEffect(() => {
    let animationFrameId: number;
    const renderLoop = () => {
      draw();
      animationFrameId = requestAnimationFrame(renderLoop);
    };
    if (metadata) animationFrameId = requestAnimationFrame(renderLoop);
    return () => cancelAnimationFrame(animationFrameId);
  }, [isPlaying, draw, metadata]);

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.volume = volume;
      videoRef.current.muted = isMuted;
    }
  }, [volume, isMuted]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || externalSeekSeconds == null) return;
    if (Math.abs(video.currentTime - externalSeekSeconds) > 0.1) {
      video.currentTime = externalSeekSeconds;
      draw();
    }
  }, [externalSeekSeconds, draw]);

  useEffect(() => { requestAnimationFrame(draw); }, [avgCourtCorners, draw]);

  // --- Controls ---
  const togglePlayPause = async () => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      await video.play();
      setIsPlaying(true);
    } else {
      video.pause();
      setIsPlaying(false);
    }
  };

  const handleTimelineChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    if (videoRef.current) {
      videoRef.current.currentTime = time;
      setCurrentTime(time);
    }
  };

  // Debug value for current frame
  const currentFrameIdx = Math.max(0, Math.floor(currentTime * (metadata?.fps || 30)));
  const ball3dDebug = ball3dPositions && ball3dPositions[currentFrameIdx] ? ball3dPositions[currentFrameIdx] : null;

  return (
    <div className="video-player" style={{ position: 'relative', background: '#000', borderRadius: '12px', overflow: 'hidden' }}>
      {/* TEMP DEBUG: Ball 3D Position at Current Frame */}
      <div style={{
        position: 'absolute',
        top: 8,
        left: 8,
        zIndex: 100,
        background: 'rgba(0,0,0,0.75)',
        color: '#facc15',
        fontSize: '13px',
        fontFamily: 'monospace',
        padding: '6px 12px',
        borderRadius: '6px',
        pointerEvents: 'none',
        minWidth: 120,
        maxWidth: 320,
        whiteSpace: 'pre',
        boxShadow: '0 2px 8px rgba(0,0,0,0.12)'
      }}>
        <span style={{ color: '#fff', fontWeight: 600 }}>ball3d:</span> {ball3dDebug ? JSON.stringify(ball3dDebug.map(v => v !== null ? Number(v).toFixed(6) : null)) : 'null'}
      </div>
      
      {/* Hidden Source Video */}
      <video
        ref={videoRef}
        src={videoUrl}
        style={{ display: 'none' }} // We draw to canvas, so hide this
        muted={isMuted}
        playsInline
        crossOrigin="anonymous"
        preload="auto"
      />
      
      {/* Container for Canvas + Overlay */}
      {/* This wrapper fills the component area */}
      <div style={{ position: 'relative', width: '100%', height: '100%' }}>
          
          {/* 1. Canvas Layer (Draws Video + Lines) */}
          <canvas 
            ref={canvasRef} 
            className="video-canvas" 
            style={{ display: 'block' }} // Positioned by JS
          />

          {/* 2. Interaction Layer (DOM Handles) */}
          {/* This matches the canvas size/position EXACTLY */}
          <div ref={overlayRef} style={{ pointerEvents: 'none' }}>
            {!courtLinesConfirmed && avgCourtCorners?.length === 4 && avgCourtCorners.map((corner, idx) => {
                const isNormalized = avgCourtCorners[0][0] < 2;
                const x = isNormalized ? corner[0] : corner[0] / metadata.width;
                const y = isNormalized ? corner[1] : corner[1] / metadata.height;
                
                return (
                    <div
                        key={idx}
                        onMouseDown={(e) => {
                            e.preventDefault(); // Prevent text selection
                            setDraggingIdx(idx);
                        }}
                        style={{
                            position: 'absolute',
                            left: `${x * 100}%`,
                            top: `${y * 100}%`,
                            width: '24px',
                            height: '24px',
                            marginLeft: '-12px', // Center the handle
                            marginTop: '-12px',
                            backgroundColor: 'rgba(14, 165, 233, 0.8)', // Cyan
                            border: '2px solid white',
                            borderRadius: '50%',
                            cursor: 'grab',
                            pointerEvents: 'auto', // Re-enable clicks
                            zIndex: 40,
                            boxShadow: '0 2px 4px rgba(0,0,0,0.5)',
                            transform: draggingIdx === idx ? 'scale(1.2)' : 'scale(1)',
                            transition: 'transform 0.1s'
                        }}
                    />
                );
            })}
          </div>
      </div>
        
      {/* Confirm Lines Button */}
      {!courtLinesConfirmed && avgCourtCorners?.length === 4 && (
          <button
            style={{
              position: 'absolute',
              top: 60,
              right: 24,
              zIndex: 50,
              padding: '12px 24px',
              background: 'linear-gradient(90deg, #22c55e, #16a34a)',
              color: 'white',
              fontWeight: 700,
              fontSize: '1.1rem',
              borderRadius: '8px',
              border: 'none',
              boxShadow: '0 4px 12px rgba(0,0,0,0.2)',
              cursor: 'pointer',
              pointerEvents: 'auto',
            }}
            onClick={() => {
              setCourtLinesConfirmed(true);
              if (onConfirmLines) onConfirmLines();
            }}
          >
            Confirm Court Lines
          </button>
      )}

        {/* PIP Net View */}
        {showNetView && (
          <div style={{
            position: 'absolute',
            top: '16px',
            right: '16px',
            width: netViewWindowSize ? `${netViewWindowSize}px` : '320px',
            height: netViewWindowSize ? `${Math.round(netViewWindowSize * 0.5625)}px` : '180px',
            backgroundColor: 'rgba(255, 255, 255, 0.95)',
            borderRadius: '12px',
            boxShadow: '0 4px 12px rgba(0,0,0,0.2)',
            overflow: 'hidden',
            zIndex: 20,
            border: '1px solid rgba(0,0,0,0.1)'
          }}>
            <div style={{ position: 'absolute', top: 8, left: 12, zIndex: 2, display: 'flex', gap: '8px', alignItems: 'center'}}>
                <span style={{ fontSize: '10px', fontWeight: 'bold', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                    Live Net View
                </span>
            </div>
            <svg 
                viewBox={`0 0 ${NET_MATH.vbWidth} ${NET_MATH.vbHeight}`} 
                preserveAspectRatio="xMidYMid meet"
                style={{ width: '100%', height: '100%', display: 'block' }}
            >
                {/* ... (Existing SVG Content) ... */}
                <defs>
                    <linearGradient id="pipSky" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#f1f5f9" />
                        <stop offset="100%" stopColor="#e2e8f0" />
                    </linearGradient>
                     <pattern id="pipNetPattern" width="10" height="10" patternUnits="userSpaceOnUse">
                        <path d="M 10 0 L 0 0 0 10" fill="none" stroke="#cbd5e1" strokeWidth="1" />
                     </pattern>
                </defs>

                <rect x="0" y="0" width={NET_MATH.vbWidth} height={NET_MATH.vbHeight} fill="url(#pipSky)" />
                <line x1="0" y1={NET_MATH.groundY} x2={NET_MATH.vbWidth} y2={NET_MATH.groundY} stroke="#94a3b8" strokeWidth="2" />

                <g>
                    <rect x={NET_MATH.leftPostX} y={NET_MATH.netTopY} width={m2px(NET_CONFIG.POST_WIDTH)} height={m2px(NET_CONFIG.NET_HEIGHT)} fill="#475569" rx="2" />
                    <rect x={NET_MATH.rightPostX} y={NET_MATH.netTopY} width={m2px(NET_CONFIG.POST_WIDTH)} height={m2px(NET_CONFIG.NET_HEIGHT)} fill="#475569" rx="2" />
                    <rect x={NET_MATH.leftPostX + m2px(NET_CONFIG.POST_WIDTH)} y={NET_MATH.netTopY + m2px(NET_CONFIG.BAND_HEIGHT)} width={(NET_MATH.rightPostX) - (NET_MATH.leftPostX + m2px(NET_CONFIG.POST_WIDTH))} height={m2px(NET_CONFIG.NET_HEIGHT) - m2px(NET_CONFIG.BAND_HEIGHT) - m2px(1.0)} fill="url(#pipNetPattern)" />
                    <rect x={NET_MATH.leftPostX + m2px(NET_CONFIG.POST_WIDTH)} y={NET_MATH.netTopY} width={(NET_MATH.rightPostX) - (NET_MATH.leftPostX + m2px(NET_CONFIG.POST_WIDTH))} height={m2px(NET_CONFIG.BAND_HEIGHT)} fill="white" stroke="#94a3b8" strokeWidth="1"/>
                    <rect x={NET_MATH.courtStartX - 2} y={NET_MATH.antennaTopY} width={4} height={m2px(NET_CONFIG.ANTENNA_HEIGHT + 1.0)} fill="white"/>
                    <line x1={NET_MATH.courtStartX} y1={NET_MATH.antennaTopY} x2={NET_MATH.courtStartX} y2={NET_MATH.netTopY + m2px(1.0)} stroke="#ef4444" strokeWidth="4" strokeDasharray="10 10" />
                    <rect x={NET_MATH.courtStartX + m2px(NET_CONFIG.COURT_WIDTH) - 2} y={NET_MATH.antennaTopY} width={4} height={m2px(NET_CONFIG.ANTENNA_HEIGHT + 1.0)} fill="white"/>
                    <line x1={NET_MATH.courtStartX + m2px(NET_CONFIG.COURT_WIDTH)} y1={NET_MATH.antennaTopY} x2={NET_MATH.courtStartX + m2px(NET_CONFIG.COURT_WIDTH)} y2={NET_MATH.netTopY + m2px(1.0)} stroke="#ef4444" strokeWidth="4" strokeDasharray="10 10" />
                </g>
                <g>
                  <rect ref={netBallLabelRectRef} fill="rgba(59,130,246,0.1)" rx="8" style={{ opacity: 0 }} />
                  <text ref={netBallLabelRef} fill="#3b82f6" fontFamily="monospace" fontSize={20} fontWeight="bold" textAnchor="middle" dominantBaseline="middle" style={{ opacity: 0 }}>-</text>
                </g>
                <circle ref={netBallRef} fill="#facc15" stroke="#000" strokeWidth="1.5" style={{ opacity: 0, transition: 'none' }} />
            </svg>
          </div>
        )}
      
      <div className="timeline-slider-container">
        <input
          type="range"
          min="0"
          max={duration || 0}
          value={currentTime}
          onChange={handleTimelineChange}
          className="timeline-slider"
          step="0.01"
          style={{ '--progress-percent': `${duration ? (currentTime / duration) * 100 : 0}%` } as React.CSSProperties}
        />
      </div>
      
      <div className="video-controls video-controls-bar">
        <button onClick={togglePlayPause} className="play-pause-btn">
          {isPlaying ? <PauseIcon /> : <PlayIcon />}
        </button>
        <button onClick={() => setIsMuted(!isMuted)} className="mute-btn">
          {isMuted || volume === 0 ? <MutedIcon /> : <VolumeIcon />}
        </button>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={isMuted ? 0 : volume}
          onChange={(e) => setVolume(Number(e.target.value))}
          className="volume-slider"
        />
        <span className="time-display">{formatTime(currentTime)} / {formatTime(duration)}</span>
      </div>
    </div>
  );
};

/* ==========================================================================
   ICONS
   ========================================================================== */
const PlayIcon = () => (
  <svg height="100%" viewBox="0 0 36 36" width="100%"><path fill="currentColor" d="M 12,26 18.5,22 18.5,14 12,10 z M 18.5,22 25,18 25,18 18.5,14 z" /></svg>
);
const PauseIcon = () => (
  <svg height="100%" viewBox="0 0 36 36" width="100%"><path fill="currentColor" d="M 12,26 16,26 16,10 12,10 z M 21,26 25,26 25,10 21,10 z" /></svg>
);
const MutedIcon = () => (
  <svg height="100%" viewBox="0 0 36 36" width="100%"><path fill="currentColor" d="m 21.48,17.98 c 0,-1.77 -1.02,-3.29 -2.5,-4.03 v 2.21 l 2.45,2.45 c .03,-.2 .05,-.41 .05,-.63 z m 2.5,0 c 0,.94 -.2,1.82 -.54,2.64 l 1.51,1.51 c .63,-1.09 1.03,-2.34 1.03,-3.65 0,-4.28 -2.99,-7.86 -7,-8.77 v 2.06 c 2.89,.86 5,3.54 5,6.71 z M 9.27,9 7.86,10.41 12.73,15.27 H 8 v 6 h 4 l 5,5 v -6.73 l 4.25,4.25 c -.67,.52 -1.42,.93 -2.25,1.18 v 2.06 c 1.38,-.31 2.63,-.95 3.69,-1.81 l 2.04,2.04 L 23,22.48 9.27,9 Z m 7.73,0 -2.09,2.09 L 17,13.18 V 9 Z"></path></svg>
);
const VolumeIcon = () => (
  <svg height="100%" viewBox="0 0 36 36" width="100%"><path fill="currentColor" d="M8,21 L12,21 L17,26 L17,10 L12,15 L8,15 L8,21 Z M19,14 L19,22 C20.48,21.32 21.5,19.77 21.5,18 C21.5,16.23 20.48,14.68 19,14 Z M19,11.29 C21.89,12.15 24,14.83 24,18 C24,21.17 21.89,23.85 19,24.71 L19,26.77 C23.01,25.86 26,22.28 26,18 C26,13.72 23.01,10.14 19,9.23 L19,11.29 Z"></path></svg>
);