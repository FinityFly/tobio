import React from 'react';

interface VolleyballCourtProps {
  ball3dPositions?: (number[] | null)[];
  currentFrame: number;
}

// React.memo prevents the dashboard layout re-renders from affecting this component
export const VolleyballCourt: React.FC<VolleyballCourtProps> = React.memo(({ 
  ball3dPositions, 
  currentFrame
}) => {
  // --- Constants ---
  const REAL_COURT_WIDTH = 9.0; 
  const REAL_NET_HEIGHT = 2.43; 
  const REAL_ANTENNA_HEIGHT = 0.8; 
  const NET_BAND_HEIGHT = 0.07; 
  const POST_OFFSET = 0.3048; 
  const POST_WIDTH = 0.15; 

  const SCALE = 50; 
  const m2px = (meters: number) => meters * SCALE;

  // ViewBox Setup
  const contentWidthMeters = REAL_COURT_WIDTH + 2 * (POST_OFFSET + POST_WIDTH);
  const paddingX = 1.0; 
  const totalWidthMeters = contentWidthMeters + (paddingX * 2);
  const totalHeightMeters = 7.0; 

  const viewBoxWidth = m2px(totalWidthMeters);
  const viewBoxHeight = m2px(totalHeightMeters);
  const groundY = viewBoxHeight - m2px(0.5); 
  const courtStartX = (viewBoxWidth - m2px(REAL_COURT_WIDTH)) / 2;

  // --- Ball Logic ---
  const ballPosition = ball3dPositions && ball3dPositions.length > currentFrame 
    ? ball3dPositions[currentFrame] 
    : null;

  const renderBall = () => {
    if (!ballPosition || ballPosition.length < 3) return null;

    const [x, y, z] = ballPosition;

    if (x === null || z === null || isNaN(x) || isNaN(z)) return null;

    const cx = courtStartX + m2px(x);
    const cy = groundY - m2px(z);

    // Depth radius calculation
    const depthFactor = Math.max(0.0, Math.min(1.0, y / 18.0)); 
    const radius = 14 - (depthFactor * 8); 

    return (
      <circle
        cx={cx}
        cy={cy}
        r={radius}
        className="ball-indicator"
        // REMOVED CSS TRANSITION to fix lag
        style={{ transition: 'none' }} 
      />
    );
  };

  // --- Static coords ---
  const netTopY = groundY - m2px(REAL_NET_HEIGHT);
  const antennaTopY = netTopY - m2px(REAL_ANTENNA_HEIGHT);
  const leftPostX = courtStartX - m2px(POST_OFFSET) - m2px(POST_WIDTH);
  const rightPostX = courtStartX + m2px(REAL_COURT_WIDTH) + m2px(POST_OFFSET);

  return (
    <div className="volleyball-court-container">
      <div style={{ position: 'absolute', top: 10, left: 10, fontSize: '10px', color: '#64748b', fontFamily: 'monospace' }}>
        NET VIEW • {ballPosition ? `H: ${ballPosition[2]?.toFixed(2)}m` : ''}
      </div>

      <svg
        className="volleyball-court-svg"
        viewBox={`0 0 ${viewBoxWidth} ${viewBoxHeight}`}
        preserveAspectRatio="xMidYMid meet"
      >
        <defs>
          <linearGradient id="skyGradient" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" className="net-sky-gradient-start" />
            <stop offset="100%" className="net-sky-gradient-end" />
          </linearGradient>
           <pattern id="netPattern" width="10" height="10" patternUnits="userSpaceOnUse">
            <path d="M 10 0 L 0 0 0 10" fill="none" stroke="#cbd5e1" strokeWidth="1" />
          </pattern>
        </defs>

        <rect x="0" y="0" width={viewBoxWidth} height={viewBoxHeight} fill="url(#skyGradient)" />
        <line x1="0" y1={groundY} x2={viewBoxWidth} y2={groundY} stroke="#94a3b8" strokeWidth="2" />

        <g>
          <rect x={leftPostX} y={netTopY} width={m2px(POST_WIDTH)} height={m2px(REAL_NET_HEIGHT)} className="net-pole" rx="3" />
          <rect x={rightPostX} y={netTopY} width={m2px(POST_WIDTH)} height={m2px(REAL_NET_HEIGHT)} className="net-pole" rx="3" />
          <rect x={leftPostX + m2px(POST_WIDTH)} y={netTopY + m2px(NET_BAND_HEIGHT)} width={(rightPostX) - (leftPostX + m2px(POST_WIDTH))} height={m2px(REAL_NET_HEIGHT) - m2px(NET_BAND_HEIGHT) - m2px(1.0)} fill="url(#netPattern)" />
          <rect x={leftPostX + m2px(POST_WIDTH)} y={netTopY} width={(rightPostX) - (leftPostX + m2px(POST_WIDTH))} height={m2px(NET_BAND_HEIGHT)} className="net-tape" />
          
          {/* Antennas */}
          <g>
             <rect x={courtStartX - 2} y={antennaTopY} width={4} height={m2px(REAL_ANTENNA_HEIGHT + 1.0)} className="antenna-white"/>
             <line x1={courtStartX} y1={antennaTopY} x2={courtStartX} y2={netTopY + m2px(1.0)} stroke="#ef4444" strokeWidth="4" strokeDasharray="10 10" />
          </g>
          <g>
             <rect x={courtStartX + m2px(REAL_COURT_WIDTH) - 2} y={antennaTopY} width={4} height={m2px(REAL_ANTENNA_HEIGHT + 1.0)} className="antenna-white"/>
             <line x1={courtStartX + m2px(REAL_COURT_WIDTH)} y1={antennaTopY} x2={courtStartX + m2px(REAL_COURT_WIDTH)} y2={netTopY + m2px(1.0)} stroke="#ef4444" strokeWidth="4" strokeDasharray="10 10" />
          </g>
        </g>

        {renderBall()}
      </svg>
    </div>
  );
});