import React, { useState, useRef, useEffect, useMemo } from 'react';

export interface VolleyballEvent {
  action: string;
  start_frame: number;
  end_frame: number;
  player_id?: number;
  ball_height_m?: number;
  block_height_m?: number;
  box?: number[];
  quality?: number; // Should be 0, 1, 2, or 3
  set_position?: number;
}

interface EventCardData {
  type: 'event';
  id: string;
  frameId: number;
  endFrameId: number;
  timestamp: number;
  player: string;
  action: string;
  extraInfo: string;
  quality?: number;
  details: {
    height?: string;
    duration?: string;
    setPosition?: string;
  };
}

interface PointDividerData {
  type: 'point';
  id: string;
  timestamp: number;
  scoreA: number;
  scoreB: number;
}

type TimelineItem = EventCardData | PointDividerData;

interface EventViewerProps {
  events?: VolleyballEvent[];
  points?: { team: 0 | 1, time: number }[];
  fps?: number;
  onSeek: (time: number) => void;
  currentFrame?: number;
  isPlaying?: boolean;
  onQualityChange: (eventId: string, quality: number | null) => void;
}

const qualityMap = {
  0: { label: 'Error',   color: '#ef4444', bgColor: '#ffcacaff' }, // Red
  1: { label: 'Okay',    color: '#f59e0b', bgColor: '#ffeca1ff' }, // Amber
  2: { label: 'Good',    color: '#84cc16', bgColor: '#e2ffa7ff' }, // Lime
  3: { label: 'Perfect', color: '#10b981', bgColor: '#abffd4ff' }, // Green
};

const PointDivider: React.FC<{ score: string; timestamp: string }> = ({ score, timestamp }) => (
  <div className="point-divider">
    <div className="point-divider-line" />
    <div className="point-divider-labels">
      <span className="point-divider-score">{score}</span>
      <span className="point-divider-timestamp">{timestamp}</span>
    </div>
    <div className="point-divider-line" />
  </div>
);

export const EventViewer: React.FC<EventViewerProps> = ({ events: rawEvents = [], points = [], fps = 30, onSeek, currentFrame, isPlaying = false, onQualityChange }) => {
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const cardRefs = useRef<(HTMLDivElement | null)[]>([]);

  const [isDown, setIsDown] = useState(false);
  const [startX, setStartX] = useState(0);
  const [scrollLeft, setScrollLeft] = useState(0);
  const isDraggingRef = useRef(false);
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null);
  const [popupPosition, setPopupPosition] = useState<{ top: number; left: number } | null>(null);

  const timelineItems: TimelineItem[] = useMemo(() => {
    const eventItems: EventCardData[] = (rawEvents || []).map((event, index) => {
      const formattedAction = event.action.charAt(0).toUpperCase() + event.action.slice(1);
      const time = event.start_frame / (fps || 30);
      const durationSec = ((event.end_frame - event.start_frame) / (fps || 30)).toFixed(2);
      const playerText = event.player_id ? `Player ${event.player_id}` : "Unknown Player";
      let extraInfoText = `${durationSec}s`;
      let heightDetail: string | undefined = undefined;
      let setPositionDetail: string | undefined = undefined;
      
      if (event.action === 'spike' && event.ball_height_m) {
        extraInfoText = `${event.ball_height_m.toFixed(2)}m`;
        heightDetail = `${event.ball_height_m.toFixed(2)}m (Ball Contact)`;
      } else if (event.action === 'block' && event.block_height_m) {
        extraInfoText = `${event.block_height_m.toFixed(2)}m`;
        heightDetail = `${event.block_height_m.toFixed(2)}m (Block Height)`;
      } else if (event.action === 'set' && event.ball_height_m) {
        extraInfoText = `${event.ball_height_m.toFixed(2)}m`;
        heightDetail = `${event.ball_height_m.toFixed(2)}m (Ball Contact)`;
      } else if (event.action === 'set' && event.set_position) {
        extraInfoText = `X: ${event.set_position.toFixed(2)}m`;
        setPositionDetail = `${event.set_position.toFixed(2)}m (Horizontal Position)`;
      }

      return {
        type: 'event',
        id: `evt-${event.start_frame}-${index}`,
        frameId: event.start_frame,
        endFrameId: event.end_frame,
        timestamp: time,
        player: playerText,
        action: formattedAction,
        extraInfo: extraInfoText,
        quality: event.quality,
        details: {
          duration: `${durationSec} seconds`,
          height: heightDetail,
          setPosition: setPositionDetail
        }
      };
    });

    let scoreA = 0;
    let scoreB = 0;
    const pointItems: PointDividerData[] = (points || []).map((point, index) => {
      if (point.team === 0) {
        scoreA++;
      } else {
        scoreB++;
      }
      return {
        type: 'point',
        id: `point-${index}`,
        timestamp: point.time,
        scoreA,
        scoreB,
      };
    });

    const combined = [...eventItems, ...pointItems];
    return combined.sort((a, b) => a.timestamp - b.timestamp);
  }, [rawEvents, points, fps]);

  useEffect(() => {
    if (!isPlaying || isDown || !scrollContainerRef.current || !cardRefs.current.length || currentFrame === undefined) return;
    const currentIndex = timelineItems.findIndex(item => item.type === 'event' && currentFrame >= item.frameId && currentFrame <= item.endFrameId);
    if (currentIndex === -1) return;
    const cardElement = cardRefs.current[currentIndex];
    const containerElement = scrollContainerRef.current;
    if (cardElement && containerElement) {
      const containerWidth = containerElement.offsetWidth;
      const targetScrollLeft = cardElement.offsetLeft + (cardElement.offsetWidth / 2) - (containerWidth / 2);
      if (Math.abs(containerElement.scrollLeft - targetScrollLeft) > 1) {
        containerElement.scrollTo({ left: targetScrollLeft, behavior: 'smooth' });
      }
    }
  }, [currentFrame, timelineItems, isDown, isPlaying]);

  useEffect(() => {
    const el = scrollContainerRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => { if (e.deltaY !== 0) { e.preventDefault(); el.scrollLeft += e.deltaY; setSelectedEventId(null); } };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, []);

  const handleMouseDown = (e: React.MouseEvent) => {
    if (!scrollContainerRef.current) return;
    setIsDown(true); isDraggingRef.current = false;
    setStartX(e.pageX - scrollContainerRef.current.offsetLeft);
    setScrollLeft(scrollContainerRef.current.scrollLeft);
    setSelectedEventId(null);
  };
  const handleMouseLeave = () => setIsDown(false);
  const handleMouseUp = () => { setIsDown(false); setTimeout(() => { isDraggingRef.current = false; }, 0); };
  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDown || !scrollContainerRef.current) return;
    e.preventDefault();
    const x = e.pageX - scrollContainerRef.current.offsetLeft;
    const walk = x - startX;
    if (Math.abs(walk) > 5) { isDraggingRef.current = true; setSelectedEventId(null); }
    scrollContainerRef.current.scrollLeft = scrollLeft - walk;
  };

  const handleCardClick = (e: React.MouseEvent<HTMLDivElement>, event: EventCardData) => {
    if (isDraggingRef.current) return;
    const rect = e.currentTarget.getBoundingClientRect();
    setPopupPosition({ top: rect.top - 12, left: rect.left + (rect.width / 2) });
    setSelectedEventId(prevId => prevId === event.id ? null : event.id);
    if (selectedEventId !== event.id) onSeek(event.timestamp);
  };

  useEffect(() => {
    if (!selectedEventId) return;
    const handleClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (!target.closest('.event-popup-overlay, .event-card')) setSelectedEventId(null);
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [selectedEventId]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 10);
    return `${mins}:${secs.toString().padStart(2, '0')}.${ms}`;
  };

  const getActionClass = (action: string) => `type-${action.toLowerCase()}`;

  const selectedEvent = timelineItems.find(e => e.id === selectedEventId) as EventCardData | undefined;

  if (timelineItems.length === 0) {
    return (
      <div className="event-viewer-container" style={{ justifyContent: 'center', color: '#94a3b8', fontSize: '0.9rem' }}>
        No events detected yet.
      </div>
    );
  }

  return (
    <>
      <div
        ref={scrollContainerRef}
        className={`event-viewer-container ${isDown ? 'is-dragging' : ''}`}
        onMouseDown={handleMouseDown}
        onMouseLeave={handleMouseLeave}
        onMouseUp={handleMouseUp}
        onMouseMove={handleMouseMove}
      >
        {timelineItems.map((item, idx) => {
          if (item.type === 'point') {
            return (
              <PointDivider
                key={item.id}
                score={`${item.scoreA} - ${item.scoreB}`}
                timestamp={formatTime(item.timestamp)}
              />
            );
          }
          
          const isCurrent = currentFrame !== undefined && currentFrame >= item.frameId && currentFrame <= item.endFrameId;
          return (
            <div
              key={item.id}
              ref={el => { cardRefs.current[idx] = el; }}
              className={`event-card ${getActionClass(item.action)}`}
              onClick={(e) => handleCardClick(e, item)}
              onDragStart={(e) => e.preventDefault()}
              style={{
                borderColor: selectedEventId === item.id ? '#94a3b8' : undefined,
                backgroundColor: selectedEventId === item.id ? '#f8fafc' : undefined,
                boxShadow: isCurrent ? '0 0 0 4px #ffe066, 0 2px 8px rgba(0,0,0,0.08)' : undefined
              }}
            >
              <div className="event-header">
                <span className="event-action">{item.action}</span>
                <span className="event-time">{formatTime(item.timestamp)}</span>
              </div>
              <div className="event-player">
                {item.player}
              </div>
              <div className="event-divider" />
              <div className="event-extra">
                <span className="event-extra-icon"></span>
                <span>{item.extraInfo}</span>
              </div>
            </div>
          );
        })}
        <div style={{ minWidth: '1px' }}></div>
      </div>

      {selectedEventId && selectedEvent && popupPosition && (
        <div
          className="event-popup-overlay"
          style={{ top: popupPosition.top, left: popupPosition.left }}
        >
          <style>{`@keyframes slideUpFade { 0% { opacity: 0; transform: translate(-50%, -90%); } 100% { opacity: 1; transform: translate(-50%, -100%); } } .event-popup-overlay { transform: translate(-50%, -100%); animation: slideUpFade 0.2s ease-out; }`}</style>
          
          <div className="popup-title">Analysis Data</div>
          <div className="popup-row">
            <span>Duration</span>
            <span>{selectedEvent.details.duration}</span>
          </div>
          {selectedEvent.details.height && (
            <div className="popup-row">
              <span>Contact Height</span>
              <span>{selectedEvent.details.height}</span>
            </div>
          )}
          {selectedEvent.details.setPosition && (
            <div className="popup-row">
              <span>Set Position</span>
              <span>{selectedEvent.details.setPosition}</span>
            </div>
          )}
          
          <div className="popup-row">
            <span>Quality</span>
            <div className="popup-quality-input-container">
              {[0, 1, 2, 3].map(q => {
                const qualityInfo = qualityMap[q as keyof typeof qualityMap];
                const isSelected = selectedEvent.quality === q;

                return (
                  <button
                    key={q}
                    className={`quality-btn ${isSelected ? 'selected' : ''}`}
                    onClick={() => onQualityChange(selectedEvent.id, q)}
                    style={{
                      backgroundColor: isSelected ? qualityInfo.bgColor : 'transparent',
                      borderColor: isSelected? '#000' : qualityInfo.color,
                    }}
                  >
                    {q}
                  </button>
                );
              })}
              {selectedEvent.quality !== undefined && (
                <button
                  className="quality-btn clear"
                  title="Clear rating"
                  onClick={() => onQualityChange(selectedEvent.id, null)}
                >
                  &times;
                </button>
              )}
            </div>
          </div>

          {!selectedEvent.details.height && (
            <div style={{ marginTop: '8px', fontSize: '0.7rem', color: '#cbd5e1', fontStyle: 'italic' }}>
              Advanced metrics unavailable
            </div>
          )}
        </div>
      )}
    </>
  );
};