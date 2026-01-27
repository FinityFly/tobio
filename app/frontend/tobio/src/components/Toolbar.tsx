import React from 'react';

interface ToolbarProps {
  toggles: {
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
}

export const Toolbar: React.FC<ToolbarProps> = ({ toggles }) => (
  <div className="video-toolbar">
    <ToggleSwitch
      label="Court Tracking"
      checked={toggles.courtTracking}
      onChange={() => toggles.onToggle('courtTracking')}
    />
    <ToggleSwitch
      label="Ball Tracking"
      checked={toggles.ballTracking}
      onChange={() => toggles.onToggle('ballTracking')}
    />
    <ToggleSwitch
      label="Player Tracking"
      checked={toggles.playerTracking}
      onChange={() => toggles.onToggle('playerTracking')}
    />
    <div className="toolbar-separator" />
    <ToggleSwitch
      label="Net View"
      checked={toggles.netView}
      onChange={() => toggles.onToggle('netView')}
    />
    <Slider
      label="Net View Window Size"
      value={toggles.netViewWindowSize}
      min={100}
      max={600}
      onChange={toggles.onNetViewWindowSizeChange}
    />
    <ToggleSwitch
      label="Ball Height"
      checked={toggles.ballHeight}
      onChange={() => toggles.onToggle('ballHeight')}
    />
    <div className="toolbar-separator" />
    <ToggleSwitch
      label="Points Navigator"
      checked={toggles.points}
      onChange={() => toggles.onToggle('points')}
    />
    <ToggleSwitch
      label="Events Navigator"
      checked={toggles.events}
      onChange={() => toggles.onToggle('events')}
    />
  </div>
);

interface ToggleSwitchProps {
  label: string;
  checked: boolean;
  onChange: () => void;
}

const ToggleSwitch: React.FC<ToggleSwitchProps> = ({ label, checked, onChange }) => (
  <label className="toggle-switch">
    <input type="checkbox" checked={checked} onChange={onChange} />
    <span className="slider">
      <span className="knob" />
    </span>
    <span className="toggle-label">{label}</span>
  </label>
);

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (value: number) => void;
}

const Slider: React.FC<SliderProps> = ({ label, value, min, max, onChange }) => {
  const [editing, setEditing] = React.useState(false);
  const [inputValue, setInputValue] = React.useState<number>(value);

  React.useEffect(() => {
    setInputValue(value);
  }, [value]);

  const handleLabelClick = () => setEditing(true);
  const handleInputBlur = () => {
    setEditing(false);
    if (!isNaN(Number(inputValue))) onChange(Number(inputValue));
  };
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = Number(e.target.value);
    if (!isNaN(val)) setInputValue(val);
  };
  const handleInputKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      setEditing(false);
      if (!isNaN(Number(inputValue))) onChange(Number(inputValue));
    }
  };

  return (
    <div className="toolbar-slider-row" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '0.3em', width: '100%' }}>
      <span className="toggle-label" style={{ cursor: 'pointer' }} onClick={handleLabelClick}>{label}</span>
      <div style={{ display: 'flex', alignItems: 'center', width: '100%' }}>
        <input
          type="range"
          min={min}
          max={max}
          value={value}
          onChange={e => onChange(Number(e.target.value))}
          className="toolbar-slider"
          style={{ width: '70%' }}
        />
        {editing ? (
          <input
            type="number"
            value={inputValue}
            min={min}
            max={max}
            onBlur={handleInputBlur}
            onChange={handleInputChange}
            onKeyDown={handleInputKeyDown}
            style={{ marginLeft: '0.7em', width: 70, fontWeight: 500, fontSize: '0.95em', borderRadius: 4, border: '1px solid #ccc', padding: '2px 6px' }}
            autoFocus
          />
        ) : (
          <span
            style={{ marginLeft: '0.7em', fontWeight: 500, color: '#222', fontSize: '0.95em', cursor: 'pointer', minWidth: 40 }}
            onClick={handleLabelClick}
            title="Click to edit value"
          >
            {value}px
          </span>
        )}
      </div>
    </div>
  );
};
