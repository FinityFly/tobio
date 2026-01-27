import React from 'react';

interface CameraSettingsProps {
  values: {
    cameraHeight: number;
    focalLength: number;
    ballHeightCalibration: number;
    ballSideCalibration: number;
    groundPlaneOffset: number;
    onChange: (key: keyof CameraSettingsValues, value: number) => void;
  };
  onApply: () => void;
}

export type CameraSettingsValues = {
  cameraHeight: number;
  focalLength: number;
  ballHeightCalibration: number;
  ballSideCalibration: number;
  groundPlaneOffset: number;
};

export const CameraSettings: React.FC<CameraSettingsProps> = ({ values, onApply }) => (
  <div className="camera-settings-toolbar">
    <Slider
      label="Camera Height"
      value={values.cameraHeight}
      min={0.1}
      max={15}
      step={0.1}
      onChange={v => values.onChange('cameraHeight', v)}
    />
    <Slider
      label="Focal Length"
      value={values.focalLength}
      min={0.5}
      max={5}
      step={0.05}
      onChange={v => values.onChange('focalLength', v)}
    />
    <Slider
      label="Ball Height Calibration"
      value={values.ballHeightCalibration}
      min={0.1}
      max={3}
      step={0.05}
      onChange={v => values.onChange('ballHeightCalibration', v)}
    />
    <Slider
      label="Ball Side Calibration"
      value={values.ballSideCalibration}
      min={0.5}
      max={5}
      step={0.05}
      onChange={v => values.onChange('ballSideCalibration', v)}
    />
    <Slider
      label="Ground Plane Offset"
      value={values.groundPlaneOffset}
      min={-15}
      max={15}
      step={0.1}
      onChange={v => values.onChange('groundPlaneOffset', v)}
    />
    <button
      style={{
        marginTop: '1.2em',
        padding: '10px 24px',
        background: 'linear-gradient(90deg, #22c55e, #16a34a)',
        color: 'white',
        fontWeight: 700,
        fontSize: '1.05rem',
        borderRadius: '8px',
        border: 'none',
        boxShadow: '0 2px 8px rgba(0,0,0,0.12)',
        cursor: 'pointer',
        width: '100%'
      }}
      onClick={onApply}
    >
      Apply
    </button>
  </div>
);

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
}

const Slider: React.FC<SliderProps> = ({ label, value, min, max, step = 0.01, onChange }) => {
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
          step={step}
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
            step={step}
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
            {value}
          </span>
        )}
      </div>
    </div>
  );
};