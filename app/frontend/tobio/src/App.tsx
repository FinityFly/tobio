import './styles/dashboard.css';
import './styles/videoplayer.css'
import './styles/sidebar.css';
import './styles/subsidebar.css';
import './styles/toolbar.css';
import './styles/navbar.css';
import './styles/hero.css';
import './styles/utilities.css';
import './styles/eventviewer.css'
import './styles/statisticsPanel.css'
import './styles/csvmodal.css'
import './styles/shadcn-components.css';

import { PhysicsButton } from './components/ui/PhysicsButton';
import { Input } from './components/ui/Input';
import { FileUp } from 'lucide-react';

import React, { useRef, useState } from 'react';
import { Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import { Loading } from './components/Loading';
import { Dashboard } from './components/Dashboard';
import volleyballBg from './assets/bg_footage.mp4';
import tobioLogo from './assets/tobio_bg.png';
import demoVideo from './assets/vod_1m.mp4';


const Hero: React.FC = () => {
  const inputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleDemoFootage = async () => {
    setLoading(true);
    try {
      const response = await fetch(demoVideo);
      const blob = await response.blob();
      const file = new File([blob], 'demo_vod.mp4', { type: 'video/mp4' });
      setSelectedFile(file);
      setTimeout(() => {
        navigate('/dashboard', { state: { file } });
      }, 1200);
    } catch (err) {
      alert('Failed to load demo footage.');
      setLoading(false);
    }
  };

  const handleButtonClick = () => {
    inputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (file.size > 50 * 1024 * 1024 * 1024) { // 50GB in bytes
        setSelectedFile(null);
        alert('File size exceeds 50GB limit.');
        return;
      }
      if (file.type === 'video/mp4') {
        setSelectedFile(file);
        setLoading(true);
        setTimeout(() => {
          navigate('/dashboard', { state: { file } });
        }, 1800);
      } else {
        setSelectedFile(null);
        alert('Please select a valid .mp4 file.');
      }
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      if (file.size > 50 * 1024 * 1024 * 1024) { // 50GB in bytes
        setSelectedFile(null);
        alert('File size exceeds 50GB limit.');
        return;
      }
      if (file.type === 'video/mp4') {
        setSelectedFile(file);
        setLoading(true);
        setTimeout(() => {
          navigate('/dashboard', { state: { file } });
        }, 1800);
      } else {
        setSelectedFile(null);
        alert('Please drop a valid .mp4 file.');
      }
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragActive(true);
  };

  const handleDragLeave = () => {
    setDragActive(false);
  };

  // Parallax effect for Tobio logo
  const logoRef = React.useRef<HTMLImageElement>(null);
  const [logoStyle, setLogoStyle] = React.useState({
    transform: 'perspective(600px) rotateX(0deg) rotateY(0deg) scale(1)',
    transition: 'transform 0.2s cubic-bezier(.4,0,.2,1)',
    boxShadow: '0 2px 16px rgba(0, 0, 0, 0.15)',
  });

  const handleLogoMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!logoRef.current) return;
    const rect = logoRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left - rect.width / 2;
    const y = e.clientY - rect.top - rect.height / 2;
    // Invert the direction for a true 3D effect (tilt away from mouse)
    const rotateX = (y / 12).toFixed(2);
    const rotateY = (-x / 12).toFixed(2);
    setLogoStyle({
      transform: `perspective(600px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale(1.07)`,
      transition: 'transform 0.15s cubic-bezier(.4,0,.2,1)',
      boxShadow: '0 8px 32px rgba(59,130,246,0.18)',
    });
  };
  const handleLogoMouseLeave = () => {
    setLogoStyle({
      transform: 'perspective(600px) rotateX(0deg) rotateY(0deg) scale(1)',
      transition: 'transform 0.3s cubic-bezier(.4,0,.2,1)',
      boxShadow: '0 2px 16px rgba(0, 0, 0, 0.15)',
    });
  };

  return loading ? (
    <Loading message="Uploading video..." />
  ) : (
    <main className="hero" style={{ position: 'relative', overflow: 'hidden' }}>
      <video
        autoPlay
        loop
        muted
        playsInline
        src={volleyballBg}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100vw',
          height: '100vh',
          objectFit: 'cover',
          opacity: 0.15,
          zIndex: 0,
          pointerEvents: 'none',
        }}
      />
      <div style={{ position: 'relative', zIndex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '100vh', width: '100%' }}>
        <div
          style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginBottom: '2rem', cursor: 'pointer' }}
          onMouseMove={handleLogoMouseMove}
          onMouseLeave={handleLogoMouseLeave}
        >
          <img
            ref={logoRef}
            src={tobioLogo}
            alt="Tobio Logo"
            style={{ width: 120, height: 120, objectFit: 'contain', borderRadius: '18px', ...logoStyle }}
          />
        </div>
        <div className="upload-label">Introducing Tobio</div>
        <div style={{ color: '#444', fontSize: '1.15rem', marginBottom: '2.5rem', textAlign: 'center', fontWeight: 500 }}>
          Volleyball stats has never been this easy.
        </div>
        <div
          className="upload-box"
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          style={dragActive ? { borderColor: '#3b82f6', background: '#eaf3ff' } : { }}
        >
          <div style={{ marginBottom: '1.5rem', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <FileUp size={56} color="#3b82f6" style={{ marginBottom: '0.5rem' }} />
            <span style={{ fontWeight: 600, fontSize: '1.1rem', color: '#3b82f6', background: '#eaf3ff', borderRadius: '8px', padding: '0.2rem 0.7rem' }}>MP4</span>
          </div>
          <PhysicsButton
            style={{ width: '100%', height: 56, maxWidth: 220, fontSize: '1.1rem', marginBottom: '0.5rem' }}
            onClick={handleDemoFootage}
          >
            Use Demo Footage
          </PhysicsButton>
          <button
            type="button"
            className="demo-footage-btn"
            onClick={handleButtonClick}
          >
            ...or upload your own
          </button>
          <Input
            ref={inputRef}
            type="file"
            accept="video/mp4"
            className="upload-input"
            style={{ display: 'none' }}
            onChange={handleFileChange}
          />
          {selectedFile && (
            <div style={{ color: '#3b82f6', fontWeight: 600, marginTop: '1rem', textAlign: 'center' }}>
              Selected file: {selectedFile.name}
            </div>
          )}
          <div style={{ color: '#888', fontSize: '0.85rem', textAlign: 'center', marginTop: '0.5rem' }}>
            Maximum file size: 50GB
          </div>
        </div>
        <div style={{ position: 'absolute', bottom: 50, fontSize: '0.9rem', color: '#aaa' }}>
          For the love of the game. ❤️
        </div>
      </div>
    </main>
  );
};

const DashboardWrapper: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const file = (location.state as any)?.file;

  React.useEffect(() => {
    if (!file) {
      const timer = setTimeout(() => {
        navigate('/');
      }, 1500);
      return () => clearTimeout(timer);
    }
  }, [file, navigate]);

  if (!file) {
    return <Loading message="No video uploaded. Redirecting..." />;
  }
  return <Dashboard file={file} />;
};

const App: React.FC = () => (
  <Routes>
    <Route path="/" element={<Hero />} />
    <Route path="/dashboard" element={<DashboardWrapper />} />
  </Routes>
);

export default App;
