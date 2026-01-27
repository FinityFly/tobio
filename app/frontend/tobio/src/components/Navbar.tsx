import React from 'react';
import logo from '../assets/tobio.svg';
import { Button } from './ui/Button';

interface NavbarProps {
  onExit: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({ onExit }) => (
  <header className="dashboard-navbar">
    <div className="navbar-left">
      <img src={logo} alt="Tobio" className="navbar-logo" />
      <span className="navbar-title">Tobio</span>
    </div>
    <div className="navbar-right">
      <Button variant="outline" onClick={onExit} style={{ fontWeight: 600, color: '#ffffff' }}>Exit</Button>
    </div>
  </header>
);
