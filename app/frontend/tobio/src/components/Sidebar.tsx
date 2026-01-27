import React from 'react';
import { Video, BarChart, Wrench, Settings as SettingsIcon, Bot } from 'lucide-react';

const ExportArrow = ({ size = 22 }: { size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 5v14"/>
    <path d="M19 12l-7 7-7-7"/>
  </svg>
);

interface SidebarProps {
  onNavigate: (path: string) => void;
  collapsed: boolean;
  setCollapsed: (collapsed: boolean) => void;
  selected: string | null;
}

export const Sidebar: React.FC<SidebarProps> = ({ onNavigate, collapsed, setCollapsed, selected }) => (
  <aside
    className={`dashboard-sidebar${collapsed ? ' collapsed' : ''}`}
    onMouseEnter={() => setCollapsed(false)}
    onMouseLeave={() => setCollapsed(true)}
  >
    <nav>
      <ul>
        <li>
          <button
            onClick={() => onNavigate('view')}
            className={`sidebar-nav-btn${selected === 'view' ? ' selected' : ''}`}
          >
            <Wrench size={22} />
            {!collapsed && <span className="sidebar-btn-label">View</span>}
          </button>
        </li>
        <li>
          <button
            onClick={() => onNavigate('stats')}
            className={`sidebar-nav-btn${selected === 'stats' ? ' selected' : ''}`}
          >
            <BarChart size={22} />
            {!collapsed && <span className="sidebar-btn-label">Stats</span>}
          </button>
        </li>
        <li>
          <button
            onClick={() => onNavigate('camera')}
            className={`sidebar-nav-btn${selected === 'camera' ? ' selected' : ''}`}
          >
            <Video size={22} />
            {!collapsed && <span className="sidebar-btn-label">Camera</span>}
          </button>
        </li>
        <li>
          <button
            onClick={() => onNavigate('agent')}
            className={`sidebar-nav-btn${selected === 'agent' ? ' selected' : ''}`}
          >
            <Bot size={22} />
            {!collapsed && <span className="sidebar-btn-label">Agent</span>}
          </button>
        </li>
        <li>
          <button
            onClick={() => onNavigate('settings')}
            className={`sidebar-nav-btn${selected === 'settings' ? ' selected' : ''}`}
          >
            <SettingsIcon size={22} />
            {!collapsed && <span className="sidebar-btn-label">Settings</span>}
          </button>
        </li>
      </ul>
    </nav>
    <div style={{ flex: 1 }} />
    <button className="sidebar-nav-btn sidebar-export-btn">
      <ExportArrow size={22} />
      {!collapsed && <span className="sidebar-btn-label">Export VOD</span>}
    </button>
  </aside>
);
