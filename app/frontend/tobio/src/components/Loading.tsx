import React from 'react';

interface LoadingProps {
  message?: string;
}

export const Loading: React.FC<LoadingProps> = ({ message = "Loading..." }) => (
  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '60vh' }}>
    <div className="loader" style={{ marginBottom: '1.5rem' }} />
    <div style={{ color: '#000000ff', fontWeight: 600, fontSize: '1.2rem' }}>{message}</div>
  </div>
);
