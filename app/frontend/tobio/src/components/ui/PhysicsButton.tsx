import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface PhysicsButtonProps {
  children: React.ReactNode;
  onClick?: () => void;
  style?: React.CSSProperties;
  className?: string;
}

// Simple SVG Icon for the cloud upload
const CloudUpArrow = ({ ...props }) => (
  <svg
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    {...props}
  >
    <path
      d="M19.35 10.04C18.67 6.59 15.64 4 12 4C9.11 4 6.6 5.64 5.35 8.04C2.34 8.36 0 10.91 0 14C0 17.31 2.69 20 6 20H19C21.76 20 24 17.76 24 15C24 12.36 21.95 10.22 19.35 10.04ZM14 13V17H10V13H7L12 8L17 13H14Z"
      fill="white"
    />
  </svg>
);

export const PhysicsButton: React.FC<PhysicsButtonProps> = ({ 
  children, 
  onClick, 
  style, 
  className,
}) => {
  const [isHovered, setIsHovered] = useState(false);

  const textVariants = {
    initial: { y: 0, opacity: 1 },
    hover: { y: '-150%', opacity: 0 },
  };

  const iconVariants = {
    initial: { y: '150%', opacity: 0 },
    hover: { y: 0, opacity: 1 },
  };

  return (
    <motion.button
      className={`physics-upload-btn${className ? ' ' + className : ''}`}
      style={{
        ...style,
        position: 'relative',
        overflow: 'hidden',
        border: 'none',
        outline: 'none',
        cursor: 'pointer',
        background: 'linear-gradient(90deg, #3b82f6 0%, #60a5fa 100%)',
        color: '#fff',
        borderRadius: '16px',
        boxShadow: '0 4px 32px rgba(59,130,246,0.18)',
        fontWeight: 600,
        fontSize: '1.1rem',
        padding: '1.1em 2.2em',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        willChange: 'transform',
      }}
      onClick={onClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      initial={{ scale: 1 }}
      animate={{ scale: isHovered ? 1.05 : 1 }} // Animate scale instead of height
      transition={{ type: 'spring', stiffness: 400, damping: 20 }} // Spring physics on the scale
    >
      <AnimatePresence>
        <motion.div
          style={{ position: 'absolute' }}
          variants={textVariants}
          initial="initial"
          animate={isHovered ? 'hover' : 'initial'}
          transition={{ duration: 0.3, ease: 'easeOut' }}
        >
          {children}
        </motion.div>
      </AnimatePresence>

      <AnimatePresence>
        <motion.div
          style={{ position: 'absolute' }}
          variants={iconVariants}
          initial="initial"
          animate={isHovered ? 'hover' : 'initial'}
          transition={{ duration: 0.3, ease: 'easeOut' }}
        >
          <CloudUpArrow />
        </motion.div>
      </AnimatePresence>
    </motion.button>
  );
};