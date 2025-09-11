// src/components/RobotSpace.jsx
import React from 'react';

export default function RobotSpace({ robotActions, isActive }) {
  return (
    <div style={{
      width: '100%',
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      alignItems: 'center',
      color: 'white',
      fontSize: '24px'
    }}>
      
      {/* {isActive ? (
        <div>🤖 Robot Active</div>
      ) : (
        <div>Robot Space</div>
      )} */}
      
      {robotActions && robotActions.length > 0 && (
        <div style={{ marginTop: '20px', fontSize: '16px' }}>
          Last action: {robotActions[robotActions.length - 1]?.tool_name}
        </div>
      )}
    </div>
  );
}