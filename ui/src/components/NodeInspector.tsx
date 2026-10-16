import React from 'react';
import type { CytoscapeNodeData } from '../api';

interface NodeInspectorProps {
  nodeData: CytoscapeNodeData | null;
  onClose: () => void;
  onWhyNotQuery?: (nodeId: string) => void;
}

export const NodeInspector: React.FC<NodeInspectorProps> = ({
  nodeData,
  onClose,
  onWhyNotQuery,
}) => {
  if (!nodeData) return null;

  // Filter out embedding vectors and internal cy properties per clinical requirements
  const filteredProperties = Object.entries(nodeData).filter(([key]) => {
    const k = key.toLowerCase();
    return !k.includes('embedding') && !k.includes('vector') && k !== 'label';
  });

  return (
    <div className="glass-card" id="node-inspector" style={{ marginTop: '12px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h4 style={{ fontSize: '13px', color: 'var(--accent-cyan)' }}>
          Node Inspector: {nodeData.type || 'Node'}
        </h4>
        <button
          className="btn btn-secondary"
          id="close-inspector-btn"
          onClick={onClose}
          style={{ padding: '2px 8px', fontSize: '11px' }}
        >
          X
        </button>
      </div>

      <div
        style={{
          marginTop: '8px',
          fontSize: '12px',
          display: 'flex',
          flexDirection: 'column',
          gap: '6px',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ color: 'var(--text-secondary)' }}>ID:</span>
          <span style={{ fontFamily: 'monospace' }}>{nodeData.id}</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ color: 'var(--text-secondary)' }}>Type:</span>
          <span style={{ fontWeight: 600 }}>{nodeData.type}</span>
        </div>

        {filteredProperties.map(([k, v]) => (
          <div
            key={k}
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              gap: '12px',
              borderTop: '1px solid rgba(255,255,255,0.05)',
              paddingTop: '4px',
            }}
          >
            <span style={{ color: 'var(--text-secondary)' }}>{k}:</span>
            <span style={{ textAlign: 'right', wordBreak: 'break-all', maxWidth: '240px' }}>
              {typeof v === 'object' ? JSON.stringify(v) : String(v)}
            </span>
          </div>
        ))}

        {onWhyNotQuery && (
          <button
            className="btn btn-secondary"
            id={`inspect-whynot-${nodeData.id}`}
            onClick={() => onWhyNotQuery(nodeData.id)}
            style={{ marginTop: '8px', fontSize: '11px', width: '100%' }}
          >
            Analyze Why-Not for this Node
          </button>
        )}
      </div>
    </div>
  );
};
