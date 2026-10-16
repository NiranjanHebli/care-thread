// CareThread API Client & Event Stream Protocol

export interface PatientSummary {
  patient_id: string;
  name: string;
  gender?: string | null;
  birth_date?: string | null;
  observation_count: number;
  condition_count: number;
}

export interface CytoscapeNodeData {
  id: string;
  label: string;
  type: string;
  name?: string;
  is_clinic?: boolean;
  status?: string;
  source_tool?: string;
  created_at?: string;
  observation_count?: number;
  [key: string]: unknown;
}

export interface CytoscapeEdgeData {
  id: string;
  source: string;
  target: string;
  label: string;
  type: string;
  [key: string]: unknown;
}

export interface GraphPayload {
  elements: {
    nodes: Array<{ data: CytoscapeNodeData }>;
    edges: Array<{ data: CytoscapeEdgeData }>;
  };
}

export interface TraceData {
  answer_id: string;
  session_id: string;
  used: string[];
  rejected: string[];
  stale_nodes: string[];
  confidence: number;
  agent_role: string;
  rules_applied?: string[];
  guidelines_consulted?: string[];
}

export interface WhyNotResult {
  answer_id: string;
  candidate_node_id: string;
  reason_code: string;
  blocking_path: Array<{
    source: string;
    relationship: string;
    target: string;
  }>;
}

export interface HandoffRecord {
  handoff_id: string;
  sender_role: string;
  receiver_role: string;
  status: string;
  created_at: string;
  expires_at?: string;
  clinical_summary?: string;
  priority?: string;
  active_flags?: string[];
}

export interface FlagRecord {
  flag_id: string;
  flag_type: string;
  severity: string;
  reason: string;
  status: string;
  created_at: string;
  resolved_at?: string;
  resolved_by?: string;
}

export interface AuditEvent {
  event_id: string;
  timestamp: string;
  agent_role: string;
  patient_id: string;
  action: string;
  status: string;
  latency_ms?: number;
}

const API_BASE = '';

export async function fetchPatients(query = ''): Promise<PatientSummary[]> {
  const endpoint = `${API_BASE}/api/patients`;
  const url = query ? `${endpoint}?q=${encodeURIComponent(query)}` : endpoint;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch patients: ${res.statusText}`);
  const data = await res.json();
  return data.patients || [];
}

export async function fetchPatientGraph(patientId: string, asOf?: string): Promise<GraphPayload> {
  let url = `${API_BASE}/api/patients/${encodeURIComponent(patientId)}/graph`;
  if (asOf) url += `?as_of=${encodeURIComponent(asOf)}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch patient graph: ${res.statusText}`);
  return res.json();
}

export async function fetchAnswerTrace(answerId: string): Promise<TraceData> {
  const res = await fetch(`${API_BASE}/api/answers/${encodeURIComponent(answerId)}/trace`);
  if (!res.ok) throw new Error(`Failed to fetch answer trace: ${res.statusText}`);
  const data = await res.json();
  return data.trace || data;
}

export async function fetchWhyNot(
  answerId: string,
  candidateNodeId: string
): Promise<WhyNotResult> {
  const res = await fetch(`${API_BASE}/api/answers/${encodeURIComponent(answerId)}/why-not`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ candidate_node_id: candidateNodeId }),
  });
  if (!res.ok) throw new Error(`Why-not query failed: ${res.statusText}`);
  return res.json();
}

export async function fetchHandoffs(patientId: string): Promise<HandoffRecord[]> {
  const res = await fetch(`${API_BASE}/api/handoffs?patient_id=${encodeURIComponent(patientId)}`);
  if (!res.ok) throw new Error(`Failed to fetch handoffs: ${res.statusText}`);
  const data = await res.json();
  return data.handoffs || [];
}

export async function fetchFlags(patientId: string): Promise<FlagRecord[]> {
  const res = await fetch(`${API_BASE}/api/flags?patient_id=${encodeURIComponent(patientId)}`);
  if (!res.ok) throw new Error(`Failed to fetch flags: ${res.statusText}`);
  const data = await res.json();
  return data.flags || [];
}

export async function resolveFlag(
  flagId: string,
  patientId: string,
  reason = 'Clinician override'
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/flags/${encodeURIComponent(flagId)}/resolve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      patient_id: patientId,
      resolution_reason: reason,
      resolved_by: 'clinician_user',
    }),
  });
  if (!res.ok) throw new Error(`Failed to resolve flag: ${res.statusText}`);
}

export async function fetchAuditLogs(patientId: string, limit = 100): Promise<AuditEvent[]> {
  const res = await fetch(
    `${API_BASE}/api/audit?patient_id=${encodeURIComponent(patientId)}&limit=${limit}`
  );
  if (!res.ok) throw new Error(`Failed to fetch audit: ${res.statusText}`);
  const data = await res.json();
  return data.events || [];
}

export async function fetchMetricsText(): Promise<string> {
  const res = await fetch(`${API_BASE}/metrics`);
  if (!res.ok) throw new Error(`Failed to fetch metrics: ${res.statusText}`);
  return res.text();
}

export async function streamAgentChat(
  role: string,
  req: { patient_id: string; message: string; session_id?: string; handoff_id?: string },
  onEvent: (event: string, data: any) => void
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/agents/${encodeURIComponent(role)}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });

  if (!res.ok) throw new Error(`Chat stream failed: ${res.statusText}`);
  if (!res.body) return;

  const reader = res.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split('\n\n');
    buffer = blocks.pop() || '';

    for (const block of blocks) {
      if (!block.trim()) continue;
      const lines = block.split('\n');
      let event = 'message';
      let dataStr = '';

      for (const line of lines) {
        if (line.startsWith('event: ')) {
          event = line.slice(7).trim();
        } else if (line.startsWith('data: ')) {
          dataStr = line.slice(6).trim();
        }
      }

      if (dataStr) {
        try {
          const parsed = JSON.parse(dataStr);
          onEvent(event, parsed);
        } catch (e) {
          onEvent(event, { raw: dataStr });
        }
      }
    }
  }
}
