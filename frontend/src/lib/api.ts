import type {
  ModelRegistryResponse,
  Project,
  ProjectCreate,
  ProjectUpdate,
  Run,
  RunCreate,
  Step,
  KnowledgeEntry,
  StepEvent,
  ScheduledTask,
  ScheduledTaskCreate,
  ScheduledTaskUpdate,
  EvalSuite,
  EvalSuiteCreate,
  EvalTestCase,
  EvalTestCaseCreate,
  EvalRun,
  EvalResult,
  AppSettings,
  AppSettingsUpdate,
} from "@/types";

// Prefer explicit Vite env configuration but fall back to sensible defaults.
// Usage: set VITE_API_BASE to a full origin (https://api.example.com) or
// VITE_API_PORT to a numeric port (8000). If neither is set the client will
// attempt to contact the API on the same host with port 8000.
const getBaseUrl = () => {
  const envBase = (import.meta as any).env?.VITE_API_BASE;
  const envPort = (import.meta as any).env?.VITE_API_PORT ?? "8000";

  if (typeof window === "undefined") {
    // Server-side / build-time fallback
    return envBase ?? `http://localhost:${envPort}`;
  }

  if (envBase) return envBase;

  // Use the current page origin but a configurable API port so the dev
  // server (Vite) and API can run on different ports.
  return `${window.location.protocol}//${window.location.hostname}:${envPort}`;
};

const BASE_URL = getBaseUrl();
const WS_BASE_URL = BASE_URL.replace(/^http/, "ws");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }

  // Some endpoints may return an empty body (204). Handle that safely.
  const text = await res.text();
  return text ? (JSON.parse(text) as T) : ({} as T);
}

export const api = {
  // Models
  getModels: () => request<ModelRegistryResponse>("/api/models"),

  // Projects
  getProjects: () => request<Project[]>("/api/projects"),

  getProject: (id: number) => request<Project>(`/api/projects/${id}`),

  createProject: (payload: ProjectCreate) =>
    request<Project>("/api/projects", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  updateProject: (id: number, payload: ProjectUpdate) =>
    request<Project>(`/api/projects/${id}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),

  deleteProject: (id: number) =>
    request<{ ok: boolean }>(`/api/projects/${id}`, { method: "DELETE" }),

  // Runs
  createRun: (payload: RunCreate) =>
    request<{ id: number; status: string }>("/api/runs", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  getRuns: (projectId: number) =>
    request<Run[]>(`/api/runs?project_id=${projectId}`),

  getRun: (runId: number) =>
    request<{ run: Run; steps: Step[] }>(`/api/runs/${runId}`),

  cancelRun: (runId: number) =>
    request<{ ok: boolean }>(`/api/runs/${runId}`, { method: "DELETE" }),

  // Knowledge
  getKnowledge: (projectId: number) =>
    request<KnowledgeEntry[]>(`/api/projects/${projectId}/knowledge`),

  deleteKnowledge: (projectId: number, entryId: number) =>
    request<{ ok: boolean }>(
      `/api/projects/${projectId}/knowledge/${entryId}`,
      { method: "DELETE" }
    ),

  resetKnowledge: (projectId: number) =>
    request<{ ok: boolean }>(`/api/projects/${projectId}/knowledge/reset`, {
      method: "POST",
    }),

  // Schedules
  getSchedules: (projectId?: number) => {
    const query = projectId !== undefined ? `?project_id=${projectId}` : "";
    return request<ScheduledTask[]>(`/api/schedules${query}`);
  },

  createSchedule: (payload: ScheduledTaskCreate) =>
    request<ScheduledTask>("/api/schedules", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  updateSchedule: (id: number, payload: ScheduledTaskUpdate) =>
    request<ScheduledTask>(`/api/schedules/${id}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),

  deleteSchedule: (id: number) =>
    request<{ ok: boolean }>(`/api/schedules/${id}`, { method: "DELETE" }),

  runScheduleNow: (id: number) =>
    request<{ run_id: number; status: string }>(`/api/schedules/${id}/run-now`, {
      method: "POST",
    }),

  // Evals
  getEvalSuites: (projectId?: number) => {
    const query = projectId !== undefined ? `?project_id=${projectId}` : "";
    return request<EvalSuite[]>(`/api/evals/suites${query}`);
  },

  createEvalSuite: (payload: EvalSuiteCreate) =>
    request<EvalSuite>("/api/evals/suites", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  getEvalSuite: (id: number) =>
    request<{ suite: EvalSuite; tests: EvalTestCase[]; runs: EvalRun[] }>(
      `/api/evals/suites/${id}`
    ),

  deleteEvalSuite: (id: number) =>
    request<{ ok: boolean }>(`/api/evals/suites/${id}`, { method: "DELETE" }),

  addEvalTest: (suiteId: number, payload: EvalTestCaseCreate) =>
    request<EvalTestCase>(`/api/evals/suites/${suiteId}/tests`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  deleteEvalTest: (testId: number) =>
    request<{ ok: boolean }>(`/api/evals/tests/${testId}`, {
      method: "DELETE",
    }),

  runEvalSuite: (suiteId: number) =>
    request<{ eval_run_id: number; status: string }>(
      `/api/evals/suites/${suiteId}/run`,
      { method: "POST" }
    ),

  getEvalRun: (runId: number) =>
    request<{ eval_run: EvalRun; results: EvalResult[] }>(
      `/api/evals/runs/${runId}`
    ),

  // Settings
  getSettings: () => request<AppSettings>("/api/settings"),

  updateSettings: (payload: AppSettingsUpdate) =>
    request<AppSettings>("/api/settings", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  // WebSocket
  connectRunSocket: (
    runId: number,
    onMessage: (event: StepEvent) => void,
    onClose?: () => void,
    onError?: (err: Event) => void
  ): WebSocket => {
    const ws = new WebSocket(`${WS_BASE_URL}/ws/run/${runId}`);
    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        onMessage(data as StepEvent);
      } catch (err) {
        console.error("Failed to parse websocket message:", err);
      }
    };
    if (onClose) ws.onclose = onClose;
    if (onError) ws.onerror = onError;
    return ws;
  },
};
