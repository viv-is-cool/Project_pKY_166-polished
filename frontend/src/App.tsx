import { useState, useEffect } from "react";
import { Shell } from "@/components/layout/Shell";
import { Dashboard } from "@/pages/Dashboard";
import { SwarmBuilder } from "@/pages/SwarmBuilder";
import { RunMonitor } from "@/pages/RunMonitor";
import { KnowledgeBase } from "@/pages/KnowledgeBase";
import { Schedules } from "@/pages/Schedules";
import { Evals } from "@/pages/Evals";
import { Settings } from "@/pages/Settings";
import { Projects } from "@/pages/Projects";
import { Chat } from "@/pages/Chat";
import { api } from "@/lib/api";
import type { Project } from "@/types";

export default function App() {
  const [activePage, setActivePage] = useState<string>("dashboard");
  const [activeProjectId, setActiveProjectId] = useState<number | null>(null);
  const [activeRunId, setActiveRunId] = useState<number | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);

  useEffect(() => {
    // Load project list once on mount. Previously this effect depended on
    // activeProjectId which could cause repeated fetches and unstable state.
    api
      .getProjects()
      .then((projs) => {
        setProjects(projs);
        if (projs.length > 0 && activeProjectId === null) {
          setActiveProjectId(projs[0].id);
        }
      })
      .catch((err) => {
        console.error("Failed to load projects:", err);
      });
  }, []);

  const handleNavigate = (page: string, params?: Record<string, any>) => {
    if (params?.projectId !== undefined) {
      setActiveProjectId(params.projectId);
    }
    if (params?.runId !== undefined) {
      setActiveRunId(params.runId);
    }
    setActivePage(page);
  };

  if (activePage === "swarm-builder") {
    return (
      <SwarmBuilder
        projectId={activeProjectId}
        onNavigate={handleNavigate}
      />
    );
  }

  if (activePage === "run-monitor") {
    return (
      <RunMonitor
        runId={activeRunId}
        projectId={activeProjectId}
        onNavigate={handleNavigate}
      />
    );
  }

  return (
    <Shell activePage={activePage} onNavigate={handleNavigate}>
      {activePage === "dashboard" && (
        <Dashboard
          onNavigate={handleNavigate}
          activeProjectId={activeProjectId}
          onSelectProject={setActiveProjectId}
        />
      )}
      {activePage === "projects" && (
        <Projects
          onNavigate={handleNavigate}
          onSelectProject={setActiveProjectId}
        />
      )}
      {activePage === "chat" && (
        <Chat
          projectId={activeProjectId}
          onNavigate={handleNavigate}
        />
      )}
      {activePage === "knowledge" && (
        <KnowledgeBase
          projectId={activeProjectId}
          projects={projects}
          onSelectProject={setActiveProjectId}
        />
      )}
      {activePage === "schedules" && (
        <Schedules onNavigate={handleNavigate} />
      )}
      {activePage === "evals" && (
        <Evals />
      )}
      {activePage === "settings" && (
        <Settings />
      )}
    </Shell>
  );
}
