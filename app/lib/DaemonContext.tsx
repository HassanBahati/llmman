// The active llmman serve as the frame shows it: whether it answers (a
// `/api/version` poll every 15 s, every 3 s while it doesn't, as llmman's
// web UI drives its status dot) and which optional features it has
// (daemonFeatures.ts). Screens read it with `useDaemon()`.

import { createContext, useContext, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { AppState } from "react-native";

import type { VersionInfo } from "@/api/client";
import { detectFeatures, type DaemonFeatures } from "@/lib/daemonFeatures";
import { useServers } from "@/lib/ServersContext";

export const POLL_MS = 15_000;
export const UNREACHABLE_POLL_MS = 3_000;

export interface DaemonState {
  /** Null until the first answer (or failure) for this server. */
  reachable: boolean | null;
  version: VersionInfo | null;
  /** Null until probed: treat as none. */
  features: DaemonFeatures | null;
  /** Asks again now, e.g. after the key changed. */
  refresh(): void;
  /** Polls now, even in the background; unlike refresh(), keeps features. */
  checkReachable(): void;
}

interface Snapshot {
  client: unknown;
  reachable: boolean | null;
  version: VersionInfo | null;
  features: DaemonFeatures | null;
}

const DaemonContext = createContext<DaemonState | null>(null);

// Answers can arrive out of order: once one is accepted, older ones are dropped.
function inOrder() {
  let started = 0;
  let accepted = 0;
  return {
    start: () => ++started,
    accept: (seq: number) => {
      if (seq < accepted) return false;
      accepted = seq;
      return true;
    },
  };
}

export function DaemonProvider({ children }: { children: ReactNode }) {
  const { client } = useServers();
  const [state, setState] = useState<Snapshot>({
    client: null,
    reachable: null,
    version: null,
    features: null,
  });
  const [tick, setTick] = useState(0);
  const pollNow = useRef<(() => void) | null>(null);

  useEffect(() => {
    if (!client) return;
    let cancelled = false;
    // This client's snapshot, started afresh when the server or key changed.
    const update = (patch: Partial<Snapshot>) =>
      setState((s) => ({
        ...(s.client === client ? s : { client, reachable: null, version: null, features: null }),
        ...patch,
      }));
    // Features: once per client, and again when the daemon comes back after
    // being unreachable (it may have been restarted with other settings).
    const probes = inOrder();
    const probe = () => {
      const seq = probes.start();
      void detectFeatures(client, { timeoutMs: 8000 }).then(
        (features) => !cancelled && probes.accept(seq) && update({ features }),
      );
    };
    let wasReachable: boolean | null = null;
    const polls = inOrder();
    let waiting = 0;
    const poll = async (pollInBackground = false) => {
      // Not while the app is in the background on a phone.
      if (!pollInBackground && AppState.currentState === "background") return;
      const seq = polls.start();
      waiting++;
      const version = await client.version({ timeoutMs: 8000 }).catch(() => null);
      waiting--;
      if (cancelled || !polls.accept(seq)) return;
      const reachable = version !== null;
      if (reachable && wasReachable === false) probe();
      wasReachable = reachable;
      update(version ? { reachable, version } : { reachable });
    };
    probe();
    void poll();
    const timer = setInterval(() => void poll(), POLL_MS);
    // Not while a poll waits, so retries don't pile up on a hanging network.
    const unreachableTimer = setInterval(
      () => wasReachable === false && waiting === 0 && void poll(),
      UNREACHABLE_POLL_MS,
    );
    pollNow.current = () => void poll(true);
    return () => {
      cancelled = true;
      clearInterval(timer);
      clearInterval(unreachableTimer);
      pollNow.current = null;
    };
  }, [client, tick]);

  const current = state.client === client ? state : null;
  const value = useMemo<DaemonState>(
    () => ({
      reachable: current?.reachable ?? null,
      version: current?.version ?? null,
      features: current?.features ?? null,
      refresh: () => setTick((t) => t + 1),
      checkReachable: () => pollNow.current?.(),
    }),
    [current],
  );
  return <DaemonContext.Provider value={value}>{children}</DaemonContext.Provider>;
}

export function useDaemon(): DaemonState {
  const ctx = useContext(DaemonContext);
  if (!ctx) throw new Error("useDaemon outside DaemonProvider");
  return ctx;
}
