import { act, renderHook } from "@testing-library/react-native";
import type { ReactNode } from "react";
import { AppState } from "react-native";

import { DaemonProvider, POLL_MS, UNREACHABLE_POLL_MS, useDaemon } from "../DaemonContext";

const mockVersion = jest.fn();
const mockClient = { version: mockVersion };
jest.mock("@/lib/ServersContext", () => ({ useServers: () => ({ client: mockClient }) }));
jest.mock("@/lib/daemonFeatures", () => ({
  detectFeatures: async () => ({ shell: { enabled: true, endpoint: true } }),
}));

const wrapper = ({ children }: { children: ReactNode }) => <DaemonProvider>{children}</DaemonProvider>;
const answer = { version: "0.1.0", exe: "llmman", pid: 1 };

function deferred<T>() {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

async function connected() {
  mockVersion.mockResolvedValueOnce(answer);
  const hook = renderHook(() => useDaemon(), { wrapper });
  await act(async () => {});
  expect(hook.result.current.reachable).toBe(true);
  return hook;
}

beforeEach(() => {
  jest.useFakeTimers();
  mockVersion.mockReset();
});
const appState = AppState.currentState;
afterEach(() => {
  jest.useRealTimers();
  AppState.currentState = appState;
});

test("checkReachable polls in the background; the timer does not", async () => {
  const { result } = await connected();
  AppState.currentState = "background";

  await act(async () => {
    jest.advanceTimersByTime(POLL_MS);
  });
  expect(mockVersion).toHaveBeenCalledTimes(1);

  mockVersion.mockRejectedValueOnce(new Error("refused"));
  await act(async () => result.current.checkReachable());
  expect(mockVersion).toHaveBeenCalledTimes(2);
  expect(result.current.reachable).toBe(false);
});

test("an older poll answering late does not undo checkReachable's newer failure", async () => {
  const { result } = await connected();

  // The 15 s poll is in flight when the shell disconnects.
  const older = deferred<typeof answer>();
  mockVersion.mockReturnValueOnce(older.promise);
  await act(async () => {
    jest.advanceTimersByTime(POLL_MS);
  });
  const newer = deferred<typeof answer>();
  mockVersion.mockReturnValueOnce(newer.promise);
  act(() => result.current.checkReachable());

  await act(async () => newer.reject(new Error("refused")));
  expect(result.current.reachable).toBe(false);
  await act(async () => older.resolve(answer));
  expect(result.current.reachable).toBe(false);
});

test("while unreachable, polls every UNREACHABLE_POLL_MS until llmman serve answers", async () => {
  const { result } = await connected();
  mockVersion.mockRejectedValueOnce(new Error("refused"));
  await act(async () => result.current.checkReachable());
  expect(result.current.reachable).toBe(false);

  mockVersion.mockResolvedValueOnce(answer);
  await act(async () => {
    jest.advanceTimersByTime(UNREACHABLE_POLL_MS);
  });
  expect(result.current.reachable).toBe(true);
});

test("while reachable, only the 15 s poll runs", async () => {
  await connected();
  await act(async () => {
    jest.advanceTimersByTime(POLL_MS - 1);
  });
  expect(mockVersion).toHaveBeenCalledTimes(1);
});
