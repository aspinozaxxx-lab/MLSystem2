import { useCallback, useEffect, useState, type RefObject } from "react";

// Общий полноэкранный режим карт: нативный API и запасной режим внутри окна браузера.
export function useMapFullscreen(
  workspaceRef: RefObject<HTMLElement | null>,
  mapRef: RefObject<{ updateSize: () => void } | null>,
) {
  const [fullscreen, setFullscreen] = useState(false);
  const resize = useCallback(() => {
    window.requestAnimationFrame(() => mapRef.current?.updateSize());
  }, [mapRef]);

  useEffect(() => {
    const onFullscreenChange = () => {
      setFullscreen(document.fullscreenElement === workspaceRef.current);
      resize();
    };
    document.addEventListener("fullscreenchange", onFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", onFullscreenChange);
  }, [workspaceRef, resize]);

  useEffect(() => {
    if (!fullscreen || document.fullscreenElement === workspaceRef.current) return;
    const previousOverflow = document.body.style.overflow;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      setFullscreen(false);
      resize();
    };
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [fullscreen, workspaceRef, resize]);

  const toggleFullscreen = useCallback(async () => {
    const workspace = workspaceRef.current;
    if (!workspace) return;
    if (fullscreen && document.fullscreenElement !== workspace) {
      setFullscreen(false);
      resize();
      return;
    }
    try {
      if (document.fullscreenElement === workspace) {
        await document.exitFullscreen();
        return;
      }
      if (!workspace.requestFullscreen) {
        setFullscreen(true);
        resize();
        return;
      }
      if (document.fullscreenElement) await document.exitFullscreen();
      await workspace.requestFullscreen();
    } catch {
      setFullscreen(true);
      resize();
    }
  }, [fullscreen, workspaceRef, resize]);

  return { fullscreen, toggleFullscreen };
}
