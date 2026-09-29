const RELOAD_KEY = "grovika-pseudo-module-reload";

// Старую вкладку просмотра можно безопасно перечитать: здесь нет несохранённой разметки.
// Редактор и формы не перезагружаем. Ограничение защищает от цикла при сетевом сбое.
export function installPseudoModuleRecovery() {
  const recover = (event: Event) => {
    if (!location.hash.startsWith("#/pseudo-markup/") || !navigator.onLine) return;
    try {
      const now = Date.now();
      const previous = sessionStorage.getItem(RELOAD_KEY);
      if (previous && now - Number(previous) < 60_000) return;
      sessionStorage.setItem(RELOAD_KEY, String(now));
    } catch { return; }
    event.preventDefault();
    location.reload();
  };
  window.addEventListener("vite:preloadError", recover);
  return () => window.removeEventListener("vite:preloadError", recover);
}
