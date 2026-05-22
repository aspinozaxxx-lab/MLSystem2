function downloadAsFile(data, filename) {
  const a = document.createElement("a");
  document.body.appendChild(a);
  a.style = "display: none";

  return () => {
    const json = JSON.stringify(data),
      blob = new Blob([json], { type: "octet/stream" }),
      url = window.URL.createObjectURL(blob);
    a.href = url;
    a.download = filename;
    a.click();
    window.URL.revokeObjectURL(url);
  };
}

export function useDownloadAsFile({ data, filename, onError } = {}) {
  const download = (_, __) => {
    try {
      downloadAsFile(data || _, filename || __)();
    } catch (error) {
      console.error(error);
      onError(error);
    }
  };
  return download;
}
