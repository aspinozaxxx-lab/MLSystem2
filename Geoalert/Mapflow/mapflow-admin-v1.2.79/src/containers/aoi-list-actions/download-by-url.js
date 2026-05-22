export const getDownloadLink = (resultId) => {
  return `${window.location.origin}/rest/result?resultId=${resultId}`;
};
export const invokeDownload = (downloadLink, filename) => {
  const link = document.createElement("a");
  link.href = downloadLink;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};
