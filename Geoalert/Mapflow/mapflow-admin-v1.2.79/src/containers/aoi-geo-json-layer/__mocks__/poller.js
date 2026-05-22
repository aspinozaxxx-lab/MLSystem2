export const abortMock = jest.fn();

function abortableFetch(request, opts) {
  return {
    abort: abortMock,
    ready: Promise.resolve({
      json() {
        return { data: { aoiLayer: "{}" } };
      },
    }),
  };
}

export function createPoller(
  request,
  args,
  onSuccess,
  onError,
  onFirstSuccess,
) {
  let interval,
    ready,
    abort,
    pollTimer,
    currentArgs = args;
  let doRequest = async () => {
    ({ abort, ready } = abortableFetch(
      request(currentArgs),
      onSuccess,
      onError,
    ));
    return ready
      .then((response) => response.json())
      .then(onSuccess)
      .catch((error) => {
        console.error(error);
        onError(error);
      });
  };
  return {
    startPolling(pollInterval = 3000) {
      interval = pollInterval;
      doRequest().then(() => {
        onFirstSuccess();
        pollTimer = setInterval(doRequest, interval);
      });
    },
    clear() {
      abort();
      clearInterval(pollTimer);
    },
    runImmediate(args, clear = false) {
      if (clear) this.clear();
      currentArgs = args;
      doRequest();
      pollTimer = setInterval(doRequest, interval);
    },
  };
}

export const createPostRequest = (url) => (data) => {
  const Authorization = localStorage.getItem("token");
  const requestHeaders = new Headers();
  requestHeaders.append("Content-Type", "application/json");
  requestHeaders.append("Authorization", Authorization);
  return new Request(url, {
    method: "POST",
    credentials: "same-origin",
    headers: requestHeaders,
    body: JSON.stringify(data),
  });
};
