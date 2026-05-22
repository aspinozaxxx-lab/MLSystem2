const padTime = (time) => String(time).padStart(2, "0");

const secondsPerDay = 86400;

export const getProcessETA = (estimate) => {
  const hours = Math.floor(estimate / 3600);
  const minutes = Math.floor((estimate % 3600) / 60);
  const seconds = estimate % 60;

  return `${padTime(hours)}:${padTime(minutes)}:${padTime(seconds)}`;
};

export const getProcessETAWithDate = (estimateInSeconds) => {
  const secondsPerMinute = 60;
  const secondsPerHour = 3600;

  let hours, remainingSeconds;

  if (estimateInSeconds >= secondsPerDay) {
    hours = Math.floor(estimateInSeconds / secondsPerHour);
    remainingSeconds = estimateInSeconds % secondsPerHour;
  } else {
    hours = Math.floor(estimateInSeconds / secondsPerHour) % 24;
    remainingSeconds = estimateInSeconds % secondsPerHour;
  }

  const minutes = Math.floor(remainingSeconds / secondsPerMinute);
  const seconds = remainingSeconds % secondsPerMinute;

  const formattedTime = `${padTime(hours)}:${padTime(minutes)}:${padTime(
    seconds,
  )}`;

  return formattedTime;
};

export const getProcessETAWithCurrentDate = (estimateInSeconds) => {
  const now = new Date(); // Get the current date and time

  // Calculate the future date and time
  const futureTime = new Date(now.getTime() + estimateInSeconds * 1000); // Convert seconds to milliseconds

  // Extract individual components from the future time
  const year = futureTime.getFullYear();
  const month = padTime(futureTime.getMonth() + 1); // Month is zero-based, so add 1
  const day = padTime(futureTime.getDate());
  const hours = padTime(futureTime.getHours());
  const minutes = padTime(futureTime.getMinutes());
  const seconds = padTime(futureTime.getSeconds());

  // Construct the formatted date and time string
  const formattedDateTime = `${day}.${month}.${year}, ${hours}:${minutes}:${seconds}`;
  return formattedDateTime;
};
