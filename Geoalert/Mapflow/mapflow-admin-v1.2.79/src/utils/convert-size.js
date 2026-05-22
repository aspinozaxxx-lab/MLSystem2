const SizeUnits = {
    bytes: "bytes",
    kb: "kb",
    mb: "mb",
    gb: "gb",
    tb: "tb",
    pb: "pb",
    eb: "eb",
    zb: "zb",
    yb: "yb",
  };
  
  export function convertSizeUnites(
    size ,
    source,
    target
  ) {
    const keys = Object.keys(SizeUnits);
    const i = keys.indexOf(source);
    const j = keys.indexOf(target);
  
    const sizeInBytes = size * Math.pow(1024, i);
  
    return sizeInBytes / Math.pow(1024, j);
  }
  