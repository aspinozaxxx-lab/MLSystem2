const regex = /[A-Za-z_.\-$]/;

/**
 * @param {String} str
 * @param {Number} pos
 */

export function getWordAt(str, pos) {
  // check ranges
  if (pos < 0 || pos > str.length) {
    return "";
  }
  // Perform type conversions.
  str = String(str);
  pos = Number(pos) >>> 0;

  let l = pos;
  for (let i = pos; i > 0; i--) {
    if (regex.test(str[i])) {
      l--;
    } else {
      // After invalid check, increase for valid characters, becouse slice left is inclusive.
      l++;
      break;
    }
  }

  let r = pos;

  for (let i = pos; i < str.length; i++) {
    if (regex.test(str[i])) {
      r++;
    } else {
      // Dont decrement, becouse right is non-inclusive
      break;
    }
  }

  return str.slice(l, r);
}
