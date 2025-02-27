function minWindow(s, t) {
    if (t.length > s.length) return '';

    let charCount = new Map();
    for (let char of t) {
        charCount.set(char, (charCount.get(char) || 0) + 1);
    }

    let left = 0,
        right = 0;
    let required = charCount.size; // Number of unique chars we need to match
    let formed = 0; // Number of unique chars matched
    let windowCount = new Map();

    let minLen = Infinity,
        minLeft = 0,
        minRight = 0;

    while (right < s.length) {
        let char = s[right];
        windowCount.set(char, (windowCount.get(char) || 0) + 1);

        if (charCount.has(char) && windowCount.get(char) === charCount.get(char)) {
            formed++;
        }

        while (formed === required) {
            if (right - left + 1 < minLen) {
                minLen = right - left + 1;
                minLeft = left;
                minRight = right;
            }

            let leftChar = s[left];
            windowCount.set(leftChar, windowCount.get(leftChar) - 1);

            if (
                charCount.has(leftChar) &&
                windowCount.get(leftChar) < charCount.get(leftChar)
            ) {
                formed--;
            }
            left++;
        }

        right++;
    }

    return minLen === Infinity ? '' : s.substring(minLeft, minRight + 1);
}

console.log(minWindow("ADOBECODEBANC", "ABC")); // Output: "BANC"
console.log(minWindow("a", "a"));             // Output: "a"
console.log(minWindow("a", "aa"));            // Output: ""