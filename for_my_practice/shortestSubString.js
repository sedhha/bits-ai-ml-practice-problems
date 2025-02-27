const minWindow = (string, substring) => {
    if (substring.length > string.length) return "";
    const subStringMap = {};

    let required = 0;

    for (let i = 0; i < substring.length; i++) {
        const exists = subStringMap[substring[i]] !== undefined;
        if (!exists) required++;
        subStringMap[substring[i]] = (subStringMap[substring[i]] ?? 0) + 1;
    }


    const windowMap = {};
    let minLeft = 0, minRight = 0, left = 0, right = 0, formed = 0, minLength = Infinity;

    while (right < string.length) {
        const char = string[right];
        windowMap[char] = (windowMap[char] ?? 0) + 1;

        if (subStringMap[char] !== undefined && subStringMap[char] === windowMap[char]) {
            formed++;
        }

        while (formed === required) {
            if (minLength > right - left + 1) {
                minLength = right - left + 1;
                minLeft = left;
                minRight = right;
            }
            const curr = string[left];
            windowMap[curr] = (windowMap[curr] ?? 0) - 1;

            if (subStringMap[curr] !== undefined && subStringMap[curr] > windowMap[curr])
                formed--;
            left++;
        }
        right++;
    }
    return minLength === Infinity ? '' : string.substring(minLeft, minRight + 1);
}

console.log(`'${minWindow("ADOBECODEBANC", "ABC")}'`); // Output: "BANC"
console.log(`'${minWindow("a", "a")}'`);             // Output: "a"
console.log(`'${minWindow("a", "aa")}'`);            // Output: ""