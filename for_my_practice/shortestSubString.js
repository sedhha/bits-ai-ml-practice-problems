// const minWindow = (string, substring) => {
//     if (substring.length > string.length) return "";
//     const subStringMap = {};

//     let required = 0;

//     for (let i = 0; i < substring.length; i++) {
//         const exists = subStringMap[substring[i]] !== undefined;
//         if (!exists) required++;
//         subStringMap[substring[i]] = (subStringMap[substring[i]] ?? 0) + 1;
//     }


//     const windowMap = {};
//     let minLeft = 0, minRight = 0, left = 0, right = 0, formed = 0, minLength = Infinity;

//     while (right < string.length) {
//         const char = string[right];
//         windowMap[char] = (windowMap[char] ?? 0) + 1;

//         if (subStringMap[char] !== undefined && subStringMap[char] === windowMap[char]) {
//             formed++;
//         }

//         while (formed === required) {
//             if (minLength > right - left + 1) {
//                 minLength = right - left + 1;
//                 minLeft = left;
//                 minRight = right;
//             }
//             const curr = string[left];
//             windowMap[curr] = (windowMap[curr] ?? 0) - 1;

//             if (subStringMap[curr] !== undefined && subStringMap[curr] > windowMap[curr])
//                 formed--;
//             left++;
//         }
//         right++;
//     }
//     return minLength === Infinity ? '' : string.substring(minLeft, minRight + 1);
// }

// const minWindow = (string, substring) => {
//     if (substring.length > string.length) return '';
//     let formed = 0, required = 0;
//     const subStringMap = {};
//     for (let i of substring) {
//         const exists = subStringMap[i] !== undefined;
//         if (!exists) required++;
//         subStringMap[i] = (subStringMap[i] ?? 0) + 1;
//     }

//     let left = 0, right = 0, leftMin = 0, rightMin = 0;
//     let minLength = Infinity;
//     const windowMap = {};

//     while (right < string.length) {
//         const char = string[right];
//         windowMap[char] = (windowMap[char] ?? 0) + 1;
//         if (subStringMap[char] !== undefined && windowMap[char] === subStringMap[char])
//             formed++;

//         while (formed === required) {
//             if (right - left + 1 < minLength) {
//                 minLength = right - left + 1;
//                 leftMin = left;
//                 rightMin = right;
//             }
//             const char = string[left];
//             windowMap[char] = (windowMap[char] ?? 0) - 1;
//             if (subStringMap[char] !== undefined && windowMap[char] < subStringMap[char]) {
//                 formed--;
//             }
//             left++;
//         }
//         right++;
//     }
//     return minLength === Infinity ? '' : string.slice(leftMin, rightMin + 1);
// }

const minWindow = (string, substring) => {
    if (substring.length > string.length) return '';
    const subStringMap = {};
    let required = 0;
    for (const char of substring) {
        const exists = subStringMap[char] !== undefined;
        if (!exists) {
            required++;
            subStringMap[char] = 0;
        }
        subStringMap[char] += 1;
    }
    const windowMap = {};
    let left = 0, right = 0, minLeft = 0, minRight = 0, minLength = Infinity;
    let formed = 0;
    while (right < string.length) {
        const char = string[right];
        windowMap[char] = (windowMap[char] ?? 0) + 1;

        if (subStringMap[char] !== undefined && subStringMap[char] === windowMap[char])
            formed++;

        while (formed === required) {
            if (right - left + 1 < minLength) {
                minLength = right - left + 1;
                minLeft = left;
                minRight = right;
            }
            const subChar = string[left];
            windowMap[subChar] = (windowMap[subChar] ?? 0) - 1;
            if (subStringMap[subChar] !== undefined && subStringMap[subChar] > windowMap[subChar])
                formed--;
            left++;
        }
        right++;
    }
    return minLength === Infinity ? '' : string.slice(minLeft, minRight + 1);
}

console.log(`'${minWindow("ADOBECODEBANC", "ABC")}'`); // Output: "BANC"
console.log(`'${minWindow("a", "a")}'`);             // Output: "a"
console.log(`'${minWindow("a", "aa")}'`);            // Output: ""