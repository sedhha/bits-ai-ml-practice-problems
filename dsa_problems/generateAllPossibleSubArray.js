function generateSubarraysRecursive(arr, start = 0, end = 0, memo = {}) {
    let key = `${start},${end}`;
    if (key in memo) return memo[key];

    // Base case: If `start` reaches end of array, return empty array
    if (start >= arr.length) return [];

    let result = [];

    // Generate subarray from start to end
    if (end < arr.length) {
        let subarray = arr.slice(start, end + 1);
        result.push(subarray);

        // Recursively expand the window to include the next element
        result = result.concat(generateSubarraysRecursive(arr, start, end + 1, memo));
    } else {
        // Move `start` index to the right to generate new subarrays
        result = result.concat(generateSubarraysRecursive(arr, start + 1, start + 1, memo));
    }

    memo[key] = result;
    return result;
}

// Wrapper function
function generateAllSubarrays(arr) {
    return generateSubarraysRecursive(arr);
}

// Example Usage
let arr = [1, 2, 3];
console.log(generateAllSubarrays(arr));
