const generateSubArray = (arr, start = 0, memo = {}) => {
    if (start >= arr.length) return [];
    if (memo[start] !== undefined) return memo[start];

    let result = [];

    for (let end = start; end < arr.length; end++) {
        result.push(arr.slice(start, end + 1)); // Fix: Include `end` in slice
    }

    result = [...result, ...generateSubArray(arr, start + 1, memo)]; // Correct merging
    memo[start] = result;

    return result;
};


const generateSubsequences = (arr, start = 0, memo = {}) => {
    if (start >= arr.length) return [[]];
    if (start in memo) return memo[start];
    let result = [];

    let withoutStart = generateSubsequences(arr, start + 1, memo);

    let withStart = withoutStart.map(subseq => ([...subseq, arr[start]]));
    result = [...withStart, ...withoutStart];
    memo[start] = result;
    return result;
}

console.log('Generating all possible', generateSubArray([1, 2, 3, 4]));
console.log('Generating all possible', generateSubsequences([1, 2, 3, 4]));
// console.log('Generating all possible', allSubsets([1, 2, 3, 4]));

