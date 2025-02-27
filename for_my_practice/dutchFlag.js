const sort0s1sAnd2s = (arr) => {
    let low = 0, mid = 0, high = arr.length - 1;

    while (mid <= high) {
        if (arr[mid] === 0) {
            [arr[mid], arr[low]] = [arr[low], arr[mid]];
            mid++;
            low++;
        } else if (arr[mid] === 1) {
            mid++;
        } else {
            [arr[mid], arr[high]] = [arr[high], arr[mid]];
            high--;
        }
    }
}

console.log(sortColors([2, 0, 2, 1, 1, 0])); // Output: [0, 0, 1, 1, 2, 2]
console.log(sortColors([0, 1, 2, 0, 1, 2, 1, 0])); // Output: [0, 0, 0, 1, 1, 1, 2, 2]
console.log(sortColors([2, 2, 2, 2])); // Output: [2, 2, 2, 2]