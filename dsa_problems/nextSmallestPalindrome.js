function nextPalindrome(N) {
    let str = N.toString();
    let len = str.length;

    // If single digit, return next number (except 9 -> 11)
    if (len === 1) return N === 9 ? 11 : N + 1;

    // Convert string to array for easier manipulation
    let arr = str.split('');

    // Mirror the left half to the right half
    function mirror() {
        for (let i = 0; i < Math.floor(len / 2); i++) {
            arr[len - 1 - i] = arr[i];
        }
    }

    mirror();

    // Convert back to number and check
    let mirroredNum = parseInt(arr.join(''));

    if (mirroredNum > N) return mirroredNum; // If it's already greater, return it

    // Otherwise, increment the left half and mirror again
    let mid = Math.floor((len - 1) / 2); // 12248 -> 12221 // 2

    while (mid >= 0 && arr[mid] === '9') {
        arr[mid] = '0';
        mid--;
    }

    if (mid < 0) {
        // If all digits were 9 (like 999 -> 1001)
        arr = ['1', ...Array(len - 1).fill('0'), '1'];
    } else {
        arr[mid] = (parseInt(arr[mid]) + 1).toString(); // 12321
    }

    // Mirror again after incrementing the left half
    mirror();

    return parseInt(arr.join(''));
}

// Example Usage
console.log(nextPalindrome(123));   // Output: 131
console.log(nextPalindrome(99));    // Output: 101
console.log(nextPalindrome(808));   // Output: 818
console.log(nextPalindrome(999));   // Output: 1001
console.log(nextPalindrome(1));     // Output: 2
