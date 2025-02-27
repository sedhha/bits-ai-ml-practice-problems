const nextSmallestPalindrome = (number) => {
    if (number < 9) return number + 1;
    if (number <= 10) return 11;

    let nString = number.toString(); // [1,0,2,4]
    let nArray = nString.split('');

    const mirror = () => {
        for (let i = 0; i < Math.floor(nArray.length / 2); i++) {
            nArray[nArray.length - 1 - i] = nArray[i]; // [1,2,8,7,6] -> 12821
        }
    }

    mirror();
    const mirroredNumber = nArray.join('');
    if (+mirroredNumber > number) return mirroredNumber;

    let mid = Math.floor((nArray.length - 1) / 2); // 2

    while (mid >= 0 && nArray[mid] === '9') {
        nArray[mid--] = '0';
    }

    if (mid > 0) { // 
        nArray[mid] = (+nArray[mid] + 1).toString(); // 12921
        mirror();
    } else {
        console.log('length---', nArray.length);
        nArray = ['1', ...Array(nArray.length - 1).fill(0), '1'];
    }
    return +nArray.join('');
}

























// Test Cases

console.log(`${nextSmallestPalindrome(9)} should be 11`);
console.log(`${nextSmallestPalindrome(10)} should be 11`);
console.log(`${nextSmallestPalindrome(6)} should be 7`);
console.log(`${nextSmallestPalindrome(131)} should be 141`);
console.log(`${nextSmallestPalindrome(128)} should be 131`);
console.log(`${nextSmallestPalindrome(144)} should be 151`);
console.log(`${nextSmallestPalindrome(999)} should be 1001`);
console.log(`${nextSmallestPalindrome(1024)} should be 1111`);
console.log(`${nextSmallestPalindrome(1876)} should be 1881`);
console.log(`${nextSmallestPalindrome(12876)} should be 12921`);
console.log(`${nextSmallestPalindrome(19876)} should be 19891`);










