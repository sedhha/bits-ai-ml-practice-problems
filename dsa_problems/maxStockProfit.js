function maxProfit(prices) {
    let profit = 0;
    for (let i = 1; i < prices.length; i++) {
        if (prices[i] > prices[i - 1]) {
            profit += prices[i] - prices[i - 1]; // Buy and sell immediately
        }
    }
    return profit;
}

// Example Usage
let prices = [7, 1, 5, 3, 6, 4];
console.log(maxProfit(prices)); // Output: 7

prices = [1, 2, 3, 4, 5];
console.log(maxProfit(prices)); // Output: 4

prices = [7, 6, 4, 3, 1];
console.log(maxProfit(prices)); // Output: 0

// Cases to cover - 
/*
    Buy once sell once
    Buy multiple times and sell multiple times
*/
// [2,1,6]