// const maxProfitNTimes = (prices) => {
//     let profit = 0;
//     for (let i = 1; i < prices.length; i++) {
//         if (prices[i] > prices[i - 1])
//             profit += prices[i] - prices[i - 1];
//     }
//     return profit;
// }

// const maxProfitOnlyOnce = (prices) => {
//     let minPrice = Infinity;
//     let maxProfit = 0;
//     for (let i = 0; i < prices.length; i++) {
//         minPrice = Math.min(minPrice, prices[i]);
//         maxProfit = Math.max(maxProfit, prices[i] - minPrice);

//     }
//     return maxProfit;
// }

const maxProfitNTimes = (prices) => {
    let profit = 0;
    for (let i = 1; i < prices.length; i++) {
        if (prices[i] - prices[i - 1] > 0)
            profit += prices[i] - prices[i - 1];
    }
    return profit;
}

const maxProfitOnlyOnce = (prices) => {
    let maxProfit = -Infinity;
    let minPrice = Infinity;
    for (let i = 0; i < prices.length; i++) {
        minPrice = Math.min(minPrice, prices[i]);
        maxProfit = Math.max(maxProfit, prices[i] - minPrice);
    }
    return maxProfit;
}

// Test cases
console.log(maxProfitOnlyOnce([1, 2, 4, 2, 4, 6, 7, 8, 9, 1]));
console.log(maxProfitNTimes([4, 5, 9])); // 1
