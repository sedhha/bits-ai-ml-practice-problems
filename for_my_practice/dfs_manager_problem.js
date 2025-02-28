// const countContributors = (list, manager) => {
//     const map = {};
//     for (const [contributor, manager] of list) {
//         if (map[manager] === undefined) map[manager] = new Set();
//         map[manager].add(contributor);
//     }

//     const cache = {};
//     const dfs = (manager) => {
//         if (cache[manager] !== undefined) return cache[manager];
//         if (!map[manager]) return 0;
//         const contributors = map[manager] ?? { size: 0 };
//         let contributorsCount = contributors.size;
//         for (const contributor of contributors) {
//             // console.log("Contributors", contributor);
//             const hierarchialContributors = dfs(contributor);
//             contributorsCount += hierarchialContributors;
//         }
//         cache[manager] = contributorsCount;
//         return contributorsCount;
//     }
//     return dfs(manager);
// }

// const countContributors = (list, manager) => {
//     const map = {};

//     // Build manager-to-contributors map
//     for (const [contributor, mgr] of list) { // ✅ Fixed variable name
//         if (!map[mgr]) map[mgr] = new Set();
//         map[mgr].add(contributor);
//     }

//     const dfs = (mgr, memo = {}) => {
//         if (!map[mgr]) return 0;
//         if (mgr in memo) return memo[mgr]; // ✅ Correct memoization check

//         let result = 0; // ✅ Start from 0 (no double counting)
//         for (const member of map[mgr]) {
//             result += 1 + dfs(member, memo); // ✅ Add direct + indirect contributors
//         }

//         memo[mgr] = result;
//         return result;
//     };

//     return dfs(manager);
// };

const countContributors = (list, manager) => {
    const managerMap = {};
    for (const [contributor, owner] of list) {
        if (managerMap[owner] === undefined) managerMap[owner] = new Set();
        managerMap[owner].add(contributor);
    }
    const dfs = (mgr, memo = {}) => {
        if (memo[mgr]) return memo[mgr];
        else if (managerMap[mgr] === undefined) return 0;

        const contributors = managerMap[mgr];
        let count = contributors.size;
        console.log(contributors);
        for (const contributor of contributors) {
            count += dfs(contributor, memo);
        }
        memo[mgr] = count;
        return count;
    }
    return dfs(manager);
}

const pairs = [
    ["A", "B"],
    ["C", "B"],
    ["B", "D"],
    ["E", "D"],
    ["D", "F"],
    ["G", "F"],
    ["H", "D"],
];

console.log(countContributors(pairs, "D")); // Output: 5 (A, C, B, E, G)
console.log(countContributors(pairs, "F")); // Output: 7 (A, C, B, E, D, G, H)
console.log(countContributors(pairs, "B")); // Output: 2 (A, C)
console.log(countContributors(pairs, "A")); // Output: 0 (A is not a manager)