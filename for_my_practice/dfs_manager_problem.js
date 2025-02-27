const countContributors = (list, manager) => {
    const map = {};
    for (const [contributor, manager] of list) {
        if (map[manager] === undefined) map[manager] = new Set();
        map[manager].add(contributor);
    }

    const cache = {};
    const dfs = (manager) => {
        if (cache[manager] !== undefined) return cache[manager];
        if (!map[manager]) return 0;
        const contributors = map[manager] ?? { size: 0 };
        let contributorsCount = contributors.size;
        for (const contributor of contributors) {
            // console.log("Contributors", contributor);
            const hierarchialContributors = dfs(contributor);
            contributorsCount += hierarchialContributors;
        }
        cache[manager] = contributorsCount;
        return contributorsCount;
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
console.log(countContributors(pairs, "F")); // Output: 5 (A, C, B, E, D)
console.log(countContributors(pairs, "B")); // Output: 2 (A, C)
console.log(countContributors(pairs, "A")); // Output: 0 (A is not a manager)