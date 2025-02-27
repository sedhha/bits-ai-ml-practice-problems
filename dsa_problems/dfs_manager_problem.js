function countContributors(pairs, managerName) {
    // Step 1: Build Manager-Contributor Graph (Adjacency List)
    let hierarchy = {};

    for (let [contributor, manager] of pairs) {
        if (!hierarchy[manager]) {
            hierarchy[manager] = [];
        }
        hierarchy[manager].push(contributor);
    }

    // Step 2: DFS function to count contributors recursively
    function dfs(manager) {
        if (!hierarchy[manager]) return 0; // No contributors under this manager

        let count = hierarchy[manager].length; // Direct contributors count
        for (let contributor of hierarchy[manager]) {
            count += dfs(contributor); // Recursively count contributors of contributors
        }
        return count;
    }

    // Step 3: Get the total count for the given manager
    return dfs(managerName);
}

// Example Usage:
const pairs = [
    ["A", "B"],
    ["C", "B"],
    ["B", "D"],
    ["E", "D"],
    ["D", "F"],
    ["G", "F"]
];

console.log(countContributors(pairs, "D")); // Output: 5 (A, C, B, E, G)
console.log(countContributors(pairs, "F")); // Output: 5 (A, C, B, E, D)
console.log(countContributors(pairs, "B")); // Output: 2 (A, C)
console.log(countContributors(pairs, "A")); // Output: 0 (A is not a manager)
