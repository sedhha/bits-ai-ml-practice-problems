class TreeNode {
    constructor(val, left = null, right = null) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

function buildTree(preorder, inorder) {
    let inorderIndexMap = new Map();

    // Build hashmap of inorder indexes for O(1) lookups
    inorder.forEach((val, idx) => inorderIndexMap.set(val, idx));

    let preIndex = 0; // Pointer for Preorder traversal

    function construct(left, right) {
        if (left > right) return null; // Base case: No elements to construct

        let rootVal = preorder[preIndex++];
        let root = new TreeNode(rootVal);

        let inorderIndex = inorderIndexMap.get(rootVal);

        // Recursively build left and right subtrees
        root.left = construct(left, inorderIndex - 1);
        root.right = construct(inorderIndex + 1, right);

        return root;
    }

    return construct(0, inorder.length - 1);
}

// Helper function to print tree in level order
function levelOrderTraversal(root) {
    if (!root) return [];
    let queue = [root], result = [];
    while (queue.length) {
        let node = queue.shift();
        result.push(node.val);
        if (node.left) queue.push(node.left);
        if (node.right) queue.push(node.right);
    }
    return result;
}

// Example Usage
let preorder = [3, 9, 20, 15, 7];
let inorder = [9, 3, 15, 20, 7];

let root = buildTree(preorder, inorder);
console.log(levelOrderTraversal(root)); // Output: [3, 9, 20, 15, 7]
