class Node {
    constructor(val, next = null, random = null) {
        this.val = val;
        this.next = next;
        this.random = random;
    }
}

function copyRandomList(head) {
    if (!head) return null;

    // Step 1: Create interleaved copy nodes
    let curr = head;
    while (curr) {
        let newNode = new Node(curr.val, curr.next);
        curr.next = newNode;
        curr = newNode.next;
    }

    // Step 2: Assign random pointers to the copied nodes
    curr = head;
    while (curr) {
        if (curr.random) {
            curr.next.random = curr.random.next;
        }
        curr = curr.next.next;
    }

    // Step 3: Separate the copied list from the original
    curr = head;
    let newHead = head.next;
    let copyCurr = newHead;

    while (curr) {
        curr.next = curr.next.next;
        copyCurr.next = copyCurr.next ? copyCurr.next.next : null;
        curr = curr.next;
        copyCurr = copyCurr.next;
    }

    return newHead;
}

// Helper Function to Print the List
function printList(head) {
    let result = [];
    let curr = head;
    while (curr) {
        result.push({
            val: curr.val,
            random: curr.random ? curr.random.val : null
        });
        curr = curr.next;
    }
    console.log(result);
}