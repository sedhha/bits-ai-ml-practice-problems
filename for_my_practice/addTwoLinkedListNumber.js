// class ListNode {
//     constructor(val = 0, next = null) {
//         this.val = val;
//         this.next = next;
//     }
// }

// const reverseList = (node) => {
//     let prev = null; curr = node;
//     while (curr) {
//         const nextNode = curr.next;
//         curr.next = prev;
//         prev = curr;
//         curr = nextNode;
//     }
//     return prev;
// }

// const reverseListV2 = (list) => {
//     // 1 -> 2 -> 3 -> null
//     // null <- 1 <- 2 <- 3
//     // null <- 1
//     let prev = null; curr = list;
//     while (curr) {
//         const next = curr.next; // null
//         curr.next = prev; // null <- 1 <- 2 <- 3
//         prev = curr; // null <- 1 <- 2 <- 3
//         curr = next; // null
//     }
//     return prev;
// }

// const addToList = (l1, l2) => {
//     l1 = reverseList(l1);
//     l2 = reverseList(l2);
//     let carry = 0;
//     let dummyHead = new ListNode(0);
//     let curr = dummyHead;

//     while (l1 || l2 || carry) {
//         let sum = (l1 ? l1.val : 0) + (l2 ? l2.val : 0) + carry;
//         carry = Math.floor(sum / 10);
//         let node = new ListNode(sum % 10);
//         curr.next = node;
//         curr = curr.next;
//         if (l1) l1 = l1.next;
//         if (l2) l2 = l2.next;
//     }

//     return reverseList(dummyHead.next);
// }

// const printList = (l1) => {
//     const nums = [];
//     while (l1) {
//         if (l1.val) nums.push(l1.val);
//         l1 = l1.next;
//     }
//     console.log(nums.join('->'));
// }

class ListNode {
    constructor(val = 0, next = null) {
        this.val = val;
        this.next = next;
    }
}

const reverseList = (head) => {
    let prev = null, curr = head;
    while (curr !== null) {
        let reference_to_tail = curr.next;
        curr.next = prev;
        prev = curr;
        curr = reference_to_tail;
    }
    return prev;
}

const addToList = (l1, l2) => {
    // 2 -> 3 -> 4
    // 5 -> 6 -> 7
    // 8 -> 0 -> 1

    l1 = reverseList(l1); // 4 ->3 ->2
    l2 = reverseList(l2); // 7 -> 6 -> 5
    let carry = 0;

    let dummyHead = new ListNode(0);
    let curr = dummyHead;

    while (l1 || l2 || carry) {
        let sum = (l1 ? l1.val : 0) + (l2 ? l2.val : 0) + carry; // 8
        carry = Math.floor(sum / 10); // 0
        curr.next = new ListNode(sum % 10); // 0 -> 1 ->0 -> 8
        curr = curr.next;
        if (l1) l1 = l1.next; //  
        if (l2) l2 = l2.next; //  
    }
    return reverseList(dummyHead.next);
}

const printList = (l1) => {
    const arr = [];
    while (l1 !== null) {
        arr.push(l1.val);
        l1 = l1.next;
    }
    console.log(arr.join('->'));
}

const num1 = new ListNode(2, new ListNode(3, new ListNode(4)));
const num2 = new ListNode(5, new ListNode(6, new ListNode(7)));
const sum = addToList(num1, num2);
printList(sum);



// 0 -> 1 -> 2
//           ^