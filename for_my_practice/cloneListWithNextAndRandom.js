

const cloneLinkedList = (head) => {
    // Step 1 - Add Clone nodes in between the list items
    if (!head) return null;

    let current = head;

    // Step 1: Clone nodes and insert them between original nodes
    while (current !== null) {
        let clone = new Node(current.val); // {1}
        clone.next = current.next; // {1, {2}}
        current.next = clone; // {1, {2, {1, {2}}}}
        current = clone.next; //              ^
    }

    // Step 2: Assign random pointers in the cloned nodes
    current = head;
    while (current !== null) {
        if (current.random) {
            current.next.random = current.random.next;
        }
        current = current.next.next;
    }

    // Step 3: Separate the cloned list from the original
    let original = head;
    let copy = head.next;
    let cloneHead = copy;

    while (original !== null) {
        original.next = original.next.next;
        copy.next = copy.next ? copy.next.next : null;
        original = original.next;
        copy = copy.next;
    }

    return cloneHead; // Return head of cloned list
}

const cloneLinkedListV2 = (head) => {
    /*
        Three Step Procedure:
        1. Insert clone nodes between actual list
        1 -> 2 -> 3
        |_________|

        2. Assign Random pointers in the cloned nodes
    */
    if (!head) return null;
    let iterator = head;


    while (iterator !== null) {
        let clone = new Node(iterator.val);
        clone.next = iterator.next;
        iterator.next = clone;
        iterator = clone.next;
    }

    iterator = head;
    while (iterator !== null) {
        if (iterator.random) {
            iterator.next.random = iterator.random.next;
        }
        iterator = iterator.next.next;
    }

    let dummyHead = new Node(-1);
    iterator = head;
    while (iterator !== null) {
        dummyHead.next = iterator.next;
        iterator.next = iterator.next.next;
        dummyHead = dummyHead.next;
        iterator = iterator.next;
    }
    return dummyHead.next;
}

const cloneListWithNextAndRandomV3 = (head) => {
    let curr = head;
    while (curr !== null) {
        // Clone and create duplicates
        // 1 -> 2 -> 3
        let clone = new Node(curr.val); // 3
        clone.next = curr.next; // 3 -> null
        curr.next = clone; // 1 -> 1 -> 2 -> 3(*) -> 3 -> null
        curr = curr.next.next; // 1 -> 1 -> 2 -> 3 -> 3 -> null(*)
    }

    curr = head;
    while (curr !== null) {
        // random pointers
        if (curr.random) {
            curr.next.random = curr.random.next;
        }
        curr = curr.next.next;
    }

    let clone = new Node(-1);
    let dummyHead = clone;
    curr = head; // 1 -> 1' -> 2 -> 2' -> 3 -> 3'
    while (curr !== null) {
        // -1
        dummyHead.next = curr.next; // -1 -> 1' -> 2'(*) -> 3'
        curr.next = (curr.next ? curr.next.next : null); // 1 -> 2 -> 3 -> null
        curr = curr.next; // 1 -> 2 -> 3(*) -> 3'
        dummyHead = dummyHead.next; // -1 -> 1' -> 2' -> 3'(*)
    }
    return clone.next;
}


const cloneListWithNextAndRandomV4 = (head) => {
    // Step 1: Add duplicate pointers
    let curr = head;
    while (curr !== null) {
        // 1 -> 2 -> 3
        let clone = new Node(curr.val); // 3
        clone.next = curr.next; // 3 -> null
        curr.next = clone; // 1 -> 1' -> 2 -> 2' -> 3 -> 3 -> null*
        curr = curr.next.next;
    }

    curr = head;
    while (curr !== null) {
        if (curr.random) {
            curr.next.random = curr.random.next;
        }
        curr = curr.next.next;
    }

    let clone = new Node(-1);
    dummyHead = clone;
    curr = head;
    while (curr !== null) {
        // 1 -> 2 -> 2' -> 3 -> 3'
        dummyHead.next = curr.next;
        curr.next = (curr.next ? curr.next.next : null);
        curr = curr.next;
        dummyHead = dummyHead.next;
    }
    return clone.next;

}