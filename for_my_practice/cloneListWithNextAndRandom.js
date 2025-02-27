

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

// 1 -> 2 -> 3
// 