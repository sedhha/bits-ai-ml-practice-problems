// class LRUCache {
//     constructor(capacity) {
//         this.capacity = capacity;
//         this.head = new Node(null, null); // Dummy nodes for avoiding edge cases
//         this.tail = new Node(null, null); // Dummy nodes for avoiding edge cases
//         this.cache = new Map();
//         this.head.next = this.tail;
//         this.tail.prev = this.head; // Head <-> Tail
//     }

//     _moveToHead(node) {
//         // Head <-> A <-> B <-> C <-> Tail
//         let prev = node.prev;
//         let next = node.next;
//         // Head <-> A <-> C <-> Tail
//         prev.next = next;
//         next.prev = prev;

//         let current_next = this.head.next;
//         node.next = current_next;
//         current_next.prev = node;
//         // Head | B <-> A <-> C <-> Tail

//         this.head.next = node;
//         node.prev = this.head;
//         return node;
//     }

//     _addNode(node) {
//         // Head <-> A <-> C <-> Tail
//         let current_next = this.head.next;
//         node.next = current_next;
//         // Head B <-> A <-> C <-> Tail
//         //          
//         current_next.prev = node;
//         node.prev = this.head;
//         this.head.next = node;
//     }

//     _popTail() {
//         // Head <-> A <-> C <-> B <-> Tail
//         const node = this.tail.prev;
//         const remaining = node.prev;
//         remaining.next = this.tail;
//         this.tail.prev = remaining;
//         // Head <-> A <-> C <-> Tail
//         return node.key;
//     }

//     get(key) {
//         // Check if the key exists and if it does get the node return the value and update the head to keep this node
//         const node = this.cache.get(key);
//         if (!node) return null;
//         this._moveToHead(node);
//         return node.value;
//     }

//     put(key, value) {
//         // Check if the key exists
//         // If it exists then update the node with new value and move it to head
//         // If it does not then create a new node while checking capacity and call the method add Node to Head
//         const node = this.cache.get(key);
//         if (node !== undefined) {
//             node.value = value;
//             this._moveToHead(node);
//             return node.value;
//         }
//         else {
//             const newNode = new Node(key, value);

//             this._addNode(newNode);
//             this.cache.set(key, newNode);

//             if (this.cache.size > this.capacity) {
//                 const key = this._popTail();
//                 this.cache.delete(key);
//             }

//             return node.value;

//         }
//     }
// }

class LRUCache {
    constructor(capacity) {
        this.capacity = capacity;
        this.head = new Node();
        this.tail = new Node();
        this.head.next = this.tail;
        this.tail.prev = this.head;
        this.cache = new Map();
    }

    _moveToHead(node) {
        let nextRef = node.next;
        let prevRef = node.prev;
        let headNext = this.head.next;
        node.next = headNext;
        node.prev = this.head;
        this.head.next = node;
        prevRef.next = nextRef;
        nextRef.prev = prevRef;
        return node;
    }

    _popTail() {
        let popTailRef = this.tail.prev;
        let prev = popTailRef.prev;
        prev.next = this.tail;
        this.tail.prev = prev;
        return popTailRef;
    }

    get(key) {
        // get the node
        const node = this.cache.get(key);
        if (!node) return null;
        // move node to the head
        // Head <-> ... <-> N <-> ... <-> Tail
        return this._moveToHead(node);
    }
    put(key, value) {
        if (this.cache.has(key)) {
            // update the value
            const node = this.cache.get(key);
            node.value = value;
            return this._moveToHead(node);
        }
        else {
            // Insert node and then evict based on capacity
            const newNode = new Node(key, value);
            let headNext = this.head.next;
            newNode.next = headNext;
            newNode.prev = this.head;
            this.head.next = newNode;
            this.cache.set(key, newNode);
            if (this.cache.size > this.capacity) {
                return this._popTail();
            }
        }
    }

}

// Update pointers of surrounding nodes first (usually the nodes you are inserting between)
// THEN update your own node pointers to point to those nodes.