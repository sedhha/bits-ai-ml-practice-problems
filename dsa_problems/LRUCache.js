class LRUCache {
    constructor(capacity) {
        this.capacity = capacity;
        this.cache = new Map();
        this.head = new Node(null, null); // Dummy head
        this.tail = new Node(null, null); // Dummy tail
        this.head.next = this.tail;
        this.tail.prev = this.head;
    }

    // Add node right after head
    _addNode(node) {
        node.prev = this.head; // {{3,1,{2}}}}
        node.next = this.head.next; // 
        this.head.next.prev = node;
        this.head.next = node;
    }

    // Remove an existing node from the linked list
    _removeNode(node) {
        let prev = node.prev;
        let next = node.next;
        prev.next = next;
        next.prev = prev;
    }

    // Move a node to the head (mark as recently used)
    _moveToHead(node) {
        this._removeNode(node);
        this._addNode(node);
    }

    // Remove the least recently used node
    _popTail() {
        let res = this.tail.prev;
        this._removeNode(res);
        return res;
    }

    // Get the value of the key if it exists
    get(key) {
        let node = this.cache.get(key);
        if (!node) return -1;
        this._moveToHead(node);
        return node.value;
    }

    // Add or update a value by key
    put(key, value) {
        let node = this.cache.get(key);
        if (node) {
            node.value = value;
            this._moveToHead(node);
        } else {
            let newNode = new Node(key, value);
            this.cache.set(key, newNode);
            this._addNode(newNode);

            if (this.cache.size > this.capacity) {
                let tail = this._popTail();
                this.cache.delete(tail.key);
            }
        }
    }
}
