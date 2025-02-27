# Low-Level Design (LLD) for Library Management System and Online Bookstore

## 1. Library Management System

### 1.1 Objective

Create a system to manage book inventories, user registrations, and borrowing/returning processes.

### 1.2 Key Considerations

- Object-oriented principles for scalability.
- Managing books, users, and transactions.
- Handling overdue books and fines.

### 1.3 Class Design

```java
class Library {
    private List<Book> books;
    private List<User> users;
    private List<Transaction> transactions;

    public void addBook(Book book);
    public void registerUser(User user);
    public Transaction borrowBook(String userId, String bookId);
    public void returnBook(String transactionId);
}

class Book {
    private String bookId;
    private String title;
    private String author;
    private boolean isAvailable;

    public void markAsBorrowed();
    public void markAsReturned();
}

class User {
    private String userId;
    private String name;
    private List<Transaction> borrowedBooks;

    public boolean canBorrow();
}

class Transaction {
    private String transactionId;
    private String userId;
    private String bookId;
    private LocalDateTime borrowDate;
    private LocalDateTime dueDate;
    private boolean isReturned;

    public void markAsReturned();
    public boolean isOverdue();
}
```

### 1.4 Database Schema

```sql
CREATE TABLE Books (
    book_id VARCHAR PRIMARY KEY,
    title VARCHAR NOT NULL,
    author VARCHAR NOT NULL,
    is_available BOOLEAN DEFAULT TRUE
);

CREATE TABLE Users (
    user_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL
);

CREATE TABLE Transactions (
    transaction_id VARCHAR PRIMARY KEY,
    user_id VARCHAR REFERENCES Users(user_id),
    book_id VARCHAR REFERENCES Books(book_id),
    borrow_date TIMESTAMP,
    due_date TIMESTAMP,
    is_returned BOOLEAN DEFAULT FALSE
);
```

## 2. Online Bookstore

### 2.1 Objective

Build a platform to browse, search, and purchase books online.

### 2.2 Key Considerations

- Catalog management for books.
- Shopping cart functionality.
- Order processing and payment integration.

### 2.3 Class Design

```java
class OnlineBookstore {
    private List<Book> catalog;
    private List<Order> orders;

    public List<Book> searchBook(String keyword);
    public void addToCart(User user, Book book);
    public Order placeOrder(User user);
}

class Book {
    private String bookId;
    private String title;
    private String author;
    private double price;
    private int stock;

    public boolean isAvailable();
    public void reduceStock();
}

class User {
    private String userId;
    private String name;
    private ShoppingCart cart;
}

class ShoppingCart {
    private List<Book> books;

    public void addBook(Book book);
    public void removeBook(Book book);
    public double calculateTotal();
}

class Order {
    private String orderId;
    private User user;
    private List<Book> orderedBooks;
    private double totalAmount;
    private Payment payment;

    public void processOrder();
}

class Payment {
    private String paymentId;
    private String orderId;
    private PaymentStatus status;

    public boolean processPayment(double amount);
}

enum PaymentStatus {
    PENDING, COMPLETED, FAILED;
}
```

### 2.4 Database Schema

```sql
CREATE TABLE Books (
    book_id VARCHAR PRIMARY KEY,
    title VARCHAR NOT NULL,
    author VARCHAR NOT NULL,
    price DECIMAL(7,2),
    stock INT
);

CREATE TABLE Users (
    user_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL
);

CREATE TABLE Orders (
    order_id VARCHAR PRIMARY KEY,
    user_id VARCHAR REFERENCES Users(user_id),
    total_amount DECIMAL(7,2)
);

CREATE TABLE Payments (
    payment_id VARCHAR PRIMARY KEY,
    order_id VARCHAR REFERENCES Orders(order_id),
    status ENUM('PENDING', 'COMPLETED', 'FAILED')
);
```

## 3. System Flow

### 3.1 Library Management System Flow

1. User **registers**.
2. User **searches for a book**.
3. If available, user **borrows the book**.
4. Due date is calculated, and transaction is created.
5. User **returns the book**, and fines are calculated if overdue.

### 3.2 Online Bookstore Flow

1. User **searches for books**.
2. User **adds books to cart**.
3. User **proceeds to checkout**.
4. Payment is processed and order is confirmed.
5. Order is **shipped to the user**.

## 4. Conclusion

This LLD provides a structured way to implement a **Library Management System** and **Online Bookstore**, covering class design, database schema, and system workflows. The design ensures flexibility, scalability, and maintainability for future enhancements.
