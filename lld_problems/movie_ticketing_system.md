# Low-Level Design (LLD) for Movie Ticket Booking System

## 1. Objective

Develop a system for users to browse movies, select showtimes, and book tickets.

## 2. Key Considerations

- Object-oriented principles for scalability and maintainability.
- Seat selection and theater layouts.
- Booking confirmation and ticket generation.
- Payment processing and transaction handling.

## 3. Class Design

### 3.1 Key Classes

```java
class Movie {
    private String movieId;
    private String title;
    private String genre;
    private int duration; // in minutes
    private List<Showtime> showtimes;
}

class Theater {
    private String theaterId;
    private String name;
    private String location;
    private List<Screen> screens;
}

class Screen {
    private String screenId;
    private int totalSeats;
    private List<Seat> seats;
    private List<Showtime> showtimes;
}

class Seat {
    private String seatId;
    private boolean isAvailable;
    private SeatType type;
}

enum SeatType {
    REGULAR, PREMIUM, VIP
}

class Showtime {
    private String showtimeId;
    private Movie movie;
    private Theater theater;
    private Screen screen;
    private LocalDateTime showTime;
    private List<Seat> availableSeats;
}

class Booking {
    private String bookingId;
    private User user;
    private Showtime showtime;
    private List<Seat> selectedSeats;
    private Payment payment;

    public void confirmBooking();
}

class User {
    private String userId;
    private String name;
    private String email;
    private List<Booking> bookingHistory;
}

class Payment {
    private String paymentId;
    private PaymentStatus status;
    private double amount;

    public boolean processPayment();
}

enum PaymentStatus {
    PENDING, COMPLETED, FAILED;
}
```

## 4. Database Schema

### 4.1 Movies Table

```sql
CREATE TABLE Movies (
    movie_id VARCHAR PRIMARY KEY,
    title VARCHAR NOT NULL,
    genre VARCHAR NOT NULL,
    duration INT NOT NULL
);
```

### 4.2 Theaters Table

```sql
CREATE TABLE Theaters (
    theater_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    location VARCHAR NOT NULL
);
```

### 4.3 Screens Table

```sql
CREATE TABLE Screens (
    screen_id VARCHAR PRIMARY KEY,
    theater_id VARCHAR REFERENCES Theaters(theater_id),
    total_seats INT NOT NULL
);
```

### 4.4 Seats Table

```sql
CREATE TABLE Seats (
    seat_id VARCHAR PRIMARY KEY,
    screen_id VARCHAR REFERENCES Screens(screen_id),
    is_available BOOLEAN DEFAULT TRUE,
    seat_type ENUM('REGULAR', 'PREMIUM', 'VIP')
);
```

### 4.5 Showtimes Table

```sql
CREATE TABLE Showtimes (
    showtime_id VARCHAR PRIMARY KEY,
    movie_id VARCHAR REFERENCES Movies(movie_id),
    theater_id VARCHAR REFERENCES Theaters(theater_id),
    screen_id VARCHAR REFERENCES Screens(screen_id),
    show_time TIMESTAMP
);
```

### 4.6 Bookings Table

```sql
CREATE TABLE Bookings (
    booking_id VARCHAR PRIMARY KEY,
    user_id VARCHAR REFERENCES Users(user_id),
    showtime_id VARCHAR REFERENCES Showtimes(showtime_id),
    total_amount DECIMAL(7,2)
);
```

### 4.7 Payments Table

```sql
CREATE TABLE Payments (
    payment_id VARCHAR PRIMARY KEY,
    booking_id VARCHAR REFERENCES Bookings(booking_id),
    amount DECIMAL(7,2),
    status ENUM('PENDING', 'COMPLETED', 'FAILED')
);
```

## 5. System Workflow

### 5.1 Booking Process

1. User **browses available movies**.
2. User **selects a showtime** and a preferred theater.
3. User **chooses available seats** from the theater layout.
4. System **reserves selected seats** for a limited time.
5. User **proceeds to payment**.
6. Payment is **processed and confirmed**.
7. A **booking confirmation and e-ticket** are generated.

### 5.2 Cancellation & Refund

1. User requests **cancellation before showtime**.
2. System **validates cancellation policy**.
3. If eligible, **refund is processed**.
4. Seats are marked **available** again.

## 6. Optimizations and Trade-offs

- **Efficient Seat Selection:** Implement **row-based indexing** for fast lookups.
- **Concurrency Handling:** Use **database locks** or **optimistic locking** to prevent double booking.
- **Caching Popular Movies:** Use **Redis** to store frequently accessed movie listings.
- **Load Balancing:** Distribute API requests across multiple servers for scalability.

## 7. Conclusion

This LLD provides a structured approach to implementing a **Movie Ticket Booking System**, covering class design, database schema, workflows, and optimizations. It ensures an efficient, scalable, and user-friendly experience for managing movie bookings.
