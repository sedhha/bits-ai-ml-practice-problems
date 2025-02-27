# Low-Level Design (LLD) for Hotel Reservation System

## 1. Objective

Build a system for users to search, book, and manage hotel room reservations.

## 2. Key Considerations

- **Room Availability Management:** Track real-time availability.
- **Booking Workflows:** Seamless reservation and confirmation.
- **Pricing Models:** Dynamic pricing based on demand and seasons.
- **Cancellation Policies:** Manage refunds and penalty rules.

## 3. Class Design

### 3.1 Key Classes

```java
class Hotel {
    private String hotelId;
    private String name;
    private String location;
    private List<Room> rooms;

    public List<Room> getAvailableRooms(Date checkIn, Date checkOut);
}

class Room {
    private String roomId;
    private RoomType type;
    private double pricePerNight;
    private boolean isAvailable;

    public boolean checkAvailability(Date checkIn, Date checkOut);
}

enum RoomType {
    SINGLE, DOUBLE, SUITE
}

class User {
    private String userId;
    private String name;
    private String email;
    private List<Booking> bookingHistory;
}

class Booking {
    private String bookingId;
    private User user;
    private Hotel hotel;
    private Room room;
    private Date checkIn;
    private Date checkOut;
    private BookingStatus status;
    private Payment payment;

    public boolean cancelBooking();
}

enum BookingStatus {
    CONFIRMED, CANCELLED, CHECKED_IN, CHECKED_OUT
}

class Payment {
    private String paymentId;
    private double amount;
    private PaymentStatus status;

    public boolean processPayment();
}

enum PaymentStatus {
    PENDING, COMPLETED, FAILED
}
```

## 4. Database Schema

### 4.1 Hotels Table

```sql
CREATE TABLE Hotels (
    hotel_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    location VARCHAR NOT NULL
);
```

### 4.2 Rooms Table

```sql
CREATE TABLE Rooms (
    room_id VARCHAR PRIMARY KEY,
    hotel_id VARCHAR REFERENCES Hotels(hotel_id),
    type ENUM('SINGLE', 'DOUBLE', 'SUITE'),
    price_per_night DECIMAL(7,2),
    is_available BOOLEAN DEFAULT TRUE
);
```

### 4.3 Users Table

```sql
CREATE TABLE Users (
    user_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    email VARCHAR UNIQUE NOT NULL
);
```

### 4.4 Bookings Table

```sql
CREATE TABLE Bookings (
    booking_id VARCHAR PRIMARY KEY,
    user_id VARCHAR REFERENCES Users(user_id),
    hotel_id VARCHAR REFERENCES Hotels(hotel_id),
    room_id VARCHAR REFERENCES Rooms(room_id),
    check_in DATE,
    check_out DATE,
    status ENUM('CONFIRMED', 'CANCELLED', 'CHECKED_IN', 'CHECKED_OUT')
);
```

### 4.5 Payments Table

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

1. **User searches** for available rooms based on location, date, and room type.
2. **System fetches available rooms** from the database.
3. **User selects a room** and proceeds with booking.
4. **Payment is processed**, and booking is confirmed.
5. **User receives a booking confirmation** via email.

### 5.2 Cancellation & Refund Policy

1. **User initiates cancellation** from their booking history.
2. **System checks cancellation policies** (e.g., full refund before 24 hrs, partial refund later).
3. **Refund is processed**, and booking status is updated to CANCELLED.

## 6. Optimizations and Trade-offs

- **Efficient Room Allocation:** Optimize searching with indexing on `room_id` and `availability`.
- **Scalability:** Implement caching for frequently accessed hotel data.
- **Load Balancing:** Distribute search requests across multiple servers.
- **Security:** Encrypt payment transactions and enforce user authentication.

## 7. Conclusion

This LLD provides a structured approach to implementing a **Hotel Reservation System**, covering class design, database schema, and system workflows. The design ensures efficient room management, seamless bookings, and optimized user experience.
