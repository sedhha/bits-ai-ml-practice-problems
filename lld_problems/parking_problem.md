# High-Level Design (HLD) for Online Parking Portal

## 1. High-Level Overview

The objective of this system is to design an online parking portal that allows users to search for available parking spaces, book a spot, make payments, and manage reservations. The design optimizes for efficient database queries while ensuring a seamless user experience.

## 2. Requirements

### 2.1 Functional Requirements

- Users should be able to register and log in.
- Users can search for available parking spots based on location, time, and vehicle type.
- Users can book a parking spot for a specific duration.
- Users can cancel or modify their reservations.
- Parking lot owners can list available spaces.
- Payment processing for reservations.
- Notification system for booking confirmations and reminders.

### 2.2 Non-Functional Requirements

- **Scalability**: System should handle peak demand efficiently.
- **Low Latency**: Optimized database queries for quick retrieval.
- **Reliability**: Ensure high availability of booking service.
- **Security**: Protect user data and transactions.
- **Data Consistency**: Ensure correctness in reservation updates.

## 3. API Design

### 3.1 User Registration API

```http
POST /api/users/register
```

#### Request Body

```json
{
  "name": "John Doe",
  "email": "john.doe@example.com",
  "password": "securepassword"
}
```

#### Response

```json
{
  "user_id": "U12345",
  "message": "User registered successfully."
}
```

### 3.2 Search Parking Spots API

```http
GET /api/parking/search?location=37.7749,-122.4194&start_time=1678905600&end_time=1678912800
```

#### Response

```json
[
  {
    "parking_id": "P123",
    "location": "123 Main St, SF",
    "price_per_hour": 5.0,
    "availability": true
  }
]
```

### 3.3 Book Parking Spot API

```http
POST /api/parking/book
```

#### Request Body

```json
{
  "user_id": "U12345",
  "parking_id": "P123",
  "start_time": 1678905600,
  "end_time": 1678912800
}
```

#### Response

```json
{
  "booking_id": "B789",
  "total_price": 10.0,
  "status": "confirmed"
}
```

### 3.4 Cancel Booking API

```http
POST /api/parking/cancel
```

#### Request Body

```json
{
  "booking_id": "B789"
}
```

#### Response

```json
{
  "message": "Booking cancelled successfully."
}
```

## 4. Database Design

### 4.1 Users Table

```sql
CREATE TABLE Users (
    user_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    email VARCHAR UNIQUE NOT NULL,
    password_hash VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 4.2 Parking Lots Table

```sql
CREATE TABLE ParkingLots (
    parking_id VARCHAR PRIMARY KEY,
    owner_id VARCHAR REFERENCES Users(user_id),
    location GEOMETRY,
    price_per_hour DECIMAL(5,2),
    total_spots INT,
    available_spots INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 4.3 Bookings Table

```sql
CREATE TABLE Bookings (
    booking_id VARCHAR PRIMARY KEY,
    user_id VARCHAR REFERENCES Users(user_id),
    parking_id VARCHAR REFERENCES ParkingLots(parking_id),
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    total_price DECIMAL(7,2),
    status ENUM('confirmed', 'cancelled') DEFAULT 'confirmed',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 4.4 Optimizations for Query Performance

- **Indexing:**
  - Index on `location` in `ParkingLots` for fast geospatial searches.
  - Index on `start_time` and `end_time` in `Bookings` for quick availability checks.
- **Partitioning:**
  - Partition `Bookings` table by `start_time` to speed up historical data lookups.
- **Caching:**
  - Use Redis for storing frequently searched parking spots.
- **Read Replicas:**
  - Use read replicas to distribute read queries and reduce load on the primary database.

## 5. System Workflow

1. User registers and logs in.
2. User searches for available parking spots in a given location and time frame.
3. The system retrieves results from the database and applies availability filters.
4. User selects a parking spot and books it.
5. The system updates the database, deducting available spots and confirming the reservation.
6. The user is charged, and a confirmation notification is sent.
7. The booking details are accessible via API for future reference or cancellation.

## 6. Conclusion

This HLD covers the core functionalities of an online parking portal with an emphasis on API design and database optimization. The system is designed to efficiently handle queries, ensuring a smooth experience for users and parking lot owners. Future improvements can include predictive analytics for demand forecasting and dynamic pricing strategies.
