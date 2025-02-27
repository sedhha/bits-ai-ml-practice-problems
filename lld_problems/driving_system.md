# Low-Level Design (LLD) for Ride-Sharing Application

## 1. Objective

Create a platform connecting drivers with passengers requesting rides.

## 2. Key Considerations

- **User Matching Algorithms:** Efficiently match riders with available drivers.
- **Ride Tracking:** Real-time location updates and route optimization.
- **Fare Calculation:** Dynamic pricing based on distance, demand, and time.
- **Payment Processing:** Secure and seamless transactions for riders and drivers.

## 3. Class Design

### 3.1 Key Classes

```java
class RideSharingApp {
    private List<Driver> availableDrivers;
    private List<Ride> activeRides;

    public Ride requestRide(User user, Location pickup, Location destination);
    public boolean completeRide(String rideId);
}

class User {
    private String userId;
    private String name;
    private String phoneNumber;

    public Ride requestRide(Location pickup, Location destination);
}

class Driver {
    private String driverId;
    private String name;
    private Vehicle vehicle;
    private Location currentLocation;
    private boolean isAvailable;

    public void updateLocation(Location location);
    public void acceptRide(Ride ride);
}

class Ride {
    private String rideId;
    private User user;
    private Driver driver;
    private Location pickup;
    private Location destination;
    private RideStatus status;
    private double fare;

    public void startRide();
    public void completeRide();
}

enum RideStatus {
    REQUESTED, ASSIGNED, IN_PROGRESS, COMPLETED, CANCELLED
}

class Vehicle {
    private String vehicleId;
    private String model;
    private String licensePlate;
    private VehicleType type;
}

enum VehicleType {
    SEDAN, SUV, HATCHBACK, MOTORBIKE
}

class Payment {
    private String paymentId;
    private Ride ride;
    private double amount;
    private PaymentStatus status;

    public boolean processPayment();
}

enum PaymentStatus {
    PENDING, COMPLETED, FAILED
}

class Location {
    private double latitude;
    private double longitude;
}
```

## 4. Database Schema

### 4.1 Users Table

```sql
CREATE TABLE Users (
    user_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    phone_number VARCHAR UNIQUE NOT NULL
);
```

### 4.2 Drivers Table

```sql
CREATE TABLE Drivers (
    driver_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    vehicle_id VARCHAR REFERENCES Vehicles(vehicle_id),
    current_lat DOUBLE,
    current_long DOUBLE,
    is_available BOOLEAN DEFAULT TRUE
);
```

### 4.3 Rides Table

```sql
CREATE TABLE Rides (
    ride_id VARCHAR PRIMARY KEY,
    user_id VARCHAR REFERENCES Users(user_id),
    driver_id VARCHAR REFERENCES Drivers(driver_id),
    pickup_lat DOUBLE,
    pickup_long DOUBLE,
    destination_lat DOUBLE,
    destination_long DOUBLE,
    status ENUM('REQUESTED', 'ASSIGNED', 'IN_PROGRESS', 'COMPLETED', 'CANCELLED'),
    fare DECIMAL(7,2)
);
```

### 4.4 Vehicles Table

```sql
CREATE TABLE Vehicles (
    vehicle_id VARCHAR PRIMARY KEY,
    model VARCHAR NOT NULL,
    license_plate VARCHAR UNIQUE NOT NULL,
    type ENUM('SEDAN', 'SUV', 'HATCHBACK', 'MOTORBIKE')
);
```

### 4.5 Payments Table

```sql
CREATE TABLE Payments (
    payment_id VARCHAR PRIMARY KEY,
    ride_id VARCHAR REFERENCES Rides(ride_id),
    amount DECIMAL(7,2),
    status ENUM('PENDING', 'COMPLETED', 'FAILED')
);
```

## 5. System Workflow

### 5.1 Ride Request Flow

1. **User requests a ride** by providing pickup and destination locations.
2. **System finds the nearest available driver** using real-time location data.
3. **Driver accepts the ride**, and both user and driver receive notifications.
4. **Ride starts**, and the system tracks real-time updates.
5. **Ride completes**, and fare is calculated dynamically.
6. **Payment is processed**, and ride status is updated.

### 5.2 Fare Calculation Considerations

- **Base Fare:** Fixed charge for initiating a ride.
- **Distance-Based Fare:** Calculated using GPS data.
- **Time-Based Fare:** Additional charges for ride duration.
- **Surge Pricing:** Adjusted based on demand and supply.

## 6. Optimizations and Trade-offs

- **Efficient Matching Algorithm:** Use **GeoHashing** for quick location-based driver matching.
- **Load Balancing:** Distribute requests across multiple servers for scalability.
- **Caching Frequently Accessed Data:** Use **Redis** for active ride lookups.
- **Handling High Traffic:** Implement **asynchronous processing** for ride requests.

## 7. Conclusion

This LLD provides a structured approach to implementing a **Ride-Sharing Application**, covering class design, database schema, and system workflows. The design ensures efficient user-driver matching, real-time ride tracking, and secure payment processing for a seamless ride experience.
