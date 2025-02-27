### **What is Apache Kafka?**

Apache Kafka is a **distributed event streaming platform** used for building **real-time data pipelines and streaming applications**. It enables high-throughput, fault-tolerant, and scalable messaging between systems.

#### **Key Features of Kafka:**

- **Publish-Subscribe Model:** Producers publish messages to topics, and consumers subscribe to them.
- **High Throughput:** Capable of handling millions of messages per second.
- **Durability & Fault Tolerance:** Messages are stored across multiple nodes, ensuring reliability.
- **Real-time Processing:** Enables event-driven architectures.

---

### **Using Kafka in the IRCTC Ticket Booking System**

IRCTC (Indian Railway Catering and Tourism Corporation) handles **millions of concurrent ticket booking requests**. Traditional synchronous database writes can lead to high latency and failed transactions when demand spikes. **Apache Kafka** can optimize the system by:

1. **Decoupling Booking Requests:** Kafka acts as a buffer between user requests and the ticket processing system.
2. **Handling High Throughput:** Kafka’s partitioning allows distributed processing of booking requests.
3. **Ensuring Precise Seat Allocation:** Messages are processed sequentially to prevent overbooking.
4. **Real-time Processing with NiFi & ElasticSearch:** Kafka streams ticket availability updates in real-time to Elasticsearch for quick searching and querying.

---

## **LLD for IRCTC Ticket Booking System using Kafka, NiFi, and Elasticsearch**

The system consists of multiple components interacting asynchronously to ensure **low latency and precise booking handling**.

### **1. Components Overview**

- **Kafka**: Message broker for handling booking requests.
- **Apache NiFi**: Data ingestion and processing pipeline.
- **Elasticsearch**: Fast indexing and querying for seat availability.
- **MySQL/PostgreSQL**: Final database for transaction persistence.
- **Microservices**: For managing user authentication, booking logic, and payment.

---

### **2. Class Design**

```java
class BookingRequest {
    private String requestId;
    private String userId;
    private String trainId;
    private String seatClass;
    private int numSeats;
    private BookingStatus status;
}

class BookingService {
    private KafkaProducer kafkaProducer;
    private SeatAvailabilityService seatAvailabilityService;

    public void initiateBooking(BookingRequest request) {
        if (seatAvailabilityService.isSeatAvailable(request.getTrainId(), request.getNumSeats())) {
            kafkaProducer.send("booking-requests", request);
        } else {
            throw new BookingException("No seats available");
        }
    }
}

class KafkaBookingConsumer {
    private KafkaConsumer kafkaConsumer;
    private PaymentService paymentService;

    public void processBooking() {
        ConsumerRecords<String, BookingRequest> records = kafkaConsumer.poll(Duration.ofMillis(100));
        for (ConsumerRecord<String, BookingRequest> record : records) {
            BookingRequest request = record.value();
            paymentService.processPayment(request);
        }
    }
}

class SeatAvailabilityService {
    private ElasticsearchClient elasticClient;

    public boolean isSeatAvailable(String trainId, int numSeats) {
        return elasticClient.queryAvailability(trainId) >= numSeats;
    }
}

class PaymentService {
    public void processPayment(BookingRequest request) {
        // Mock Payment Processing
        request.setStatus(BookingStatus.CONFIRMED);
        saveToDatabase(request);
    }

    private void saveToDatabase(BookingRequest request) {
        // Persist to MySQL/PostgreSQL
    }
}

enum BookingStatus {
    PENDING, CONFIRMED, FAILED
}
```

---

### **3. Database Schema**

#### **Bookings Table**

```sql
CREATE TABLE Bookings (
    booking_id VARCHAR PRIMARY KEY,
    user_id VARCHAR REFERENCES Users(user_id),
    train_id VARCHAR NOT NULL,
    seat_class VARCHAR NOT NULL,
    num_seats INT NOT NULL,
    status ENUM('PENDING', 'CONFIRMED', 'FAILED'),
    booking_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### **Seats Availability Table (Synced with Elasticsearch)**

```sql
CREATE TABLE SeatAvailability (
    train_id VARCHAR PRIMARY KEY,
    seat_class VARCHAR NOT NULL,
    available_seats INT
);
```

---

### **4. Kafka Topics & Events**

| **Topic**           | **Purpose**                                 |
| ------------------- | ------------------------------------------- |
| `booking-requests`  | Stores raw booking requests                 |
| `payment-confirmed` | Stores successful payment confirmations     |
| `booking-failed`    | Handles booking failures & refunds          |
| `seat-availability` | Streams real-time seat availability updates |

---

### **5. System Workflow**

#### **Step 1: User Requests Booking**

1. **User submits a booking request via API**.
2. **BookingService checks seat availability in Elasticsearch**.
3. If seats are available, **Kafka Producer publishes the request** to `booking-requests`.

#### **Step 2: Kafka Processing**

4. **Kafka Consumers process booking requests asynchronously**.
5. **PaymentService processes payments**.
6. If successful, **Kafka sends confirmation message** to `payment-confirmed`.

#### **Step 3: Updating Seat Availability**

7. **Apache NiFi consumes `payment-confirmed` messages** and updates:
   - **Elasticsearch** (for real-time seat search).
   - **Relational Database** (for transaction records).
8. **If the request fails, it gets published to `booking-failed`**.

---

### **6. Using Apache NiFi for Data Flow**

#### **NiFi Use Cases**

1. **Stream booking confirmations** from Kafka to Elasticsearch.
2. **Handle failed bookings** and trigger refund processes.
3. **Sync real-time seat availability** to Kafka topic `seat-availability`.

#### **NiFi Data Flow Example**

1. **KafkaConsumerProcessor** → Reads from `payment-confirmed`.
2. **ConvertToElasticsearchFormat** → Formats data.
3. **PutElasticsearchProcessor** → Indexes seat availability in Elasticsearch.

---

### **7. Optimizations and Trade-offs**

| **Optimization**                  | **Benefit**                   | **Trade-off**                     |
| --------------------------------- | ----------------------------- | --------------------------------- |
| **Kafka for Async Booking**       | Handles millions of requests  | Requires managing Kafka consumers |
| **Elasticsearch for Seat Lookup** | Fast queries for availability | Eventual consistency with DB      |
| **NiFi for Data Ingestion**       | Automated data pipelines      | Additional infrastructure needed  |
| **Sharded DB for Scalability**    | Faster reads & writes         | Complexity in data partitioning   |

---

### **8. Conclusion**

This **LLD for IRCTC Ticket Booking** integrates **Kafka, NiFi, and Elasticsearch** to handle **high traffic, real-time updates, and precise seat booking**.

- **Kafka ensures async processing** for handling peak loads.
- **NiFi manages data movement** between systems.
- **Elasticsearch provides fast seat availability lookups**.

This approach **minimizes latency** and ensures **fair seat allocation** even under extreme traffic conditions. 🚄🔥

Would you like to extend this with more failover mechanisms? 🚀
