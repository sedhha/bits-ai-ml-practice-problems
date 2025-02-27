# Low-Level Design (LLD) for Notification System

## 1. Objective

Develop a system to send notifications to users via multiple channels (email, SMS, push notifications).

## 2. Key Considerations

- **Message Templates:** Standardize messages for different notification types.
- **User Preferences:** Allow users to configure their preferred notification channels.
- **Scheduling:** Support scheduled and real-time notifications.
- **Delivery Tracking:** Log delivery status and retries for failed messages.
- **Scalability:** Handle high volumes of notifications efficiently.

## 3. Class Design

### 3.1 Key Classes

```java
class NotificationSystem {
    private NotificationQueue queue;
    private DeliveryService deliveryService;

    public void sendNotification(Notification notification);
}

class Notification {
    private String notificationId;
    private User recipient;
    private NotificationType type;
    private String message;
    private List<Channel> channels;
    private LocalDateTime scheduledTime;

    public boolean schedule();
    public boolean send();
}

class User {
    private String userId;
    private String email;
    private String phoneNumber;
    private NotificationPreferences preferences;
}

class NotificationPreferences {
    private boolean emailEnabled;
    private boolean smsEnabled;
    private boolean pushEnabled;
}

enum NotificationType {
    TRANSACTIONAL, PROMOTIONAL, ALERT
}

enum Channel {
    EMAIL, SMS, PUSH_NOTIFICATION
}

class NotificationQueue {
    private Queue<Notification> pendingNotifications;
    public void add(Notification notification);
    public Notification getNext();
}

class DeliveryService {
    public void process(Notification notification);
    public boolean trackDelivery(String notificationId);
}
```

## 4. Database Schema

### 4.1 Users Table

```sql
CREATE TABLE Users (
    user_id VARCHAR PRIMARY KEY,
    email VARCHAR,
    phone_number VARCHAR,
    email_enabled BOOLEAN DEFAULT TRUE,
    sms_enabled BOOLEAN DEFAULT FALSE,
    push_enabled BOOLEAN DEFAULT TRUE
);
```

### 4.2 Notifications Table

```sql
CREATE TABLE Notifications (
    notification_id VARCHAR PRIMARY KEY,
    user_id VARCHAR REFERENCES Users(user_id),
    type ENUM('TRANSACTIONAL', 'PROMOTIONAL', 'ALERT'),
    message TEXT,
    scheduled_time TIMESTAMP,
    status ENUM('PENDING', 'SENT', 'FAILED')
);
```

### 4.3 Delivery Logs Table

```sql
CREATE TABLE DeliveryLogs (
    log_id SERIAL PRIMARY KEY,
    notification_id VARCHAR REFERENCES Notifications(notification_id),
    channel ENUM('EMAIL', 'SMS', 'PUSH_NOTIFICATION'),
    status ENUM('SUCCESS', 'FAILED', 'RETRY'),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 5. System Workflow

### 5.1 Notification Flow

1. **User triggers an event** requiring a notification.
2. **Notification is created** with user preferences and scheduling options.
3. **Queued for processing** based on priority and scheduled time.
4. **Delivery Service picks up notification**, sends it via preferred channels.
5. **Logs delivery status** and retries failed messages if needed.
6. **User receives the notification** via email, SMS, or push.

## 6. Optimizations and Trade-offs

- **Batch Processing:** Optimize delivery by grouping notifications.
- **Retry Mechanism:** Implement exponential backoff for failed messages.
- **Push vs Pull:** Use WebSockets for real-time push notifications.
- **Load Balancing:** Distribute notification processing across multiple servers.

## 7. Conclusion

This LLD provides a structured way to implement a **Notification System**, covering class design, database schema, and system workflows. The design ensures efficient notification delivery, tracking, and scalability for high-volume messaging systems.
