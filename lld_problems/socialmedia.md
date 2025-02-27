# Low-Level Design (LLD) for Social Media Feed

## 1. Objective

Build a system to display a user's feed with posts from friends and followed pages.

## 2. Key Considerations

- **Post Ranking Algorithm:** Determine the most relevant posts for a user.
- **Real-Time Updates:** Ensure users see fresh content.
- **Multimedia Support:** Handle images, videos, and text efficiently.
- **Scalability:** Support millions of users fetching posts concurrently.

## 3. Class Design

### 3.1 Key Classes

```java
class User {
    private String userId;
    private String name;
    private List<User> friends;
    private List<Page> followedPages;

    public List<Post> getFeed();
}

class Post {
    private String postId;
    private User author;
    private String content;
    private List<Media> media;
    private LocalDateTime timestamp;
    private int likes;
    private int comments;

    public void likePost(User user);
    public void addComment(Comment comment);
}

class Media {
    private String mediaId;
    private MediaType type;
    private String url;
}

enum MediaType {
    IMAGE, VIDEO, TEXT
}

class Comment {
    private String commentId;
    private User commenter;
    private String text;
    private LocalDateTime timestamp;
}

class FeedService {
    public List<Post> fetchUserFeed(User user);
    private List<Post> rankPosts(List<Post> posts, User user);
}
```

## 4. Database Schema

### 4.1 Users Table

```sql
CREATE TABLE Users (
    user_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL
);
```

### 4.2 Posts Table

```sql
CREATE TABLE Posts (
    post_id VARCHAR PRIMARY KEY,
    user_id VARCHAR REFERENCES Users(user_id),
    content TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    likes INT DEFAULT 0
);
```

### 4.3 Media Table

```sql
CREATE TABLE Media (
    media_id VARCHAR PRIMARY KEY,
    post_id VARCHAR REFERENCES Posts(post_id),
    url TEXT,
    type ENUM('IMAGE', 'VIDEO', 'TEXT')
);
```

### 4.4 Comments Table

```sql
CREATE TABLE Comments (
    comment_id VARCHAR PRIMARY KEY,
    post_id VARCHAR REFERENCES Posts(post_id),
    user_id VARCHAR REFERENCES Users(user_id),
    text TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 5. System Workflow

### 5.1 Feed Generation Process

1. **User requests feed** from the system.
2. **Fetch posts** from friends and followed pages.
3. **Apply ranking algorithm** based on recency, engagement, and user interactions.
4. **Return sorted feed** with multimedia and engagement counts.
5. **Render feed** on the user interface.

### 5.2 Ranking Algorithm Considerations

- **Recency:** Newer posts get higher priority.
- **Engagement:** Posts with more likes/comments rank higher.
- **User Interaction History:** Posts from frequently interacted friends appear first.
- **Trending Content:** Popular posts may get a boost.

## 6. Optimizations and Trade-offs

- **Caching:** Use Redis to store frequently accessed feeds.
- **Precomputed Feeds:** Periodically compute and store feeds for active users.
- **Load Balancing:** Distribute requests across multiple servers for scalability.
- **Delayed Media Loading:** Load text first, then fetch images/videos asynchronously.

## 7. Conclusion

This LLD provides a structured approach to implementing a **Social Media Feed**, covering class design, database schema, and system workflows. The design ensures an optimized and scalable experience for users engaging with social content.
